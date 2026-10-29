"""ReadKV-style logical ledger and evidence-backed physical DRAM receipts."""
import csv,io,re,hashlib,json,os,time,subprocess,signal
from pathlib import Path
import torch
from common import ROOT,atomic,sha
NCU='/usr/local/cuda-12.6/bin/ncu'


def run_bounded(command,timeout):
    child=subprocess.Popen(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        output,_=child.communicate(timeout=timeout)
        return subprocess.CompletedProcess(command,child.returncode,output)
    except subprocess.TimeoutExpired:
        try:os.killpg(child.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        output,_=child.communicate(timeout=10)
        raise TimeoutError('Profiler timeout; its entire process group was stopped. '+output[-2000:])


class Ledger:
    def __init__(self,device='cuda'):
        # layer, K/V, payload bits / packet bytes / depth2 / depth3 / depth4 / depth16
        self.values=torch.zeros((36,2,6),dtype=torch.int64,device=device)
        self.calls=0

    def record(self,layer,n,kdepth,vdepth):
        for side,depth,units,width,padded in [(0,kdepth,8*128,n,(n+7)//8),(1,vdepth,8*n,128,16)]:
            if isinstance(depth,int):
                counts=[0,0,0,0];counts[(2,3,4,16).index(depth)]=units
                hist=torch.tensor(counts,dtype=torch.int64,device=self.values.device)
            else:
                assert depth.numel()==units
                small=torch.bincount((depth.flatten()-2),minlength=3)
                assert len(small)==3
                hist=torch.cat((small,torch.zeros(1,dtype=torch.int64,device=depth.device)))
            bits=(hist*torch.tensor([2,3,4,16],device=hist.device)).sum()
            packet=(bits*width)//8 if isinstance(depth,int) and depth==16 else bits*padded
            self.values[layer,side]+=torch.cat(((bits*width)[None],packet[None],hist))
        self.calls+=1

    def finish(self,policy):
        a=self.values.cpu().tolist();per_layer=[]
        for l,sides in enumerate(a):
            per_layer.append(dict(layer=l,**{name:dict(payload_bits=v[0],packet_read_bytes=v[1],depth_counts=dict(zip(['2','3','4','16'],v[2:]))) for name,v in zip(['K','V'],sides)}))
        totals={}
        for s,name in enumerate(['K','V']):
            total=[sum(a[l][s][j] for l in range(36)) for j in range(6)]
            totals[name]=dict(payload_bits=total[0],packet_read_bytes=total[1],depth_counts=dict(zip(['2','3','4','16'],total[2:])))
        # Conservative single reload of the actual reference decoder input state.
        # These are analytical allowances, NOT physical counter observations.
        decoder=0 if policy=='fp16' else self.calls*(2*8*128*2*4+2*8*3*16*4)
        curves=self.calls*2*8*128*3*8 if policy=='query_curve3' else 0
        transform=0 if policy=='fp16' else self.calls*2*128*128*4
        return dict(K=totals['K'],V=totals['V'],per_layer=per_layer,decode_attention_calls=self.calls,decoder_metadata_read_allowance_bytes=decoder,controller_curve_read_allowance_bytes=curves,reference_dense_rotation_read_allowance_bytes=transform,logical_total_bytes=sum(v['packet_read_bytes'] for v in totals.values())+decoder+curves+transform,scope='actual decode calls; packet payload plus conservative shared-state reload allowance; excludes prefill, model weights, cache writes, schedules and workspace',reference_code_storage_bits_per_scalar=16 if policy=='fp16' else 8,intended_progressive_storage_bits_per_scalar=None if policy=='fp16' else 4)

    def sample(self,layer,policy,n):
        data=self.values[layer].cpu().tolist()
        sides={name:dict(payload_bits=v[0],packet_read_bytes=v[1],depth_counts=dict(zip(['2','3','4','16'],v[2:]))) for name,v in zip(['K','V'],data)}
        metadata=0 if policy=='fp16' else 19456
        curves=49152 if policy=='query_curve3' else 0
        transform=0 if policy=='fp16' else 131072
        return dict(layer=layer,decode_position=1,context=n,**sides,decoder_metadata_read_allowance_bytes=metadata,controller_curve_read_allowance_bytes=curves,reference_dense_rotation_read_allowance_bytes=transform,logical_total_bytes=sum(x['packet_read_bytes'] for x in sides.values())+metadata+curves+transform,scope='the first actual attention call in this layer, matching the hardware fixture')


def parse_ncu(path):
    lines=Path(path).read_text(errors='replace').splitlines()
    start=next((i for i,s in enumerate(lines) if 'Metric Name' in s and 'Metric Value' in s and 'ID' in s),None)
    if start is None:raise ValueError('No Nsight raw metric CSV header; no physical measurement')
    records={};wanted={'dram__bytes_read.sum':'read','dram__bytes_write.sum':'write'}
    for row in csv.DictReader(io.StringIO('\n'.join(lines[start:]))):
        name=row.get('Metric Name','')
        if name not in wanted:continue
        unit=row.get('Metric Unit','').strip().lower()
        if unit not in ('byte','bytes'):raise ValueError(f'Expected base byte units, got {unit!r}')
        value=row.get('Metric Value','').replace(',','').strip()
        try:value=float(value)
        except ValueError:raise ValueError(f'Unavailable counter {name}: {value}')
        if not value>=0 or not value.is_integer():raise ValueError('Invalid DRAM byte count')
        key=(row.get('Process ID',''),row['ID']);metric=wanted[name]
        bucket=records.setdefault(key,{})
        if metric in bucket:raise ValueError('Duplicate metric row; do not double-count replay/export')
        bucket[metric]=int(value)
    if not records or any(set(v)!=set(['read','write']) for v in records.values()):
        raise ValueError('Missing read/write counters for profiled launches')
    read=sum(v['read'] for v in records.values());write=sum(v['write'] for v in records.values())
    return dict(dram_read_bytes=read,dram_write_bytes=write,dram_total_bytes=read+write,profiled_kernel_launches=len(records),raw_csv_sha256=sha(path),measurement='hardware DRAM counters, complete eager attention including intermediates/output, not address-attributed K/V-only reads')


def ncu_command(fixture,condition,base):
    return [NCU,'--metrics','dram__bytes_read.sum,dram__bytes_write.sum','--nvtx','--nvtx-include','readkv_attention/','--replay-mode','application','--app-replay-mode','strict','--cache-control','none','--clock-control','none','--csv','--page','raw','--print-units','base','--log-file',str(base.with_suffix('.csv')),'--export',str(base),str(ROOT.parents[1]/'.venv/bin/python'),'-u',str(ROOT/'src/profile_fixture.py'),'--fixture',str(fixture),'--condition',condition]


def execute_fixture(fixture,condition,base):
    base.parent.mkdir(parents=True,exist_ok=True)
    # A completed report is immutable and can be resumed; stale failed files
    # are retained with separate retry names by the caller.
    command=ncu_command(fixture,condition,base)
    p=run_bounded(command,120)
    base.with_suffix('.log').write_text(p.stdout)
    raw=base.with_suffix('.csv')
    if p.returncode:
        msg=p.stdout+(raw.read_text(errors='replace') if raw.exists() else '')
        if 'ERR_NVGPUCTRPERM' in msg:raise PermissionError('NVIDIA hardware counters denied; privileged collector is required')
        raise RuntimeError(f'Nsight exit {p.returncode}: '+msg[-3000:])
    measured=parse_ncu(raw)
    measured.update(condition=condition,fixture_sha256=sha(fixture),raw_csv=str(raw),report=str(base.with_suffix('.ncu-rep')),command=command,complete=True,cache_control='application-managed warm/128MiB eviction; ncu cache-control none')
    atomic(base.with_suffix('.json'),measured)
    return measured


def collect(fixture,condition,base,timeout=150):
    receipt=base.with_suffix('.json')
    if receipt.exists():
        a=json.loads(receipt.read_text());assert a['fixture_sha256']==sha(fixture);return a
    # Use a running privileged service only when the driver denies user counters.
    service=ROOT/'collector_status.json'
    if service.exists() and time.time()-json.loads(service.read_text()).get('heartbeat_epoch',0)<30:
        job_id=hashlib.sha256(str(base.resolve()).encode()).hexdigest()[:24]
        job=ROOT/'collector_jobs'/f'{job_id}_{condition}.json'
        atomic(job,dict(fixture=str(fixture.resolve()),condition=condition,base=str(base.resolve())))
        until=time.time()+timeout
        while time.time()<until:
            if receipt.exists():
                a=json.loads(receipt.read_text());assert a['fixture_sha256']==sha(fixture);return a
            error=job.with_suffix('.error.json')
            if error.exists():raise RuntimeError(json.loads(error.read_text()))
            time.sleep(1)
        raise TimeoutError('Privileged collector did not finish the request')
    return execute_fixture(fixture,condition,base)
