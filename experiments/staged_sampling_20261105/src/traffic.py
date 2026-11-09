"""Hardware counters on the actual closed-loop generation, executed once."""
import csv
import io
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from common import ROOT, atomic, sha

NCU='/usr/local/cuda-12.6/bin/ncu'


def run_bounded(command, timeout):
    child=subprocess.Popen(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        output,_=child.communicate(timeout=timeout)
        return subprocess.CompletedProcess(command,child.returncode,output)
    except subprocess.TimeoutExpired:
        try:os.killpg(child.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        output,_=child.communicate(timeout=10)
        raise TimeoutError('Counter probe timeout: '+output[-2000:])


def metric_rows(path):
    lines=Path(path).read_text(errors='replace').splitlines()
    start=next(i for i,s in enumerate(lines) if s.startswith('"ID"'))
    reader=csv.DictReader(io.StringIO('\n'.join(lines[start:])))
    if 'Metric Name' in reader.fieldnames:
        yield from reader
    else:
        units=next(reader)
        for row in reader:
            if not row.get('ID'):continue
            for name in ('dram__bytes_read.sum','dram__bytes_write.sum'):
                yield dict(row,**{'Metric Name':name,'Metric Unit':units[name],'Metric Value':row[name]})


def parse_ncu(path):
    records={}
    for row in metric_rows(path):
        key=row['Metric Name']
        if key not in ('dram__bytes_read.sum','dram__bytes_write.sum'):continue
        assert row['Metric Unit'].strip().lower() in ('byte','bytes')
        value=float(row['Metric Value'].replace(',',''))
        assert value>=0 and value.is_integer()
        bucket=records.setdefault((row.get('Process ID',''),row['ID']),{})
        assert key not in bucket
        bucket[key]=int(value)
    assert records and all(len(r)==2 for r in records.values())
    read=sum(r['dram__bytes_read.sum'] for r in records.values())
    write=sum(r['dram__bytes_write.sum'] for r in records.values())
    return dict(dram_read_bytes=read,dram_write_bytes=write,dram_total_bytes=read+write,
                profiled_kernel_launches=len(records),raw_csv_sha256=sha(path))


def parse_live(path, generation):
    grouped={}
    for row in metric_rows(path):
        labels=' '.join(str(v) for k,v in row.items() if 'Range' in k)
        stage=next((s for s in ('prefill','decode') if 'generation_'+s in labels),None)
        if stage is None:raise ValueError('Counter range does not match live generation')
        assert row['Metric Unit'].strip().lower() in ('byte','bytes')
        metric={'dram__bytes_read.sum':'dram_read_bytes','dram__bytes_write.sum':'dram_write_bytes'}[row['Metric Name']]
        bucket=grouped.setdefault(stage,{})
        assert metric not in bucket,'Duplicate phase counter, not a single execution'
        value=float(row['Metric Value'].replace(',',''))
        assert value>=0 and value.is_integer()
        bucket[metric]=int(value)
    assert 'prefill' in grouped and len(grouped['prefill'])==2
    if generation['decode_attention_calls']:
        assert 'decode' in grouped and len(grouped['decode'])==2
    else:
        grouped.setdefault('decode',dict(dram_read_bytes=0,dram_write_bytes=0,no_decode_call=True))
    for values in grouped.values():values['dram_total_bytes']=values['dram_read_bytes']+values['dram_write_bytes']
    total={k:sum(v[k] for v in grouped.values()) for k in ('dram_read_bytes','dram_write_bytes','dram_total_bytes')}
    return dict(**total,stages=grouped,profiled_ranges=len(grouped),raw_csv_sha256=sha(path))


def execute_fixture(fixture,condition,requested_base):
    assert condition=='actual_request'
    requested_base.parent.mkdir(parents=True,exist_ok=True)
    base=requested_base
    # Incomplete attempts remain evidence. A restarted request gets fresh files.
    if any(base.parent.glob(base.name+'.*')):
        base=base.with_name(base.name+'_attempt_'+str(time.time_ns()))
    output=base.with_suffix('.generation.json')
    guard=base.with_suffix('.invocation.json')
    command=[NCU,'--metrics','dram__bytes_read.sum,dram__bytes_write.sum',
        '--nvtx','--nvtx-include','generation_prefill/','--nvtx-include','generation_decode/',
        '--replay-mode','app-range','--cache-control','none','--clock-control','none',
        '--csv','--page','raw','--print-units','base','--log-file',str(base.with_suffix('.csv')),
        str(ROOT.parents[1]/'.venv/bin/python'),'-u',
        str(ROOT/'src/actual_request.py'),'--request',str(fixture),'--output',str(output),'--invocation-guard',str(guard)]
    result=run_bounded(command,None)
    base.with_suffix('.log').write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f'Actual profiled generation failed ({result.returncode}): '+result.stdout[-6000:])
    generation=json.loads(output.read_text())
    invocation=json.loads(guard.read_text())
    assert invocation['execution_id']==generation['execution_id']
    raw=base.with_suffix('.csv')
    receipt=dict(status='measured',**parse_live(raw,generation),
        case_id=generation['case_id'],dataset=generation['dataset'],generated_tokens=generation['generated_tokens'],
        same_execution_as_quality=True,actual_generation_executions=1,execution_id=generation['execution_id'],
        generation_path=str(output),raw_generation_sha256=sha(output),fixture_sha256=sha(fixture),
        raw_csv=str(raw),report=None,binary_report_exported=False,command=command,complete=True,
        scope='actual single closed-loop end-to-end execution: complete prefill plus every decode step including thinking, weights, KV writes/reconstruction and sampling; not KV-only',
        measurement='live hardware counters, no reference trajectory or separate generation replay',
        counter_mode='app-range; exclusive invocation guard rejects any requested extra application pass',
        performance_scope='instrumented generation elapsed time, not unprofiled serving latency')
    atomic(base.with_suffix('.json'),receipt)
    if base!=requested_base:atomic(requested_base.with_suffix('.json'),receipt)
    return receipt


def collect(fixture,condition,base,timeout=None):
    receipt=base.with_suffix('.json')
    if receipt.exists():
        result=json.loads(receipt.read_text())
        assert result['fixture_sha256']==sha(fixture)
        assert sha(result['generation_path'])==result['raw_generation_sha256']
        return result
    service=ROOT/'collector_status.json'
    if not service.exists() or time.time()-json.loads(service.read_text()).get('heartbeat_epoch',0)>=30:
        raise RuntimeError('Live hardware collector unavailable; no unmeasured fallback')
    import hashlib
    job_id=hashlib.sha256(str(base.resolve()).encode()).hexdigest()[:24]
    job=ROOT/'collector_jobs'/f'{job_id}_{condition}.json'
    atomic(job,dict(fixture=str(fixture.resolve()),condition=condition,base=str(base.resolve())))
    while True:
        if receipt.exists():
            result=json.loads(receipt.read_text())
            assert result['fixture_sha256']==sha(fixture)
            assert sha(result['generation_path'])==result['raw_generation_sha256']
            return result
        error=job.with_suffix('.error.json')
        if error.exists():raise RuntimeError(json.loads(error.read_text()))
        if time.time()-json.loads(service.read_text()).get('heartbeat_epoch',0)>=30:
            raise RuntimeError('Live hardware collector heartbeat expired')
        time.sleep(1)
