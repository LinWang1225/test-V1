"""Full quality generation with per-request logical and physical traffic status.

Hardware failure never suppresses generation or official scoring. The ReadKV
G.2 fixture represents one existing-cache call, not the whole generated trace.
"""
import sys,time,traceback
from common import *
import torch
import generate as base
from policy_gpu import keys,values
from traffic import Ledger,collect


class IOQualityModel(base.QualityModel):
    def __init__(self,policy,profile_enabled=True):
        self.profile_enabled=profile_enabled
        super().__init__(policy)
        original=self.original
        def dense(module,q,k,v,mask,**kw):
            out=original(module,q,k,v,mask,**kw)
            if not self.prefill and self.policy=='fp16':
                self.flow.record(module.layer_idx,k.shape[2],16,16)
                if module.layer_idx==17 and self.sample_logical is None:self.sample_logical=self.flow.sample(17,self.policy,k.shape[2])
                if self.capture_enabled and module.layer_idx==17 and self.fixture is None:
                    self.fixture=dict(q=q.detach().cpu(),k=k.detach().cpu(),v=v.detach().cpu(),expected_output=out[0].detach().cpu(),layer=17,context=k.shape[2],scale=kw['scaling'],policy=self.policy)
            return out
        self.original=dense

    def reset(self,capacity):
        super().reset(capacity);self.flow=Ledger();self.fixture=None;self.sample_logical=None
        hardware=json.loads((ROOT/'hardware_capability.json').read_text()) if (ROOT/'hardware_capability.json').exists() else {}
        service=ROOT/'collector_status.json'
        service_up=service.exists() and time.time()-json.loads(service.read_text()).get('heartbeat_epoch',0)<30
        self.capture_enabled=bool(self.profile_enabled and (hardware.get('available') or service_up))

    def attention(self,layer,q,n,scale):
        query=(q[0,:,0].float()@self.U).reshape(8,4,128)
        if self.policy=='full4':kd=vd=4;meanbits=4
        elif self.policy=='uniform3':kd=vd=3;meanbits=3
        else:kd=keys(query,self.curves[layer][0],scale,self.policy);vd=None;meanbits=3
        khat=self.decode(layer,n,0,kd)
        phat=(torch.einsum('hgd,htd->hgt',query,khat)*scale).softmax(-1);del khat
        if vd is None:vd=values(phat,self.curves[layer][1],self.policy)
        vhat=self.decode(layer,n,1,vd);out=((phat@vhat)@self.U.T).reshape(1,1,32,128).to(q.dtype)
        self.flow.record(layer,n,kd,vd)
        if layer==17 and self.sample_logical is None:self.sample_logical=self.flow.sample(layer,self.policy,n)
        self.attention_calls+=1;self.logical_read_bits+=2*8*n*128*meanbits
        self.padded_read_bytes+=meanbits*8*128*((n+7)//8)+meanbits*8*n*16
        if self.capture_enabled and layer==17 and self.fixture is None:
            self.fixture=dict(q=q.detach().cpu(),codes=self.codes[layer][:,:,:n].detach().cpu(),U=self.U.detach().cpu(),book={k:v.detach().cpu() for k,v in self.books[layer].items()},lut=self.lut[layer].detach().cpu(),curves=self.curves[layer].detach().cpu(),expected_output=out.detach().cpu(),layer=layer,context=n,scale=scale,policy=self.policy)
        return out

    def generate(self,row,limit=None):
        result=super().generate(row,limit)
        io_begin=time.perf_counter()
        result['traffic_schema_version']=2
        result['logical_traffic']=self.flow.finish(self.policy)
        logical=result['logical_traffic']
        assert logical['decode_attention_calls']==36*max(result['generated_tokens']-1,0)
        if self.policy!='fp16':
            assert logical['K']['payload_bits']+logical['V']['payload_bits']==result['logical_KV_read_bits']
        # The FP16 baseline no longer uses the old placeholder zero counter.
        result['logical_KV_read_bits']=logical['K']['payload_bits']+logical['V']['payload_bits']
        result['padded_plane_read_bytes']=logical['K']['packet_read_bytes']+logical['V']['packet_read_bytes']
        result['decode_attention_calls']=logical['decode_attention_calls']
        measured=dict(status='unavailable',dram_read_bytes=None,dram_write_bytes=None,dram_total_bytes=None,sample_scope='layer17 first actual decode attention, warm and evicted; not whole-generation traffic')
        measured['matching_sample_logical_traffic']=self.sample_logical
        if logical['decode_attention_calls']==0:
            measured.update(status='no_decode_call',reason='Natural EOS on first emitted token; no decode attention call exists')
        elif not self.capture_enabled:
            capability=json.loads((ROOT/'hardware_capability.json').read_text()) if (ROOT/'hardware_capability.json').exists() else {}
            measured['reason']=capability.get('reason','Hardware counter availability not established')
            measured['capability_probe']=capability
        else:
            assert self.fixture is not None,'Missing layer17 fixture for a decoded request'
            fixture=ROOT/'fixtures'/f'{self.policy}_{row["dataset"]}_{row["index"]:05d}.pt'
            fixture.parent.mkdir(exist_ok=True)
            self.fixture.update(case_id=row['case_id'],input_ids_sha256=row['input_ids_sha256'])
            torch.save(self.fixture,fixture)
            receipts={}
            try:
                for condition in ('warm','evicted'):
                    path=ROOT/'profiles'/self.policy/row['dataset']/f'{row["index"]:05d}_{condition}'
                    receipts[condition]=collect(fixture,condition,path)
                measured.update(status='measured',conditions=receipts,reason=None)
            except Exception as exc:
                measured.update(status='partial' if receipts else 'unavailable',reason=repr(exc),partial_conditions=receipts,error_traceback=traceback.format_exc())
                atomic(ROOT/'hardware_capability.json',dict(available=False,attempted=True,reason=repr(exc),checked_at=time.strftime('%Y-%m-%d %H:%M:%S'),run_without_hardware_measurement=True))
            finally:
                # Owned temporary fixture only; full raw hardware reports remain.
                fixture.unlink(missing_ok=True)
        result['physical_traffic']=measured
        result['traffic_postprocess_seconds']=time.perf_counter()-io_begin
        result['request_total_seconds']=result['elapsed_seconds']+result['traffic_postprocess_seconds']
        self.fixture=None
        return result


if __name__=='__main__':
    base.QualityModel=IOQualityModel
    base.main()
