"""KVarN unchanged completed tiles; valid-only partial-tile quantization, no FP16 KV."""
import sys,math
from pathlib import Path
import torch
P=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(P/'E_extension/src'))
from native import NativeCodec as Original,store,sink
class NativeCodec(Original):
    def quantize(self,k,v):
        a=super().quantize(k,v);full=a['full'];t=k.shape[1]-full
        a['partial']=None
        if not t:return a
        kr=(k[:,full:].half()@self.H16).float().transpose(-1,-2).contiguous()
        vr=(v[:,full:].half()@self.H16).float()
        def balance(x):
            if min(x.shape[1:])==1:
                return x,torch.ones(x.shape[0],1,x.shape[2],device=x.device),torch.ones(x.shape[0],x.shape[1],1,device=x.device)
            return sink.variance_normalize_batched(x,iterations=8)
        kb,kc,krw=balance(kr);vb,vc,vrw=balance(vr)
        qk,sk,zk=store._asymmetric_rtn_per_row(kb.reshape(-1,t),4)
        qk=qk.reshape_as(kb);sk=sk.reshape(8,128,1);zk=zk.reshape(8,128,1)
        qv,sv,zv=store._asymmetric_rtn_per_row(vb.reshape(-1,128),4)
        qv=qv.reshape_as(vb);sv=sv.reshape(8,t,1);zv=zv.reshape(8,t,1)
        a['partial']=dict(qk=qk,qv=qv,sk=(sk*krw).half(),zk=(zk*krw).half(),tk=kc.half(),sv=(sv*vrw).half(),zv=(zv*vrw).half(),cv=vc.half())
        return a
    def reconstruct(self,a,bits,midpoint=True,omit=0):
        assert omit==0,'No low-bit deletion in the complete-4bit main condition'
        out=super().reconstruct(a,bits,midpoint,0);p=a['partial']
        if p is not None:
            def prefix(q):return (q>>(4-bits)).float()*2**(4-bits)+((2**(4-bits)-1)/2 if midpoint else 0)
            kr=(prefix(p['qk'])*p['sk'].float()+p['zk'].float())*p['tk'].float()
            vr=(prefix(p['qv'])*p['sv'].float()+p['zv'].float())*p['cv'].float()
            out[0,:,a['full']:]=kr.transpose(-1,-2)@self.H.T
            out[1,:,a['full']:]=vr@self.H.T
        return out
    @staticmethod
    def storage(n,layers=36):
        blocks,t=divmod(n,128);streams=layers*8
        payload=(blocks*128*128+(128*math.ceil(t/2)+t*64 if t else 0))*streams
        meta=(blocks*1536+(6*128+6*t if t else 0))*streams
        return dict(context=n,full_tokens=n,completed_native_tile_tokens=n-t,partial_tile_tokens=t,tail_fp16_bytes=0,stored_payload_bytes=payload,valid_4bit_payload_bytes=n*128*streams,packing_padding_bytes=payload-n*128*streams,native_metadata_bytes=meta,total_bytes=payload+meta,all_valid_KV_bits=4)
