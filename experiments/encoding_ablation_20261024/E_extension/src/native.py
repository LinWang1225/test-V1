"""Call frozen, unmodified native KVarN quantization and reconstruction sources."""
import ast,functools,importlib.util,math,sys,os
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m
# Native store imports this exact module name; reuse snapshot without editing source.
sink=load('vllm.model_executor.layers.quantization.kvarn.sinkhorn',ROOT/'references/sinkhorn.py')
store=load('native_kvarn_store',ROOT/'references/kvarn_store.py')
decode=load('native_kvarn_decode',ROOT/'references/kvarn_decode.py')
triton_sink=load('native_kvarn_triton_sinkhorn',ROOT/'references/triton_kvarn_sinkhorn.py')
# Execute only the unchanged native Hadamard helper, avoiding the full serving backend.
tree=ast.parse((ROOT/'references/kvarn_attn.py').read_text())
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_hadamard_cached')
namespace=dict(torch=torch,math=math,functools=functools)
exec(compile(ast.Module(body=[node],type_ignores=[]),str(ROOT/'references/kvarn_attn.py'),'exec'),namespace)
hadamard=namespace['_hadamard_cached']

class NativeCodec:
    def __init__(self):
        os.environ['KVARN_RTN_QUANTILE']='0'
        self.H=hadamard(128,'cuda');self.H16=self.H.half()
    def quantize(self,k,v):
        """Raw tensors [8,T,128]; completed native tiles only, no sink protection."""
        h,n,d=k.shape;full=n//128*128;nb=full//128
        kr=(k[:,:full].half()@self.H16).float();vr=(v[:,:full].half()@self.H16).float()
        kt=kr.reshape(h,nb,128,d).transpose(-1,-2).contiguous().reshape(-1,d,128)
        vt=vr.reshape(h*nb,128,d).contiguous()
        bal,sc,sr=triton_sink.kvarn_sinkhorn_triton(torch.cat((kt,vt)),iterations=8)
        ntiles=h*nb
        ks=store.kvarn_store_tile_k_batch_from_sinkhorn(bal[:ntiles],sc[:ntiles],sr[:ntiles],4)
        vs=store.kvarn_store_tile_v_batch_from_sinkhorn(bal[ntiles:],sc[ntiles:],sr[ntiles:],4)
        qk=decode._unpack_4bit(ks['q_packed_uint8'],128).long()
        qv=decode._unpack_4bit(vs['q_packed_uint8'],128).long()
        return dict(ks=ks,vs=vs,qk=qk,qv=qv,k=k,v=v,full=full,nb=nb)
    def reconstruct(self,data,bits,midpoint=True,omit=0):
        full=data['full'];h,n,d=data['k'].shape;nb=data['nb']
        def code(q):
            size=full*d
            complete=omit//size if omit else 0
            partial=omit%size if omit else 0
            kept=max(0,4-complete)
            use=min(bits,kept)
            def prefix(c,b):
                return (c>>(4-b)).float()*2**(4-b)+((2**(4-b)-1)/2 if midpoint else 0)
            a=prefix(q,use)
            if partial:
                flat=a.view(h,-1);base=q.view(h,-1)[:,:partial]
                flat[:,:partial]=prefix(base,min(bits,max(0,kept-1)))
            return a
        ks,vs=data['ks'],data['vs'];qk=code(data['qk']);qv=code(data['qv'])
        kk=(qk*ks['s_col_K'].float()[:,:,None]+ks['zp_K'].float()[:,:,None])*ks['s_row_K'].float()[:,None,:]
        vv=(qv*vs['s_row_V'].float()[:,:,None]+vs['zp_V'].float()[:,:,None])*vs['s_col_V'].float()[:,None,:]
        kr=kk.reshape(h,nb,d,128).transpose(-1,-2).reshape(h,full,d)
        vr=vv.reshape(h,full,d)
        # Full read is native FP32 dequant; invert rotation before canonical-error measurement.
        k=torch.cat((kr@self.H.T,data['k'][:,full:].float()),1)
        v=torch.cat((vr@self.H.T,data['v'][:,full:].float()),1)
        return torch.stack((k,v))
    @staticmethod
    def storage(n,layers=36):
        full=n//128*128;tail=n-full;blocks=full//128;h=8;d=128
        payload_per_head=full*d # two sides at 4bits
        meta_per_head=blocks*(3*d+3*128)*2
        tail_per_head=tail*d*2*2
        baseline_per_head=n*d+1024+(128+86)/(layers*h)
        delta=max(0.,payload_per_head+meta_per_head+tail_per_head-baseline_per_head)
        # Byte-aligned equal omission on K and V, identical per head/layer.
        omit=int(math.ceil(delta/2))*8
        if omit>4*full*d:raise RuntimeError('Cannot fit E under B budget even without quantized payload')
        e=(payload_per_head+meta_per_head+tail_per_head)*layers*h
        budget=int(n*d*layers*h+1024*layers*h+128+86)
        equal=e-(omit//8)*2*layers*h
        assert equal<=budget
        return dict(native_tail_pool_capacity_bytes=(128*d*2*2*layers*h if tail else 0),logical_read_bytes={str(b):int(full*d*2*b//8*layers*h+meta_per_head*layers*h+tail_per_head*layers*h) for b in (2,3,4)},context=n,full_tokens=full,tail_tokens=tail,stored_payload_bytes=payload_per_head*layers*h,native_metadata_bytes=meta_per_head*layers*h,tail_fp16_bytes=tail_per_head*layers*h,total_bytes=e,B_total_budget_bytes=budget,equal_total_bytes=equal,equal_removed_bits_per_head_side=omit,equal_complete_removed_planes=omit//(full*d),equal_partial_next_plane_scalars=omit%(full*d),minimum_stored_prefix_bits=max(0,4-int(math.ceil(omit/(full*d)))))
