"""Paper-faithful vector quantizer in real vLLM generation; quality only."""
import re
import numpy as np
import torch
from common import ROOT
from matched_prefill import STATE
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend, FlashAttentionImpl
from turboquant_kernel import attention


class PaperTurboQuantBackend(FlashAttentionBackend):
    @staticmethod
    def get_impl_cls():
        return PaperTurboQuantImpl


class PaperTurboQuantImpl(FlashAttentionImpl):
    def ensure(self, layer, kv_cache):
        if not hasattr(self, 'parameters'):
            with np.load(ROOT/'references/turboquant_parameters.npz') as data:
                self.parameters={k:torch.tensor(data[k],device=kv_cache.device,dtype=torch.float32) for k in data.files}
            for b in (3,4):
                centers=self.parameters[f'C{b}']
                self.parameters[f'B{b}']=(centers[:-1]+centers[1:])/2
            self.packed=torch.empty((kv_cache.shape[0],2,128,8,64),device=kv_cache.device,dtype=torch.uint8)
            self.norms=torch.empty((kv_cache.shape[0],3,128,8),device=kv_cache.device,dtype=torch.float32)
            self.quality_counts=torch.zeros(3,device=kv_cache.device,dtype=torch.int64)
            index=int(re.search(r'layers\.(\d+)\.',layer.layer_name).group(1))
            STATE.impls[index]=self

    def do_kv_cache_update(self,layer,key,value,kv_cache,slot_mapping):
        if kv_cache.numel()==0:return
        self.ensure(layer,kv_cache)
        if STATE.phase=='prefill':
            super().do_kv_cache_update(layer,key,value,kv_cache,slot_mapping)
        n=slot_mapping.shape[0]
        p=self.parameters
        k=key[:n].view(n,8,128).float();v=value[:n].view(n,8,128).float()
        kn=k.norm(dim=-1,keepdim=True);vn=v.norm(dim=-1,keepdim=True)
        kr=(k@p['RK'].T)/kn.clamp_min(1e-30)
        vr=(v@p['RV'].T)/vn.clamp_min(1e-30)
        ki=torch.bucketize(kr.contiguous(),p['B3'])
        vi=torch.bucketize(vr.contiguous(),p['B4'])
        residual=kr-p['C3'][ki]
        gamma=residual.norm(dim=-1)
        signs=(residual@p['S'].T)>=0
        code=(ki|(signs.long()<<3)).to(torch.uint8)
        packed_k=code[...,::2]|(code[...,1::2]<<4)
        vi=vi.to(torch.uint8)
        packed_v=vi[...,::2]|(vi[...,1::2]<<4)
        block=slot_mapping//128;offset=slot_mapping%128
        valid=slot_mapping>=0
        # Slot padding is never used as a valid cache write.
        # Static graph batches here are C1, hence all one-token slots valid.
        # Masked padding uses an existing null slot rather than negative indices.
        safe_block=torch.where(valid,block,0);safe_offset=torch.where(valid,offset,0)
        self.packed[safe_block,0,safe_offset]=packed_k
        self.packed[safe_block,1,safe_offset]=packed_v
        self.norms[safe_block,0,safe_offset]=kn[...,0]
        self.norms[safe_block,1,safe_offset]=gamma
        self.norms[safe_block,2,safe_offset]=vn[...,0]
        self.quality_counts[1].add_(n)

    def forward(self,layer,query,key,value,kv_cache,attn_metadata,output,
                output_scale=None,output_block_scale=None):
        if attn_metadata is None:return output.fill_(0)
        self.ensure(layer,kv_cache)
        if STATE.phase=='prefill':
            self.quality_counts[2].add_(attn_metadata.num_actual_tokens)
            return super().forward(layer,query,key,value,kv_cache,attn_metadata,output,output_scale,output_block_scale)
        assert attn_metadata.max_query_len==1
        n=attn_metadata.num_actual_tokens;p=self.parameters
        q=query[:n].view(n,32,128).float()@p['RK'].T
        qjlq=q@p['S'].T
        rotated=attention(q,qjlq,self.packed,self.norms,attn_metadata.block_table,
                          attn_metadata.seq_lens,p['C3'],p['C4'],self.scale)
        result=(rotated@p['RV']).to(query.dtype)
        output[:n].view(n,32,128).copy_(result)
        self.quality_counts[0].add_(1)
        return output
