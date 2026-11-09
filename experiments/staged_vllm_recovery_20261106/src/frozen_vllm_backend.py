"""vLLM attention plugin: FP16 prefill and frozen-tree uint8 reference decode."""
import os,re
from types import SimpleNamespace
import numpy as np
import torch
from common import ROOT
from prefix_codec import hadamard
from tree_kernels import store,decode
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend,FlashAttentionImpl

STATE=SimpleNamespace(codec=None,phase='decode',active=False,impls={},last_graph_mode=None,
                      last_scheduled=0,scheduled_history=[])


class FrozenTreeBackend(FlashAttentionBackend):
    @staticmethod
    def get_impl_cls():return FrozenTreeImpl


class FrozenTreeImpl(FlashAttentionImpl):
    def ensure(self,layer,cache):
        if not hasattr(self,'book'):
            index=int(re.search(r'layers\.(\d+)\.',layer.layer_name).group(1))
            with np.load(ROOT/'codebooks'/STATE.codec/f'layer{index}.npz') as a:
                self.book={k:torch.tensor(a[k].astype(np.float32),device=cache.device) for k in a.files if k not in ('D','J')}
            if 'lo' in self.book:
                self.book['c4']=self.book['lo'][...,None]+torch.arange(16,device=cache.device,dtype=torch.float32)/15*(self.book['hi']-self.book['lo'])[...,None]
            self.U=torch.tensor(hadamard(),device=cache.device,dtype=torch.float32)
            self.counts=torch.zeros(2,device=cache.device,dtype=torch.int64)
            self.index=index;STATE.impls[index]=self
        if not hasattr(self,'packed'):
            self.packed=torch.empty((cache.shape[0],2,cache.shape[2],8,128),device=cache.device,dtype=torch.uint8)

    def do_kv_cache_update(self,layer,key,value,kv_cache,slot_mapping):
        if kv_cache.numel()==0:return
        self.ensure(layer,kv_cache)
        # The standard FP16 pages serve prefill only. Decode never reads them.
        if STATE.phase=='prefill':
            super().do_kv_cache_update(layer,key,value,kv_cache,slot_mapping)
        n=slot_mapping.shape[0]
        k=key[:n].view(n,8,128).float()@self.U
        v=value[:n].view(n,8,128).float()@self.U
        store(k,v,slot_mapping,self.packed,self.book,self.counts)

    def forward(self,layer,query,key,value,kv_cache,attn_metadata,output,output_scale=None,output_block_scale=None):
        if attn_metadata is None:return output.fill_(0)
        self.ensure(layer,kv_cache)
        if STATE.phase=='prefill':
            return super().forward(layer,query,key,value,kv_cache,attn_metadata,output,output_scale,output_block_scale)
        assert attn_metadata.max_query_len==1,'Decode plugin requires one live query per request'
        n=attn_metadata.num_actual_tokens
        q=query[:n].view(n,32,128).float()@self.U
        rotated=decode(q,self.packed,attn_metadata.block_table,attn_metadata.seq_lens,
                       self.book,self.counts,self.scale)
        out=(rotated@self.U.T).to(query.dtype)
        output[:n].view(n,32,128).copy_(out)
        return output


def install(codec):
    STATE.codec=codec
    if codec!='fp16':
        from vllm.v1.attention.backends.registry import register_backend,AttentionBackendEnum
        register_backend(AttentionBackendEnum.FLASH_ATTN,'frozen_vllm_backend.FrozenTreeBackend')
    # Prefill is always native FP16, including a one-token final chunk.
    # Live decode keeps the vLLM CUDA graph path; warmup/capture is unaffected.
    from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
    from vllm.config import CUDAGraphMode
    original=CudagraphDispatcher.dispatch
    def dispatch(self,*args,**kwargs):
        if STATE.active and STATE.phase=='prefill':kwargs['valid_modes']={CUDAGraphMode.NONE}
        result=original(self,*args,**kwargs)
        if STATE.active:STATE.last_graph_mode=str(result[0])
        return result
    CudagraphDispatcher.dispatch=dispatch
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    original_execute=GPUModelRunner.execute_model
    def execute(self,scheduler_output,*args,**kwargs):
        if STATE.active:
            STATE.last_scheduled=scheduler_output.total_num_scheduled_tokens
            STATE.scheduled_history.append(STATE.last_scheduled)
        return original_execute(self,scheduler_output,*args,**kwargs)
    GPUModelRunner.execute_model=execute
