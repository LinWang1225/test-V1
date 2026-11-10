import re
from types import SimpleNamespace
import torch
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend,FlashAttentionImpl
STATE=SimpleNamespace(active=False,phase='idle',decode_position=0,last_scheduled=0,callback=None,layer_calls={},impls={},weight_provider=None)
class CaptureBackend(FlashAttentionBackend):
    @staticmethod
    def get_impl_cls(): return CaptureImpl
class CaptureImpl(FlashAttentionImpl):
    def forward(self,layer,query,key,value,kv_cache,attn_metadata,output,output_scale=None,output_block_scale=None):
        if attn_metadata is not None and STATE.active and STATE.phase=='decode':
            if not hasattr(self,'factorial_layer'):
                self.factorial_layer=int(re.search(r'layers\.(\d+)\.',layer.layer_name).group(1))
            index=self.factorial_layer
            STATE.impls[index]=self
            STATE.layer_calls[index]=STATE.layer_calls.get(index,0)+1
            if STATE.callback is not None and STATE.decode_position in (1,8,32,64):
                assert attn_metadata.max_query_len==1
                length=int(attn_metadata.seq_lens[0].item())
                blocks=attn_metadata.block_table[0,:(length+127)//128].long()
                pages=kv_cache.index_select(0,blocks)
                k=pages[:,0].reshape(-1,8,128)[:length].transpose(0,1)
                v=pages[:,1].reshape(-1,8,128)[:length].transpose(0,1)
                q=query[0].reshape(8,4,128)
                STATE.callback(index,STATE.decode_position,q,k,v,float(self.scale))
        return super().forward(layer,query,key,value,kv_cache,attn_metadata,output,output_scale,output_block_scale)
def install(callback=None,backend='capture'):
    STATE.callback=callback
    from vllm.v1.attention.backends.registry import register_backend,AttentionBackendEnum
    register_backend(AttentionBackendEnum.FLASH_ATTN,'native_backend.CaptureBackend')
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    original=GPUModelRunner.execute_model
    def execute(self,scheduler_output,*args,**kwargs):
        if STATE.active: STATE.last_scheduled=scheduler_output.total_num_scheduled_tokens
        return original(self,scheduler_output,*args,**kwargs)
    GPUModelRunner.execute_model=execute
