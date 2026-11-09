"""Keep FP16 prefill fixed while using each upstream compressed decode.

The shadow cache is a quality-control device, not a memory optimization. Decode
never reads it. KVarN native sink and incomplete-block staging remain intact.
"""
import re
from types import SimpleNamespace

import torch
from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl, FlashAttentionMetadata
from vllm.v1.attention.backends.kvarn_attn import KVarNAttentionBackend, KVarNAttentionImpl

STATE = SimpleNamespace(active=False, phase='decode', impls={}, last_scheduled=0,
                        last_graph_mode=None)


class MatchedPrefill:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_impl = FlashAttentionImpl(
            num_heads=self.num_heads, head_size=self.head_size, scale=self.scale,
            num_kv_heads=self.num_kv_heads, alibi_slopes=None, sliding_window=None,
            kv_cache_dtype='auto')

    def ensure_quality_state(self, layer, kv_cache):
        if not hasattr(self, 'quality_counts'):
            index = int(re.search(r'layers\.(\d+)\.', layer.layer_name).group(1))
            self.quality_counts = torch.zeros(3, device=kv_cache.device, dtype=torch.int64)
            STATE.impls[index] = self
        if STATE.phase == 'prefill' and not hasattr(self, 'prompt_cache'):
            self.prompt_cache = torch.empty(
                (kv_cache.shape[0], 2, 128, self.num_kv_heads, self.head_size),
                device=kv_cache.device, dtype=torch.float16)

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        if kv_cache.numel() == 0:
            return
        self.ensure_quality_state(layer, kv_cache)
        super().do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)
        self.quality_counts[1].add_(slot_mapping.shape[0])
        if STATE.phase == 'prefill':
            self.prompt_impl.do_kv_cache_update(
                layer, key, value, self.prompt_cache, slot_mapping)

    def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                output=None, output_scale=None, output_block_scale=None):
        if attn_metadata is None:
            return super().forward(layer, query, key, value, kv_cache, attn_metadata,
                                   output, output_scale, output_block_scale)
        self.ensure_quality_state(layer, kv_cache)
        if STATE.phase == 'prefill':
            # KVarN's builder must be able to flush completed prompt blocks
            # into the native cache before the next step, even though its
            # attention forward is bypassed during matched FP16 prefill.
            if isinstance(self, KVarNAttentionImpl):
                self._kv_cache_ref = kv_cache
            meta = FlashAttentionMetadata(
                num_actual_tokens=attn_metadata.num_actual_tokens,
                max_query_len=attn_metadata.max_query_len,
                query_start_loc=attn_metadata.query_start_loc,
                max_seq_len=attn_metadata.max_seq_len, seq_lens=attn_metadata.seq_lens,
                block_table=attn_metadata.block_table, slot_mapping=attn_metadata.slot_mapping,
                use_cascade=False, common_prefix_len=0, cu_prefix_query_lens=None,
                prefix_kv_lens=None, suffix_kv_lens=None)
            self.quality_counts[2].add_(attn_metadata.num_actual_tokens)
            return self.prompt_impl.forward(
                layer, query, key, value, self.prompt_cache, meta, output,
                output_scale, output_block_scale)
        assert attn_metadata.max_query_len == 1
        self.quality_counts[0].add_(1)
        return super().forward(layer, query, key, value, kv_cache, attn_metadata,
                               output, output_scale, output_block_scale)


class MatchedKVarNImpl(MatchedPrefill, KVarNAttentionImpl):
    pass


class MatchedKVarNBackend(KVarNAttentionBackend):
    @staticmethod
    def get_impl_cls():
        return MatchedKVarNImpl


def install(method):
    from vllm.v1.attention.backends.registry import register_backend, AttentionBackendEnum
    if method == 'turboquant_paper_k4v4':
        register_backend(AttentionBackendEnum.FLASH_ATTN, 'turboquant_backend.PaperTurboQuantBackend')
    else:
        assert method == 'kvarn_k4v4_g128'
        register_backend(AttentionBackendEnum.KVARN, 'matched_prefill.MatchedKVarNBackend')
    from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
    from vllm.config import CUDAGraphMode
    original = CudagraphDispatcher.dispatch

    def dispatch(self, *args, **kwargs):
        if STATE.active and STATE.phase == 'prefill':
            kwargs['valid_modes'] = {CUDAGraphMode.NONE}
        result = original(self, *args, **kwargs)
        if STATE.active:
            STATE.last_graph_mode = str(result[0])
        return result

    CudagraphDispatcher.dispatch = dispatch
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    execute = GPUModelRunner.execute_model

    def execute_model(self, scheduler_output, *args, **kwargs):
        if STATE.active:
            STATE.last_scheduled = scheduler_output.total_num_scheduled_tokens
        return execute(self, scheduler_output, *args, **kwargs)

    GPUModelRunner.execute_model = execute_model
