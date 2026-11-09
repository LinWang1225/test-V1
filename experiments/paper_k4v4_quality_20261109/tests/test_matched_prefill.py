"""CPU routing tests; actual native GPU validation runs after acquiring the lock."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]


class Native:
    def __init__(self, **kwargs):
        self.num_heads = 32
        self.num_kv_heads = 8
        self.head_size = 128
        self.scale = 128 ** -.5
        self.events = []

    def do_kv_cache_update(self, *args):
        self.events.append('native_store')

    def forward(self, *args):
        self.events.append('native_attention')
        return 'native_result'


class FakeFlash(Native):
    def do_kv_cache_update(self, *args):
        self.events.append('fp16_store')

    def forward(self, *args):
        self.events.append('fp16_attention')
        return 'fp16_result'


class TestRouting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fake = {}
        modules = {
            'vllm.v1.attention.backends.flash_attn': dict(FlashAttentionImpl=FakeFlash, FlashAttentionMetadata=SimpleNamespace),
            'vllm.v1.attention.backends.kvarn_attn': dict(KVarNAttentionBackend=type('KB', (), {}), KVarNAttentionImpl=type('KI', (Native,), {})),
            'vllm.v1.attention.backends.turboquant_attn': dict(TurboQuantAttentionBackend=type('TB', (), {}), TurboQuantAttentionImpl=type('TI', (Native,), {}))}
        for name, attrs in modules.items():
            module = ModuleType(name)
            module.__dict__.update(attrs)
            fake[name] = module
        spec = importlib.util.spec_from_file_location('tested_matched_prefill', ROOT / 'src/matched_prefill.py')
        cls.module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, fake):
            spec.loader.exec_module(cls.module)

    def exercise(self, native_class):
        module = self.module
        impl = native_class()
        impl.quality_counts = torch.zeros(3, dtype=torch.int64)
        impl.prompt_cache = torch.empty((4, 2, 128, 8, 128))
        impl.ensure_quality_state = lambda *args: None
        layer = SimpleNamespace(layer_name='model.layers.0.self_attn')
        cache = torch.zeros(4, 1, dtype=torch.uint8)
        kv = torch.zeros(1, 8, 128)
        q = torch.zeros(1, 32, 128)
        meta = SimpleNamespace(num_actual_tokens=1, max_query_len=1,
            query_start_loc=torch.tensor([0,1]), max_seq_len=513,
            seq_lens=torch.tensor([513]), block_table=torch.tensor([[0,1,2,3]]),
            slot_mapping=torch.tensor([128]))
        module.STATE.phase = 'prefill'
        impl.do_kv_cache_update(layer, kv, kv, cache, meta.slot_mapping)
        result = impl.forward(layer, q, kv, kv, cache, meta)
        self.assertEqual(result, 'fp16_result')
        self.assertEqual(impl.events, ['native_store'])
        self.assertEqual(impl.prompt_impl.events, ['fp16_store', 'fp16_attention'])
        self.assertEqual(impl.quality_counts.tolist(), [0, 1, 1])
        if isinstance(impl, module.MatchedKVarNImpl):
            self.assertIs(impl._kv_cache_ref, cache)
        module.STATE.phase = 'decode'
        impl.do_kv_cache_update(layer, kv, kv, cache, meta.slot_mapping)
        self.assertEqual(impl.forward(layer, q, kv, kv, cache, meta), 'native_result')
        self.assertEqual(impl.events, ['native_store', 'native_store', 'native_attention'])
        self.assertEqual(impl.prompt_impl.events, ['fp16_store', 'fp16_attention'])
        self.assertEqual(impl.quality_counts.tolist(), [1, 2, 1])

    def test_kvarn_one_token_prefill_and_native_decode(self):
        self.exercise(self.module.MatchedKVarNImpl)


if __name__ == '__main__':
    unittest.main()
