"""Checks for asynchronous phase attribution and non-additive GPU timing."""
import gzip
import json
import tempfile
import contextlib
import types
import importlib.util
import unittest
from unittest.mock import patch
from pathlib import Path

from summarize_operator_trace import classify, summarize, union_us
from run_operator_experiment import make_plan


class AnalysisTests(unittest.TestCase):
    def test_recomputed_output_prefix_is_prefill_not_decode(self):
        path = Path(__file__).parent / 'operator_hooks/operator_hooks/__init__.py'
        spec = importlib.util.spec_from_file_location('operator_hooks_test', path)
        operator_hooks = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(operator_hooks)
        import torch
        class FakeRunner:
            def execute_model(self, scheduler_output):
                return None
        module = types.SimpleNamespace(GPUModelRunner=FakeRunner, __name__='fake_runner')
        operator_hooks.patch(module)
        labels = []
        @contextlib.contextmanager
        def capture(name):
            labels.append(name)
            yield
        new = types.SimpleNamespace(req_id='r', prompt_token_ids=[1] * 10,
                                    prefill_token_ids=[1] * 20, num_computed_tokens=12)
        sched = types.SimpleNamespace(finished_req_ids=set(), scheduled_new_reqs=[new],
            scheduled_cached_reqs=types.SimpleNamespace(req_ids=[], num_computed_tokens=[]),
            num_scheduled_tokens={'r': 4})
        with patch.object(torch.profiler, 'record_function', capture):
            FakeRunner().execute_model(sched)
        self.assertIn('phase=prefill', labels[0])

    def test_launch_correlation_not_gpu_timestamp_assigns_phase(self):
        events = [
            {'ph': 'X', 'cat': 'user_annotation', 'name': 'op_phase|phase=prefill|seqs=8|step=1',
             'ts': 0, 'dur': 50, 'pid': 100, 'tid': 7},
            {'ph': 'X', 'cat': 'cuda_runtime', 'name': 'cudaLaunchKernel',
             'ts': 10, 'dur': 1, 'pid': 100, 'tid': 7, 'args': {'correlation': 1}},
            {'ph': 'X', 'cat': 'cuda_runtime', 'name': 'cudaLaunchKernel',
             'ts': 20, 'dur': 1, 'pid': 100, 'tid': 7, 'args': {'correlation': 2}},
            # Both kernels execute after the CPU scope ended; they overlap on GPU.
            {'ph': 'X', 'cat': 'kernel', 'name': '_tq_fused_store_mse',
             'ts': 100, 'dur': 10, 'pid': 0, 'tid': 1, 'args': {'correlation': 1, 'device': 0}},
            {'ph': 'X', 'cat': 'kernel', 'name': '_tq_decode_stage1',
             'ts': 105, 'dur': 20, 'pid': 0, 'tid': 2, 'args': {'correlation': 2, 'device': 0}},
        ]
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / 'worker.pt.trace.json.gz'
            with gzip.open(path, 'wt') as f:
                json.dump({'traceEvents': events}, f)
            result = summarize(t)
            self.assertEqual(result['phase_matched_fraction'], 1)
            self.assertAlmostEqual(result['cumulative_gpu_event_ms'], .030)
            self.assertAlmostEqual(result['gpu_busy_union_ms'], .025)
            self.assertEqual({r['phase'] for r in result['categories']}, {'prefill'})

    def test_fused_and_standalone_categories_stay_distinct(self):
        self.assertEqual(classify('_kvarn_fused_decode_stage1'), 'kv_read_dequant_attention_fused')
        self.assertEqual(classify('_kvarn_dequant_blocks_kernel'), 'standalone_dequant')
        self.assertEqual(classify('_kvarn_build_packed_kv_kernel'), 'kv_read_pack_dequant_fused')
        self.assertEqual(union_us([(0, 10), (5, 20), (30, 40)]), 30)

    def test_plan_covers_every_case_once_then_repeats_full_batches(self):
        rows = [{'case_id': str(i)} for i in range(50)]
        plan = make_plan({'qasper': rows})
        first = [p for p in plan if p['repeat'] == 1]
        self.assertEqual([c for p in first for c in p['case_ids']], [r['case_id'] for r in rows])
        self.assertEqual(len(first[-1]['case_ids']), 2)
        self.assertEqual(len(plan), 13)
        self.assertTrue(all(len(p['case_ids']) == 8 for p in plan if p['repeat'] > 1))


if __name__ == '__main__':
    unittest.main()
