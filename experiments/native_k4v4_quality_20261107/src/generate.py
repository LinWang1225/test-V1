"""Native closed-loop quality runs, sharing the frozen inputs and sampler."""
import os
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '0'
os.environ['KVARN_SINK_TOKENS'] = '128'
os.environ['KVARN_SINKHORN_ITERS'] = '8'
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')

import argparse
from collections import Counter
import importlib.util
import json
import time
import uuid

import torch
from common import ROOT, PARENT, MODEL, METHODS, GENERATION, atomic, cases, freeze, sha, status


def response(tokenizer, ids):
    # Reuse the exact parent's thinking/final-response split, without modifying it.
    spec = importlib.util.spec_from_file_location('frozen_response_split', PARENT / 'src/sampling.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.split_response(tokenizer, ids)


class Runner:
    def __init__(self, method):
        from matched_prefill import install, STATE
        from vllm import LLM
        self.state = STATE
        self.method = method
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        install(method)
        if method == 'kvarn_k4v4_g128':
            from vllm.model_executor.layers.quantization.kvarn.config import KVarNConfig
            config = KVarNConfig.from_cache_dtype(method, 128)
            page_bytes = 8 * config.tile_bytes_aligned
            self.native_config = dict(key_bits=config.key_bits, value_bits=config.value_bits,
                                      group=config.group, sink_tokens=config.sink_tokens,
                                      sinkhorn_iters=config.sinkhorn_iters)
            backend = 'KVARN'
        else:
            from vllm.model_executor.layers.quantization.turboquant.config import TurboQuantConfig
            config = TurboQuantConfig.from_cache_dtype(method, 128)
            page_bytes = 128 * 8 * config.slot_size_aligned
            self.native_config = dict(key_bits=config.key_quant_bits, value_bits=config.value_quant_bits,
                                      norm_correction=config.norm_correction)
            backend = 'TURBOQUANT'
        assert self.native_config['key_bits'] == self.native_config['value_bits'] == 4
        self.llm = LLM(
            model=str(MODEL), dtype='float16', kv_cache_dtype=method,
            max_model_len=40960, block_size=128, max_num_seqs=1,
            max_num_batched_tokens=512, kv_cache_memory_bytes=36 * 321 * page_bytes,
            gpu_memory_utilization=.8, enable_prefix_caching=False,
            enable_chunked_prefill=True, distributed_executor_backend='uni',
            async_scheduling=False, seed=GENERATION['seed'], generation_config='vllm',
            attention_config=dict(backend=backend), enforce_eager=False,
            compilation_config=dict(cudagraph_mode='FULL_DECODE_ONLY', cudagraph_capture_sizes=[1]),
            disable_log_stats=True)
        self.engine = self.llm.llm_engine
        self.tokenizer = self.llm.get_tokenizer()
        pool = self.engine.engine_core.engine_core.scheduler.kv_cache_manager.block_pool
        self.usable_tokens = (pool.num_gpu_blocks - 1) * 128
        assert self.usable_tokens >= 40960

    @torch.inference_mode()
    def generate(self, row, validation_max_tokens=None):
        from vllm import SamplingParams
        for impl in self.state.impls.values():
            impl.quality_counts.zero_()
        self.state.active = True
        self.state.phase = 'prefill'
        limit = 40960 - row['input_tokens']
        if validation_max_tokens is not None:
            limit = min(limit, validation_max_tokens)
        manifest = json.loads((PARENT / 'inputs/manifest.json').read_text())
        params = SamplingParams(
            temperature=.6, top_p=.95, top_k=20, min_p=0,
            seed=row['sampling_seed'], max_tokens=limit,
            stop_token_ids=manifest['eos_token_ids'], ignore_eos=False)
        execution_id = uuid.uuid4().hex
        self.engine.add_request('quality-' + execution_id,
                                dict(prompt_token_ids=row['input_ids']), params)
        remaining = row['input_tokens']
        decode_steps = 0
        finished = None
        modes = Counter()
        begin = time.perf_counter()
        while self.engine.has_unfinished_requests():
            self.state.phase = 'prefill' if remaining else 'decode'
            phase = self.state.phase
            self.state.last_scheduled = 0
            outputs = self.engine.step()
            scheduled = self.state.last_scheduled
            assert scheduled > 0
            if phase == 'prefill':
                assert scheduled <= remaining
                remaining -= scheduled
            else:
                assert scheduled == 1
                decode_steps += 1
                modes[self.state.last_graph_mode] += 1
            for out in outputs:
                if out.finished:
                    finished = out
            if decode_steps and decode_steps % 64 == 0:
                status('generating_quality', method=self.method, case=row['case_id'],
                       generated_tokens=decode_steps + 1)
        torch.cuda.synchronize()
        self.state.active = False
        assert finished is not None and remaining == 0
        output = finished.outputs[0]
        ids = list(output.token_ids)
        assert len(ids) in (decode_steps, decode_steps + 1)
        assert len(self.state.impls) == 36
        dispatch = {}
        for index, impl in self.state.impls.items():
            counts = impl.quality_counts.tolist()
            assert counts == [decode_steps, row['input_tokens'] + decode_steps, row['input_tokens']], (index, counts)
            dispatch[index] = dict(native_decode_calls=counts[0], encoded_tokens=counts[1],
                                   fp16_prefill_tokens=counts[2], impl=type(impl).__name__)
        reason = 'eos' if output.finish_reason == 'stop' else 'validation_limit' if validation_max_tokens else 'native_context_limit'
        return dict(
            index=row['index'], case_id=row['case_id'], dataset=row['dataset'], run=self.method,
            input_ids_sha256=row['input_ids_sha256'], input_tokens=row['input_tokens'],
            sampling_seed=row['sampling_seed'], generation_config=GENERATION,
            input_truncated=False, generated_ids=ids, generated_tokens=len(ids),
            **response(self.tokenizer, ids), finish_reason=reason, natural_completion=reason == 'eos',
            native_finish_reason=output.finish_reason, native_stop_reason=output.stop_reason,
            native_context_length=40960, artificial_output_limit=validation_max_tokens,
            backend='vllm', native_codec_config=self.native_config, layer_dispatch=dispatch,
            prefill='FP16 FlashAttention for every prompt chunk; shadow never read in decode',
            decode='Original native compressed-cache store and decode, native sink/tail semantics',
            physical_traffic=dict(status='not_measured', reason='User-requested quality-only comparison'),
            performance_claims=False, usable_kv_tokens=self.usable_tokens,
            decode_graph_modes=dict(modes), execution_id=execution_id,
            elapsed_seconds=time.perf_counter() - begin,
            completed_at=time.strftime('%Y-%m-%d %H:%M:%S'))

    def close(self):
        self.engine.model_executor.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=METHODS, required=True)
    args = parser.parse_args()
    frozen = freeze()
    rows = cases()
    method = args.method
    folder = ROOT / 'results' / method
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'scores').mkdir(exist_ok=True)
    for path in folder.glob('[0-9]*.json'):
        previous = json.loads(path.read_text())
        row = rows[previous['index']]
        assert previous['run_fingerprint'] == frozen['fingerprint']
        assert previous['input_ids_sha256'] == row['input_ids_sha256']
        assert previous['sampling_seed'] == row['sampling_seed']
    pending = [r for r in rows if not (folder / f"{r['index']:06d}.json").exists()
               and not (ROOT / 'failures' / method / f"{r['index']:06d}.json").exists()]
    if not pending:
        return
    # Checkpoint identifies the actual request if a subprocess crashes.
    atomic(ROOT / 'checkpoint.json', dict(method=method, stage='initializing'))
    status('initializing_quality', method=method)
    runner = Runner(method)
    try:
        validation = ROOT / 'validation' / (method + '.json')
        if not validation.exists():
            # Include short and long prompts, multi-chunk prefill and a final
            # one-token chunk. Validation outputs never enter formal scores.
            selected = list(dict.fromkeys([0, max(rows, key=lambda r:r['input_tokens'])['index'],
                                          next((r['index'] for r in rows if r['input_tokens'] % 512 == 1), 0)]))
            checks = []
            for index in selected:
                row = rows[index]
                result = runner.generate(row, validation_max_tokens=16)
                path = ROOT / 'validation' / f'{method}_{index}.json'
                atomic(path, result)
                reference = json.loads((PARENT / 'results/fp16' / f'{index:06d}.json').read_text())
                # Compare only after independent generation. No reference
                # tokens are supplied to the model or sampler.
                assert result['generated_ids'][0] == reference['generated_ids'][0], 'Matched prefill first-token mismatch'
                checks.append(dict(index=index, result=str(path), result_sha256=sha(path),
                                   matched_prefill_first_token=True, all_36_native_decode_layers=True))
            atomic(validation, dict(passed=True, checks=checks, run_fingerprint=frozen['fingerprint']))
        else:
            assert json.loads(validation.read_text())['run_fingerprint'] == frozen['fingerprint']
        for row in pending:
            index = row['index']
            # Give the inference function no source answers or grading tests.
            request = {k:v for k,v in row.items() if k in
                       ('index','case_id','dataset','input_ids','input_ids_sha256','input_tokens','sampling_seed')}
            atomic(ROOT / 'checkpoint.json', dict(method=method, stage='request_active',
                   row={k:v for k,v in request.items() if k != 'input_ids'}))
            status('request_start', method=method, case=row['case_id'], index=index)
            result = runner.generate(request)
            result['run_fingerprint'] = frozen['fingerprint']
            reference = json.loads((PARENT / 'results/fp16' / f'{index:06d}.json').read_text())
            if result['generated_ids'][0] != reference['generated_ids'][0]:
                # A prefill mismatch invalidates the matched-quality control.
                # Preserve the independent answer outside formal results and
                # stop this method rather than scoring a changed protocol.
                atomic(ROOT / 'validation' / f'{method}_{index}_prefill_mismatch.json', result)
                atomic(ROOT / 'checkpoint.json', dict(method=method, stage='prefill_validation_failed', index=index))
                raise RuntimeError('FP16 prefill first-token mismatch; method halted')
            result['matched_fp16_prefill_first_token'] = True
            atomic(folder / f'{index:06d}.json', result)
            atomic(ROOT / 'checkpoint.json', dict(method=method, stage='request_complete', index=index))
            status('request_complete', method=method, case=row['case_id'], index=index,
                   generated_tokens=result['generated_tokens'], finish_reason=result['finish_reason'])
    finally:
        runner.close()


if __name__ == '__main__':
    main()
