"""Freeze the independent adapter, inputs and codebooks before formal work."""
from common import ROOT, GENERATION, CODECS, MODEL, atomic, sha, digest
from source_guard import verify_original


QUEUE = ('fp16', 'Jacobian_Tree', 'JacK_MixV', 'Data_Tree', 'Gaussian_Tree', 'U_Clipped')


def fingerprint():
    paths = [*(ROOT / 'src').glob('*.py'), *(ROOT / 'native').glob('*'), ROOT / 'service.py']
    paths += [ROOT / 'inputs/cases.jsonl', ROOT / 'inputs/manifest.json',
              MODEL / 'config.json', MODEL / 'generation_config.json']
    paths += [p for c in CODECS for p in (ROOT / 'codebooks' / c).glob('*.npz')]
    hashes = {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): sha(p)
              for p in sorted(paths) if p.is_file()}
    value = dict(source_hashes=hashes, original_source_audit=verify_original(),
                 generation=GENERATION, model=str(MODEL), queue=list(QUEUE),
                 backend='vllm', stage='encoding_fixed4_reference', cases_per_run=464,
                 max_model_len=40960, max_num_seqs=1, max_num_batched_tokens=512,
                 physical_measurement_required=True, same_execution_quality_and_counters=True,
                 execution='native vLLM Qwen3, scheduler, sampler and CUDA Graph decode',
                 representation='uint8 reference codes in vLLM pages; FP16 cache retained for prefill; no bit-plane implementation',
                 entrypoint='src/run_queue.py')
    return dict(**value, fingerprint=digest(value))


def freeze():
    import json
    path = ROOT / 'run_manifest.json'
    current = fingerprint()
    if path.exists():
        assert json.loads(path.read_text()) == current, 'Formal sources or inputs changed; refusing to mix results'
    else:
        atomic(path, current)
    return current
