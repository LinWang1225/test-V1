"""CPU scope annotations only; no device timing or kernel modifications."""
import functools
import importlib.abc
import importlib.machinery
import os
import sys

TARGETS = {"vllm.v1.worker.gpu.model_runner", "vllm.v1.worker.gpu_model_runner"}
COMPONENTS = {
    'vllm.model_executor.models.qwen2': [('Qwen2MLP', 'forward', 'mlp')],
    'vllm.model_executor.models.qwen3': [('Qwen3Attention', 'forward', 'attention_module')],
    'vllm.v1.attention.backends.kvarn_attn': [('KVarNAttentionImpl', 'do_kv_cache_update', 'kv_cache_update'), ('KVarNAttentionImpl', 'forward', 'attention_backend')],
    'vllm.v1.attention.backends.turboquant_attn': [('TurboQuantAttentionImpl', 'do_kv_cache_update', 'kv_cache_update'), ('TurboQuantAttentionImpl', 'forward', 'attention_backend')],
    'vllm.v1.attention.backends.flash_attn': [('FlashAttentionImpl', 'do_kv_cache_update', 'kv_cache_update'), ('FlashAttentionImpl', 'forward', 'attention_backend')],
    'vllm.v1.attention.backends.triton_attn': [('TritonAttentionImpl', 'do_kv_cache_update', 'kv_cache_update'), ('TritonAttentionImpl', 'forward', 'attention_backend')],
    'vllm.v1.attention.backends.flashinfer': [('FlashInferImpl', 'do_kv_cache_update', 'kv_cache_update'), ('FlashInferImpl', 'forward', 'attention_backend')],
}


def patch_components(module):
    import torch
    for class_name, method_name, label in COMPONENTS.get(module.__name__, []):
        cls = getattr(module, class_name, None)
        if cls is None or not hasattr(cls, method_name):
            continue
        fn = getattr(cls, method_name)
        if getattr(fn, '_operator_component', False):
            continue
        def wrap(original, group):
            @functools.wraps(original)
            def scoped(*args, **kwargs):
                with torch.profiler.record_function('op_component|' + group):
                    return original(*args, **kwargs)
            scoped._operator_component = True
            return scoped
        setattr(cls, method_name, wrap(fn, label))


def patch(module):
    cls = module.GPUModelRunner
    if getattr(cls, "_operator_scopes_installed", False):
        return
    import torch
    execute = cls.execute_model

    @functools.wraps(execute)
    def wrapped(self, scheduler_output, *args, **kwargs):
        state = getattr(self, "_op_profile_state", None)
        if state is None:
            state = self._op_profile_state = {"prompt": {}, "computed": {}, "step": 0}
        for rid in scheduler_output.finished_req_ids:
            state["prompt"].pop(rid, None)
            state["computed"].pop(rid, None)
        for req in scheduler_output.scheduled_new_reqs:
            # The V2 prefill boundary includes generated tokens that must be
            # recomputed after preemption, unlike the original prompt length.
            ids = req.prefill_token_ids
            if ids is None:
                ids = req.prompt_token_ids
            state["prompt"][req.req_id] = len(ids) if ids is not None else None
            state["computed"][req.req_id] = req.num_computed_tokens
        cached = scheduler_output.scheduled_cached_reqs
        for rid, computed in zip(cached.req_ids, cached.num_computed_tokens):
            state["computed"][rid] = computed
        prefill = decode = unknown = 0
        scheduled = scheduler_output.num_scheduled_tokens
        for rid, count in scheduled.items():
            plen = state["prompt"].get(rid)
            computed = state["computed"].get(rid)
            if plen is None or computed is None:
                unknown += 1
            elif computed < plen:
                prefill += 1
            else:
                decode += 1
        phase = "unknown" if unknown else ("mixed" if prefill and decode else "prefill" if prefill else "decode" if decode else "empty")
        state["step"] += 1
        label = (f"op_phase|phase={phase}|step={state['step']}|seqs={len(scheduled)}"
                 f"|prefill_seqs={prefill}|decode_seqs={decode}|tokens={sum(scheduled.values())}")
        self._op_profile_last_label = label
        with torch.profiler.record_function(label + "|part=execute"):
            result = execute(self, scheduler_output, *args, **kwargs)
        for rid, count in scheduled.items():
            if rid in state["computed"]:
                state["computed"][rid] += count
        return result

    cls.execute_model = wrapped
    if hasattr(cls, "sample_tokens"):
        sample = cls.sample_tokens

        @functools.wraps(sample)
        def wrapped_sample(self, *args, **kwargs):
            label = getattr(self, "_op_profile_last_label", "op_phase|phase=unknown|step=-1|seqs=0")
            with torch.profiler.record_function(label + "|part=sample"):
                return sample(self, *args, **kwargs)
        cls.sample_tokens = wrapped_sample
    cls._operator_scopes_installed = True
    print(f"[operator-hooks] CPU phase scopes installed: {module.__name__}", flush=True)


class Loader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original
    def create_module(self, spec):
        return self.original.create_module(spec)
    def exec_module(self, module):
        self.original.exec_module(module)
        if module.__name__ in TARGETS:
            patch(module)
        else:
            patch_components(module)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in TARGETS and fullname not in COMPONENTS:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = Loader(spec.loader)
        return spec


def register():
    if os.environ.get("OPERATOR_PROFILE_SCOPES") != "1":
        return
    for name in TARGETS:
        if name in sys.modules:
            patch(sys.modules[name])
    for name in COMPONENTS:
        if name in sys.modules:
            patch_components(sys.modules[name])
    if not any(isinstance(f, Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, Finder())
