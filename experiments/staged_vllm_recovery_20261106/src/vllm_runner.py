"""Real vLLM engine, native sampler/scheduler and live same-process CUPTI."""
import os
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0'
os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
os.environ['VLLM_USE_V2_MODEL_RUNNER']='0'
import math,time,resource,uuid
from collections import Counter
import torch
from common import *
from direct_counter import DirectCounter,LIBRARY
from sampling import split_response
from frozen_vllm_backend import install,STATE

# Root uses the same local build tools as the unprivileged vLLM environment.
os.environ['PATH']=str(ROOT.parents[1]/'.venv/bin')+':/usr/local/cuda-12.6/bin:'+os.environ.get('PATH','/usr/bin:/bin')


class VllmRunner:
    def __init__(self,codec,eager=False):
        from source_guard import verify_original
        self.original_source_audit=verify_original()
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        install(codec)
        from vllm import LLM
        self.codec=codec;self.identity='fp16' if codec=='fp16' else codec+'__fixed4'
        self.manifest=json.loads((ROOT/'inputs/manifest.json').read_text())
        self.llm=LLM(model=str(MODEL),dtype='float16',kv_cache_dtype='auto',
            max_model_len=self.manifest['native_context_length'],block_size=128,
            max_num_seqs=1,max_num_batched_tokens=512,
            # vLLM reserves block zero as its null block. Provision that block
            # in addition to the full, unchanged 40960-token request capacity.
            kv_cache_memory_bytes=36*2*8*128*2*(self.manifest['native_context_length']+128),
            gpu_memory_utilization=.8,enable_prefix_caching=False,
            enable_chunked_prefill=True,distributed_executor_backend='uni',
            async_scheduling=False,seed=GENERATION['seed'],generation_config='vllm',
            attention_config=dict(backend='FLASH_ATTN'),enforce_eager=eager,
            compilation_config=dict(cudagraph_mode='FULL_DECODE_ONLY',cudagraph_capture_sizes=[1]),
            disable_log_stats=True)
        self.engine=self.llm.llm_engine
        assert hasattr(self.engine,'model_executor'),'GPU worker must stay in this process for same-execution counters'
        worker=self.engine.model_executor.driver_worker.worker
        self.model_runner=worker.model_runner
        pool=self.engine.engine_core.engine_core.scheduler.kv_cache_manager.block_pool
        self.usable_kv_tokens=(pool.num_gpu_blocks-1)*128
        assert self.usable_kv_tokens>=self.manifest['native_context_length'], 'KV pool cannot hold the native context boundary'
        self.model_classes=sorted({type(m).__module__+'.'+type(m).__name__ for m in self.model_runner.model.modules() if 'qwen3' in type(m).__module__})
        assert self.model_classes and all(s.startswith('vllm.') for s in self.model_classes)
        self.tokenizer=self.llm.get_tokenizer()
        self.eager=eager
        import vllm
        self.environment=dict(effective_uid=os.geteuid(),vllm_version=vllm.__version__,
            torch_version=str(torch.__version__),torch_cuda_version=torch.version.cuda,
            gpu_name=torch.cuda.get_device_name(),max_num_seqs=1,kv_page_tokens=128,
            max_num_batched_tokens=512,enable_prefix_caching=False,
            vllm_use_v2_model_runner=False,enforce_eager=eager)
        self.environment['usable_kv_tokens']=self.usable_kv_tokens
        self.environment['reserved_null_blocks']=1

    @torch.inference_mode()
    def generate(self,row,base,execution_id,with_counters=True):
        from vllm import SamplingParams
        STATE.active=True;STATE.phase='prefill';STATE.scheduled_history=[]
        for impl in STATE.impls.values():impl.counts.zero_()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        counter=DirectCounter(base,execution_id) if with_counters else None
        params=SamplingParams(temperature=.6,top_p=.95,top_k=20,min_p=0,
            seed=row['sampling_seed'],max_tokens=self.manifest['native_context_length']-row['input_tokens'],
            stop_token_ids=self.manifest['eos_token_ids'],ignore_eos=False)
        request_id='case-'+str(row['index'])+'-'+uuid.uuid4().hex[:10]
        self.engine.add_request(request_id,dict(prompt_token_ids=row['input_ids']),params)
        remaining=row['input_tokens'];decode_steps=0;window=0;finished=None;graph_modes=Counter()
        begin=time.perf_counter();prefill_seconds=None
        while self.engine.has_unfinished_requests():
            STATE.phase='prefill' if remaining>0 else 'decode'
            phase=STATE.phase;STATE.last_scheduled=0
            if counter and (phase=='prefill' or window==0):counter.start(phase)
            outputs=self.engine.step()
            scheduled=STATE.last_scheduled
            if scheduled<=0:raise RuntimeError('Unexpected no-work engine iteration inside measured request')
            if phase=='prefill':
                assert scheduled<=remaining
                remaining-=scheduled
                if counter:counter.stop(scheduled)
                if remaining==0:prefill_seconds=time.perf_counter()-begin
            else:
                assert scheduled==1,'This isolated per-request experiment requires C1 decode'
                decode_steps+=1;window+=1;graph_modes[STATE.last_graph_mode]+=1
                if counter and window==32:counter.stop(window);window=0
            for output in outputs:
                if output.finished:finished=output
            if decode_steps and decode_steps%64==0:
                status('vllm_live_generation',run=self.identity,case=row['case_id'],generated_tokens=decode_steps+1,graph_mode=STATE.last_graph_mode)
        if counter and window:counter.stop(window)
        torch.cuda.synchronize();elapsed=time.perf_counter()-begin;STATE.active=False
        assert finished is not None and remaining==0
        output=finished.outputs[0];ids=list(output.token_ids)
        # Tokenizer/stop processing may omit the stop ID; keep both counts explicit.
        assert len(ids) in (decode_steps,decode_steps+1),(len(ids),decode_steps)
        physical=None;finalize=0
        if counter:
            status('vllm_finalizing_counters',case=row['case_id'],generated_tokens=len(ids))
            start=time.perf_counter();physical=counter.finish();finalize=time.perf_counter()-start
            assert physical['stages']['prefill']['units']==row['input_tokens']
            assert physical['stages']['decode']['units']==decode_steps
        dispatch={}
        if self.codec!='fp16':
            assert len(STATE.impls)==36
            for index,impl in STATE.impls.items():
                calls,writes=impl.counts.tolist()
                assert calls==decode_steps,(index,calls,decode_steps)
                assert writes==row['input_tokens']+decode_steps,(index,writes,row['input_tokens']+decode_steps)
                dispatch[index]=dict(decode_calls=calls,encoded_tokens=writes,impl=type(impl).__module__+'.'+type(impl).__name__)
        bits=16 if self.codec=='fp16' else 4
        elements=36*8*128*(decode_steps*row['input_tokens']+decode_steps*(decode_steps+1)//2)
        reason='eos' if output.finish_reason=='stop' else 'native_context_limit'
        result=dict(case_id=row['case_id'],index=row['index'],dataset=row['dataset'],run=self.identity,
            input_ids_sha256=row['input_ids_sha256'],input_tokens=row['input_tokens'],input_truncated=False,
            generated_ids=ids,generated_tokens=len(ids),actual_sampled_tokens=decode_steps+1,
            **split_response(self.tokenizer,ids),finish_reason=reason,natural_completion=reason=='eos',
            native_finish_reason=output.finish_reason,native_stop_reason=output.stop_reason,
            artificial_output_limit=None,native_context_length=self.manifest['native_context_length'],
            generation_config=GENERATION,sampling_seed=row['sampling_seed'],backend='vllm',
            vllm_model_classes=self.model_classes,decode_graph_modes=dict(graph_modes),
            execution_environment=self.environment,
            original_source_audit=self.original_source_audit,
            vllm_model_runner_class=type(self.model_runner).__module__+'.'+type(self.model_runner).__name__,
            vllm_tree_dispatch=dispatch,execution_id=execution_id,
            execution_scope='native vLLM scheduler, sampler, paged KV, own live decode trajectory; no HF model inference',
            logical_K_V_bits=[elements*bits]*2,logical_KV_read_bits=2*elements*bits,
            accessed_K_V_elements=[elements]*2,element_weighted_mean_bits=bits,
            logical_bytes_per_output_token=2*elements*bits/8/max(len(ids),1),
            decode_attention_calls=36*decode_steps,depth_elements_K_V=[{str(bits):elements}]*2,
            local_metadata_peak_bytes=0,local_metadata_write_bytes=0,curve_read_allowance_bytes=0,
            elapsed_seconds=elapsed,prefill_seconds=prefill_seconds,counter_finalize_seconds=finalize,
            reference_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            representation='vLLM FP16 pages' if bits==16 else 'uint8 reference code per scalar in separate paged cache; FP16 pages retained for prefill; no bit-plane loading',
            code_storage_bits_per_scalar=16 if bits==16 else 8,
            completed_at=time.strftime('%Y-%m-%d %H:%M:%S'))
        if physical:result['physical_traffic']=dict(status='measured',same_execution_as_quality=True,**physical)
        return result

    def close(self):
        self.engine.model_executor.shutdown()
        import torch.distributed as dist
        if dist.is_initialized():dist.destroy_process_group()
