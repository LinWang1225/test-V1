import os
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0'
os.environ['VLLM_USE_V2_MODEL_RUNNER']='0'
os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import re,time,uuid
import torch
from common import CONFIG,GENERATION,MODEL,split_response,progress
from native_backend import STATE,install
class NativeRunner:
    def __init__(self,callback=None,backend='capture'):
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        install(callback,backend)
        from vllm import LLM
        self.llm=LLM(model=str(MODEL),dtype='float16',kv_cache_dtype='auto',max_model_len=40960,block_size=128,max_num_seqs=1,max_num_batched_tokens=512,kv_cache_memory_bytes=36*2*8*128*2*(40960+128),gpu_memory_utilization=.8,enable_prefix_caching=False,enable_chunked_prefill=True,distributed_executor_backend='uni',async_scheduling=False,seed=GENERATION['seed'],generation_config='vllm',attention_config={'backend':'FLASH_ATTN'},enforce_eager=True,disable_log_stats=True)
        self.engine=self.llm.llm_engine
        worker=self.engine.model_executor.driver_worker.worker
        self.model_runner=worker.model_runner
        self.tokenizer=self.llm.get_tokenizer()
        self.model_classes=sorted({type(m).__module__+'.'+type(m).__name__ for m in self.model_runner.model.modules() if 'qwen3' in type(m).__module__})
        assert self.model_classes and all(x.startswith('vllm.') for x in self.model_classes)
        self.output_weights={}
        for name,parameter in self.model_runner.model.named_parameters():
            match=re.search(r'layers\.(\d+)\.self_attn\.o_proj\.weight$',name)
            if match:self.output_weights[int(match.group(1))]=parameter
        assert len(self.output_weights)==36
        STATE.weight_provider=lambda layer:self.output_weights[layer]
        pool=self.engine.engine_core.engine_core.scheduler.kv_cache_manager.block_pool
        assert (pool.num_gpu_blocks-1)*128>=40960
    @torch.inference_mode()
    def generate(self,row,*,calibration=False,job_name='generation',run_id='fp16'):
        from vllm import SamplingParams
        ids=row['input_ids'];n=len(ids)
        assert n==row.get('input_tokens',row.get('context',n)) and n<40960
        if calibration:
            params=SamplingParams(temperature=0,seed=20261024,max_tokens=65,ignore_eos=True,stop_token_ids=[])
        else:
            params=SamplingParams(temperature=.6,top_p=.95,top_k=20,min_p=0,seed=row['sampling_seed'],max_tokens=40960-n,stop_token_ids=CONFIG['eos_token_ids'],ignore_eos=False)
        STATE.layer_calls={};STATE.active=True;STATE.phase='prefill'
        request='factorial-'+uuid.uuid4().hex
        self.engine.add_request(request,{'prompt_token_ids':ids},params)
        remaining=n;steps=0;finished=None;begin=time.perf_counter();prefill=None
        while self.engine.has_unfinished_requests():
            STATE.phase='prefill' if remaining else 'decode'
            STATE.decode_position=steps+1 if not remaining else 0
            STATE.last_scheduled=0
            outputs=self.engine.step()
            scheduled=STATE.last_scheduled
            if scheduled<=0: raise RuntimeError('Unexpected idle native engine step')
            if remaining:
                assert scheduled<=remaining
                remaining-=scheduled
                if remaining==0: prefill=time.perf_counter()-begin
            else:
                assert scheduled==1
                steps+=1
            for item in outputs:
                if item.finished: finished=item
            if steps and steps%128==0:
                progress(job_name,'generating',run_id=run_id,case_id=row['case_id'],generated_tokens=steps+1)
        torch.cuda.synchronize();STATE.active=False
        assert finished is not None and remaining==0
        if steps:
            assert len(STATE.layer_calls)==36 and all(value==steps for value in STATE.layer_calls.values()),STATE.layer_calls
        output=finished.outputs[0];generated=list(output.token_ids)
        assert len(generated) in (steps,steps+1)
        result={'case_id':row['case_id'],'dataset':row['dataset'],'index':row.get('index'),'input_tokens':n,'input_ids_sha256':row.get('input_ids_sha256'),'sampling_seed':row.get('sampling_seed',20261024),'run_id':run_id,'generated_ids':generated,'generated_tokens':len(generated),'actual_decode_steps':steps,'generation':GENERATION if not calibration else {'mode':'fixed greedy calibration64 captured decode steps'},'native_context_length':40960,'artificial_output_limit':65 if calibration else None,'input_truncated':False,'finish_reason':'eos' if output.finish_reason=='stop' else 'native_context_limit','native_finish_reason':output.finish_reason,'native_stop_reason':output.stop_reason,'natural_completion':output.finish_reason=='stop','elapsed_seconds':time.perf_counter()-begin,'prefill_seconds':prefill,'backend':'native_vllm','model_classes':self.model_classes,'layers_exercised':sorted(STATE.layer_calls),'completed_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
        result.update(split_response(self.tokenizer,generated))
        return result
    def close(self):
        STATE.active=False
        self.engine.model_executor.shutdown()
        import torch.distributed as dist
        if dist.is_initialized():dist.destroy_process_group()
