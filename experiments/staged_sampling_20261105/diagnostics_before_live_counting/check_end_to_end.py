"""Natural-EOS smoke, real model replay and actual DRAM receipts (no short cap)."""
import torch
from transformers import AutoTokenizer
from common import *
from generate import Model


def row(tok, case, index, prompt):
    ids=tok.apply_chat_template([dict(role='user',content=prompt)],tokenize=True,
                               add_generation_prompt=True,enable_thinking=True)
    return dict(case_id=case,index=index,dataset='diagnostic',input_ids=ids,
                input_ids_sha256=digest(ids),sampling_seed=12345)


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(42)
    tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True)
    results={}
    for codec in ('fp16','Jacobian_Tree'):
        m=Model(codec)
        sample=row(tok,'sampling_natural_smoke',99990,'What is 1+1? Give only the final number.')
        first=m.generate(sample)
        assert first['finish_reason']=='eos' and first['thinking_complete']
        assert first['physical_traffic']['status']=='measured'
        assert first['text'].strip()=='2'
        prefix=first['generated_ids'][:min(8,len(first['generated_ids'])-1)]
        checkpoint=ROOT/'smoke'/f'{codec}_replay.json'
        atomic(checkpoint,dict(case_id=sample['case_id'],fingerprint='smoke',run=m.identity,
            input_ids_sha256=sample['input_ids_sha256'],generated_ids=prefix))
        sample['index']=99991
        resumed=m.generate(sample,checkpoint,'smoke')
        assert first['generated_ids']==resumed['generated_ids']
        assert first['logical_K_V_bits']==resumed['logical_K_V_bits']
        assert resumed['resumed_prefix_tokens']==len(prefix)
        for condition in ('warm','evicted'):
            assert first['physical_traffic']['conditions'][condition]['profiled_kernel_launches']>0
        atomic(ROOT/'smoke'/f'{codec}_natural.json',first)
        atomic(ROOT/'smoke'/f'{codec}_resumed.json',resumed)
        results[codec]=dict(natural_eos=True,thinking_complete=True,final_answer=first['text'],
            generated_tokens=first['generated_tokens'],exact_model_sampling_replay=True,
            physical_traffic=first['physical_traffic'])
        # Remove the global hook's reference before freeing model memory.
        from transformers.models.qwen3.modeling_qwen3 import ALL_ATTENTION_FUNCTIONS
        ALL_ATTENTION_FUNCTIONS.register('staged_algorithm',ALL_ATTENTION_FUNCTIONS['sdpa'])
        del m
        torch.cuda.empty_cache()
    atomic(ROOT/'checks_end_to_end.json',dict(passed=True,results=results))
    print('Natural EOS, thinking, real-model sampling replay, hardware counters passed',flush=True)


if __name__=='__main__':main()
