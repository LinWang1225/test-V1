"""The real profiled request; each sampled token feeds the next model step."""
import argparse
import os
import time
import torch
from common import *
from generate import Model


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--request',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--invocation-guard',required=True)
    args=parser.parse_args()
    # With these two DRAM metrics one pass must suffice. A profiler relaunch
    # must fail instead of silently replacing the quality trajectory.
    execution_id=digest([args.output,os.getpid(),time.time_ns()])
    with open(args.invocation_guard,'x') as f:
        json.dump(dict(execution_id=execution_id,pid=os.getpid(),started=time.time(),actual_executions=1),f)
    data=torch.load(args.request,map_location='cpu',weights_only=True)
    assert data['kind']=='actual_request'
    assert not any(k in data for k in ('expected_ids','generated_ids','reference_tokens','expected_output'))
    assert data['source_hashes']=={p.name:sha(p) for p in (ROOT/'src').glob('*.py')}
    row=data['row'];assert digest(row['input_ids'])==row['input_ids_sha256']
    assert 'source_row' not in row and 'answer' not in row and 'generated_ids' not in row
    torch.set_num_threads(4);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    status('live_profiled_loading_model',case=row['case_id'],codec=data['model_args']['codec'])
    model=Model(**data['model_args'])
    result=model.generate(row,profile_ranges=True)
    assert result['decode_attention_calls']==36*max(0,result['generated_tokens']-1)
    result.update(execution_id=execution_id,fingerprint=data['fingerprint'],
                  request_sampling_seed=row['sampling_seed'])
    atomic(args.output,result)
    print('actual_end_to_end_request_completed',row['case_id'],result['generated_tokens'],execution_id,flush=True)


if __name__=='__main__':main()
