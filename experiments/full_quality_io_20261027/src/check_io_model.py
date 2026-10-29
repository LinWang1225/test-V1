"""At the dataset boundary, verify IO instrumentation preserves generation."""
from common import *
import torch
from generate_io import IOQualityModel


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    previous=json.loads((ROOT.parent/'encoding_ablation_20261024/inputs/manifest.json').read_text())
    old=next(r for r in previous if r['split']=='test')
    row=dict(old,index=0,input_tokens=len(old['input_ids']),original_input_tokens=len(old['input_ids']),input_truncated=False,max_new_tokens=8,input_ids_sha256='io_numeric_gate')
    model=IOQualityModel('query_curve3',profile_enabled=False)
    checked=[]
    for policy in ('fp16','full4','uniform3','query_only3','query_curve3'):
        model.policy=policy;a=model.generate(row)
        assert a['traffic_schema_version']==2
        if policy!='query_only3':
            path=ROOT.parent/'controller_jacobian_20261026/results/online'/(old['case_id'].replace('/','_')+'_'+policy+'.json')
            target=json.loads(path.read_text())['generated_ids'][:8]
            assert a['generated_ids']==target,(policy,a['generated_ids'],target)
        bits=16 if policy=='fp16' else 4 if policy=='full4' else 3
        tokens=sum(len(old['input_ids'])+i for i in range(1,len(a['generated_ids'])))
        for side in ('K','V'):assert a['logical_traffic'][side]['payload_bits']==36*8*128*bits*tokens
        assert a['physical_traffic']['dram_read_bytes'] is None
        assert a['physical_traffic']['matching_sample_logical_traffic']['context']==len(old['input_ids'])+1
        checked.append(policy)
    atomic(ROOT/'io_model_checks.json',dict(passed=True,policies=checked,previous_reference_tokens_preserved=True,separate_K_V_bits_match_all_decode_calls=True,missing_hardware_not_fabricated=True))
    print('IO instrumentation generation parity passed',flush=True)


if __name__=='__main__':main()
