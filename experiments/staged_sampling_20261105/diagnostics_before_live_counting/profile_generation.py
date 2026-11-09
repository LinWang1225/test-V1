"""Measure complete original trajectories; no caps, skipping or forced tokens."""
import argparse
import torch
from common import *
from generate import Model


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fixture',required=True)
    args=parser.parse_args()
    torch.set_num_threads(4);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    data=torch.load(args.fixture,map_location='cpu',weights_only=True)
    assert data['kind']=='whole_generation'
    assert data['source_hashes']=={p.name:sha(p) for p in (ROOT/'src').glob('*.py')}
    row=data['row']
    assert digest(row['input_ids'])==row['input_ids_sha256']
    m=Model(**data['model_args'])
    actual=m.generate(row,collect_physical=False,profile_ranges=True)
    assert actual['generated_ids']==data['expected_ids'],'Profiled sampling trajectory differs from quality run'
    assert actual['logical_K_V_bits']==data['expected_logical_K_V_bits']
    print('whole_generation_trajectory_verified',row['case_id'],len(actual['generated_ids']),flush=True)


if __name__=='__main__':main()
