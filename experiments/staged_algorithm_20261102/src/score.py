"""Pinned official scorers; preserve native-limit outputs and completion status."""
import argparse
import importlib.util
import sys
from common import *

def load_scorers():
    sources=json.loads((LEGACY/'vendor/official_sources.json').read_text())
    for rel, info in sources.items():
        if rel.startswith('vendor/'):
            assert sha(LEGACY/rel)==info['sha256'], rel
    sys.path.insert(0,str(LEGACY/'vendor'))
    spec=importlib.util.spec_from_file_location('frozen_task_scorers',LEGACY/'src/score.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT=LEGACY
    return module

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--smoke-tag',default='current');a=p.parse_args()
    scorers=load_scorers()
    folder=(ROOT/'smoke'/a.smoke_tag if a.smoke else ROOT/'results')/a.run
    manifest=json.loads((ROOT/'inputs/manifest.json').read_text())
    assert sha(ROOT/'inputs/cases.jsonl')==manifest['cases_sha256']
    cases={r['index']:r for r in rows(ROOT/'inputs/cases.jsonl')}
    n=0
    for path in sorted(folder.glob('[0-9]*.json')):
        record=json.loads(path.read_text());row=cases[record['index']]
        assert record['case_id']==row['case_id'] and record['input_ids_sha256']==row['input_ids_sha256']
        dest=folder/'scores'/path.name
        if dest.exists():
            assert json.loads(dest.read_text())['generation_sha256']==sha(path)
        else:
            try:
                if row['dataset']=='humaneval':
                    score=scorers.human_score(record['text'],row['source_row'])
                elif row['dataset']=='math500':
                    score=scorers.math_score(record['text'],row['source_row'])
                else:
                    score=scorers.longbench_score(record['text'],row['source_row'],row['dataset'])
                atomic(dest,dict(case_id=row['case_id'],index=row['index'],dataset=row['dataset'],
                    generation_sha256=sha(path),natural_completion=record['natural_completion'],**score))
            except BaseException as exc:
                atomic(folder/'grading_error.json',dict(case_id=row['case_id'],generation_sha256=sha(path),error=repr(exc)))
                raise
        n+=1
    atomic(folder/'scoring_status.json',dict(scored=n,expected=464,complete=n==464,smoke=a.smoke))
    print('scored',a.run,n,flush=True)

if __name__=='__main__':
    main()
