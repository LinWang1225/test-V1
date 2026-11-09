"""After releasing the quality model, collect whole-request physical totals."""
import argparse
import torch
from common import *
from traffic import collect


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True)
    args=parser.parse_args()
    folder=ROOT/'results'/args.run
    provenance=json.loads((folder/'provenance.json').read_text())
    samples={r['index']:r for r in rows(ROOT/'inputs/cases.jsonl')}
    paths=sorted(folder.glob('[0-9]*.json'))
    for index,path in enumerate(paths):
        dest=folder/'traffic_totals'/path.name
        record=json.loads(path.read_text())
        if dest.exists():
            assert json.loads(dest.read_text())['generation_sha256']==sha(path)
            continue
        status('whole_generation_physical_counting',run=args.run,case=record['case_id'],remaining=len(paths)-index)
        fixture=ROOT/'fixtures'/f'{args.run}_{record["index"]:05d}_whole.pt'
        data=dict(kind='whole_generation',row=samples[record['index']],
            model_args={k:provenance[k] for k in ('codec','policy','depth','curve','budget','threshold')},
            expected_ids=record['generated_ids'],expected_logical_K_V_bits=record['logical_K_V_bits'],
            source_hashes={p.name:sha(p) for p in (ROOT/'src').glob('*.py')})
        torch.save(data,fixture)
        receipt=collect(fixture,'whole_generation',ROOT/'profiles'/args.run/f'{record["index"]:05d}_whole',timeout=None)
        atomic(dest,dict(status='measured',generation_sha256=sha(path),case_id=record['case_id'],
            dataset=record['dataset'],generated_tokens=record['generated_tokens'],**receipt))
        fixture.unlink()
    atomic(folder/'traffic_totals/complete.json',dict(cases=len(paths),expected=464,complete=len(paths)==464))
    print('Whole-generation physical receipts',args.run,len(paths),flush=True)


if __name__=='__main__':main()
