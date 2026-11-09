"""Merge compatible portable quality shards without overwriting conflicts."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect(shards):
    manifests = [json.loads((p / 'portable_manifest.json').read_text()) for p in shards]
    if not manifests:
        raise ValueError('No input shards')
    fingerprint = manifests[0]['protocol_fingerprint']
    if any(m['protocol_fingerprint'] != fingerprint for m in manifests):
        raise ValueError('Shards have different methods, sources, inputs, model or runtime versions')
    run = manifests[0]['plan']['run_name']
    eligible = set(manifests[0]['protocol']['eligible_indices'])
    results = {};scores = {}
    for directory, manifest in zip(shards, manifests):
        assigned = set(manifest['plan']['indices'])
        for path in (directory / 'results' / run).glob('[0-9]*.json'):
            record = json.loads(path.read_text());index = record['index']
            if record['run_fingerprint'] != fingerprint or index not in assigned or index not in eligible:
                raise ValueError(f'Result does not belong to this shard: {path}')
            if index in results and sha(results[index]) != sha(path):
                raise ValueError(f'Conflicting independent generations for case {index}')
            results[index] = path
            score_path = path.parent / 'scores' / path.name
            if score_path.exists():
                score = json.loads(score_path.read_text())
                if score['generation_sha256'] != sha(path):
                    raise ValueError(f'Score belongs to a different generation: {score_path}')
                if index in scores and sha(scores[index]) != sha(score_path):
                    raise ValueError(f'Conflicting scores for case {index}')
                scores[index] = score_path
    return manifests[0], results, scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('shards',type=Path,nargs='+')
    args=parser.parse_args()
    manifest,results,scores=collect(args.shards)
    out=args.output.resolve();run=manifest['plan']['run_name']
    previous=out/'merged_summary.json'
    if previous.exists() and json.loads(previous.read_text())['protocol_fingerprint']!=manifest['protocol_fingerprint']:
        raise ValueError('Output directory already belongs to a different merged protocol')
    target=out/'results'/run
    (target/'scores').mkdir(parents=True,exist_ok=True)
    # Preflight every destination before copying any files.
    copies=[(path,target/path.name) for path in results.values()]
    copies += [(path,target/'scores'/path.name) for path in scores.values()]
    for source,dest in copies:
        if dest.exists() and sha(source)!=sha(dest):
            raise ValueError(f'Refusing to overwrite different existing result: {dest}')
    for source,dest in copies:
        if not dest.exists():
            temporary=dest.with_suffix('.tmp');shutil.copy2(source,temporary);temporary.replace(dest)
    by=defaultdict(list)
    for path in scores.values():
        score=json.loads(path.read_text());by[score['dataset']].append(score['score'])
    eligible=set(manifest['protocol']['eligible_indices'])
    summary=dict(protocol_fingerprint=manifest['protocol_fingerprint'],method=manifest['protocol']['method'],
        expected=len(eligible),generated=len(results),scored=len(scores),
        complete=set(results)==eligible and set(scores)==eligible,
        missing_generation_indices=sorted(eligible-set(results)),
        missing_score_indices=sorted(eligible-set(scores)),
        tasks={k:dict(scored=len(v),score_percent=100*sum(v)/len(v)) for k,v in by.items()},
        quality_only=True,sources=[str(p.resolve()) for p in args.shards])
    (out/'merged_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (out/'merged_protocol.json').write_text(json.dumps(manifest['protocol'],ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
