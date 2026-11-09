"""Portable, independent quality shards; never resume the original service.

Reuse the frozen numerical implementations with explicit local model/output
paths. Each request keeps its original seed, independent of server or shard.
Physical counters are disabled, so these results are quality-only.
"""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
TREE = PROJECT / 'experiments/staged_vllm_recovery_20261106'
PAPER = PROJECT / 'experiments/paper_k4v4_quality_20261109'
METHODS = ('fp16', 'Jacobian_Tree', 'JacK_MixV', 'Data_Tree', 'Gaussian_Tree',
           'turboquant_paper_k4v4', 'kvarn_k4v4_g128')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_cases(rows, index, count, datasets=()):
    if count < 1 or not 0 <= index < count:
        raise ValueError('Require num-shards >= 1 and 0 <= shard-index < num-shards')
    eligible = [r for r in rows if not datasets or r['dataset'] in datasets]
    return eligible, [r for r in eligible if r['index'] % count == index]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def audit_source():
    original = json.loads((TREE / 'source_baseline/manifest.json').read_text())
    source = PROJECT / 'src/KVarN/vllm'
    # Editable installation regenerates version metadata on a new checkout;
    # all numerical/source modules must still match the original snapshot.
    generated = {'_version.py'}
    mismatches = [name for name, digest in original['source_sha256'].items() if name not in generated
                  if not (source / name).is_file() or sha(source / name) != digest]
    if mismatches:
        raise RuntimeError(f'Pinned vLLM source differs or is missing: {mismatches[:10]}')
    return dict(source=str(source), files_checked=len(original['source_sha256']) - len(generated), changed_files=[],
                generated_version_sha256=sha(source / '_version.py') if (source / '_version.py').exists() else None,
                baseline_sha256=sha(TREE / 'source_baseline/manifest.json'))


def build_protocol(method, model, source_root, eligible, runtime_versions=None):
    paths = list((source_root / 'src').glob('*.py'))
    paths += [Path(__file__), TREE / 'inputs/cases.jsonl', TREE / 'inputs/manifest.json',
              TREE / 'src/sampling.py', TREE / 'src/score.py',
              PROJECT / 'experiments/full_quality_20261027/src/score.py',
              PROJECT / 'experiments/full_quality_20261027/vendor/official_sources.json']
    if source_root == PAPER:
        paths += list((PAPER / 'references').glob('*.npz')) + [PAPER / 'experiment.json']
    elif method != 'fp16':
        paths += list((TREE / 'codebooks' / method).glob('*.npz'))
    model_hashes = {n:sha(model / n) for n in ('config.json', 'generation_config.json')}
    expected = json.loads((TREE / 'run_manifest.json').read_text())['source_hashes']
    for name, digest in model_hashes.items():
        matching = [v for k,v in expected.items() if k.endswith('/' + name)]
        if len(matching) != 1 or matching[0] != digest:
            raise RuntimeError(f'Model {name} does not match the frozen Qwen3-4B revision')
    versions = dict(runtime_versions) if runtime_versions is not None else {}
    if runtime_versions is None:
        for package in ('torch', 'vllm', 'triton'):
            try: versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError: versions[package] = 'not-installed'
    protocol = dict(method=method, quality_only=True, model_revision='1cfa9a7208912126459214e8b04321603b3df60c',
        model_config_hashes=model_hashes, input_sha256=sha(TREE / 'inputs/cases.jsonl'),
        generation=json.loads((TREE / 'inputs/manifest.json').read_text())['generation'],
        eligible_indices=[r['index'] for r in eligible], runtime_versions=versions,
        source_hashes={str(p.relative_to(PROJECT)):sha(p) for p in sorted(set(paths))})
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    return protocol, fingerprint


def bind_runtime(source_root, output, model):
    sys.path.insert(0, str(source_root / 'src'))
    common = load_module('common', source_root / 'src/common.py')
    common.ROOT = output
    common.MODEL = model
    common.PROJECT = PROJECT
    common.PARENT = TREE
    # Both scoring implementations see the same frozen case definitions.
    common.cases = lambda: [json.loads(line) for line in (TREE / 'inputs/cases.jsonl').read_text().splitlines() if line]
    for name, target in [('inputs', TREE / 'inputs'), ('codebooks', TREE / 'codebooks'),
                         ('references', PAPER / 'references'), ('native', TREE / 'native')]:
        link = output / name
        if link.is_symlink():
            if link.resolve() != target.resolve():
                raise RuntimeError(f'Runtime link points to another checkout: {link}')
        elif link.exists():
            raise RuntimeError(f'Refusing to replace existing runtime directory: {link}')
        else:
            link.symlink_to(target.resolve(), target_is_directory=True)
    if source_root == TREE:
        guard = load_module('source_guard', TREE / 'src/source_guard.py')
        guard.verify_original = audit_source
    return common


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--dataset', action='append', default=[])
    parser.add_argument('--gpu', default='0', help='One physical GPU index or UUID')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--score-only', action='store_true')
    args = parser.parse_args()
    source_root = PAPER if args.method in ('turboquant_paper_k4v4','kvarn_k4v4_g128') else TREE
    run_name = args.method if source_root == PAPER or args.method == 'fp16' else args.method + '__fixed4'
    output = args.output.expanduser().resolve()
    # Protect all experiment sources, the running service and historical results.
    if output == PROJECT or output.is_relative_to(PROJECT / 'experiments'):
        parser.error('Use an independent output directory, e.g. ./portable_runs/method-shard0')
    rows = [json.loads(line) for line in (TREE / 'inputs/cases.jsonl').read_text().splitlines() if line]
    manifest = json.loads((TREE / 'inputs/manifest.json').read_text())
    if sha(TREE / 'inputs/cases.jsonl') != manifest['cases_sha256']:
        raise RuntimeError('Frozen input manifest mismatch')
    if args.dataset and not set(args.dataset) <= {r['dataset'] for r in rows}:
        parser.error('Unknown dataset name')
    eligible, selected = select_cases(rows, args.shard_index, args.num_shards, args.dataset)
    plan = dict(method=args.method, run_name=run_name, shard_index=args.shard_index,
        num_shards=args.num_shards, eligible_cases=len(eligible), selected_cases=len(selected),
        datasets=dict(Counter(r['dataset'] for r in selected)), indices=[r['index'] for r in selected],
        model=str(args.model.expanduser().resolve()), output=str(output), quality_only=True)
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2));return
    model = args.model.expanduser().resolve()
    audit = audit_source()
    old = output / 'portable_manifest.json'
    prior = json.loads(old.read_text()) if old.exists() else None
    if args.score_only and prior is None:
        raise RuntimeError('--score-only requires an existing portable_manifest.json')
    versions = prior['protocol']['runtime_versions'] if args.score_only else None
    protocol, fingerprint = build_protocol(args.method, model, source_root, eligible, versions)
    output.mkdir(parents=True, exist_ok=True)
    if prior is not None:
        if prior['protocol_fingerprint'] != fingerprint or prior['plan']['indices'] != plan['indices']:
            raise RuntimeError('Output already belongs to a different protocol/shard')
    common = bind_runtime(source_root, output, model)
    common.atomic(old, dict(protocol=protocol, protocol_fingerprint=fingerprint, plan=plan))
    folder = output / 'results' / run_name
    (folder / 'scores').mkdir(parents=True, exist_ok=True)
    if args.score_only:
        scorer = load_module('portable_official_score', PAPER / 'src/score.py')
        sys.argv = [str(PAPER / 'src/score.py'), '--method', run_name]
        scorer.main();return
    if not selected:
        common.atomic(output / 'generation_complete.json', dict(selected_cases=0,generated=0,
                      protocol_fingerprint=fingerprint))
        print('This shard has no selected cases; no GPU started.');return
    # Honor the existing server's GPU lock, including the original root queue.
    if ',' in args.gpu:
        parser.error('Each shard uses exactly one GPU')
    def uuid(selector):
        return subprocess.check_output(['nvidia-smi','-i',selector,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    gpu_uuid = uuid(args.gpu)
    lock_name = 'staged_algorithm_gpu.lock' if gpu_uuid == uuid('0') else 'portable_' + re.sub(r'[^A-Za-z0-9-]','_',gpu_uuid) + '.lock'
    (PROJECT / 'locks').mkdir(exist_ok=True)
    import fcntl
    with (PROJECT / 'locks' / lock_name).open('a') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Selected GPU is occupied by another experiment; choose another GPU/server')
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        if source_root == PAPER:
            module = load_module('portable_generation', PAPER / 'src/generate.py')
            if args.method == 'turboquant_paper_k4v4':
                from validate_paper_gpu import validate
                validate()
            runner = module.Runner(args.method)
        else:
            module = load_module('portable_generation', TREE / 'src/vllm_runner.py')
            runner = module.VllmRunner(args.method)
        try:
            for row in selected:
                path = folder / f"{row['index']:06d}.json"
                if path.exists():
                    record = json.loads(path.read_text())
                    if record.get('run_fingerprint') != fingerprint or record['input_ids_sha256'] != row['input_ids_sha256']:
                        raise RuntimeError(f'Existing result identity mismatch: {path}')
                    continue
                request = {k:row[k] for k in ('index','case_id','dataset','input_ids','input_ids_sha256','input_tokens','sampling_seed')}
                common.atomic(output / 'active_request.json', dict(index=row['index'],case_id=row['case_id']))
                if source_root == PAPER:
                    record = runner.generate(request)
                else:
                    import uuid as uuid_module
                    record = runner.generate(request, output / 'unused_counter_path', uuid_module.uuid4().hex, with_counters=False)
                    record['physical_traffic'] = dict(status='not_measured', reason='Portable quality-only shard')
                record.update(run_fingerprint=fingerprint, portable_shard=dict(index=args.shard_index,count=args.num_shards),
                              host=socket.gethostname(), gpu_uuid=gpu_uuid, model_path=str(model),
                              original_source_audit=audit, performance_claims=False)
                common.atomic(path, record)
                print(json.dumps(dict(completed=row['case_id'],index=row['index'],tokens=record['generated_tokens'])),flush=True)
        finally:
            runner.close()
    common.atomic(output / 'generation_complete.json', dict(method=args.method,selected_cases=len(selected),
                   generated=sum((folder / f"{r['index']:06d}.json").exists() for r in selected),
                   protocol_fingerprint=fingerprint))
    print('Generation complete. Run --score-only with .venv-eval/bin/python to score this shard.',flush=True)


if __name__ == '__main__':
    main()
