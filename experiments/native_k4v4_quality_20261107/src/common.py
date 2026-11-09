"""Independent quality extension; never alter the running frozen experiment."""
import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
PARENT = PROJECT / 'experiments/staged_vllm_recovery_20261106'
LEGACY = PROJECT / 'experiments/full_quality_20261027'
CONFIG = json.loads((ROOT / 'experiment.json').read_text())
GENERATION = CONFIG['generation']
METHODS = tuple(CONFIG['methods'])
MODEL = Path(json.loads((PARENT / 'run_manifest.json').read_text())['model'])


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with temp.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def cases():
    manifest = json.loads((PARENT / 'inputs/manifest.json').read_text())
    assert manifest['generation'] == GENERATION
    assert manifest['native_context_length'] == CONFIG['max_model_len'] == 40960
    assert sha(PARENT / 'inputs/cases.jsonl') == manifest['cases_sha256']
    with (PARENT / 'inputs/cases.jsonl').open() as stream:
        result = [json.loads(line) for line in stream if line.strip()]
    assert len(result) == 464
    assert [r['index'] for r in result] == list(range(464))
    for row in result:
        token_hash = hashlib.sha256(json.dumps(row['input_ids'], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        assert row['input_ids_sha256'] == token_hash
        assert 0 < row['input_tokens'] == len(row['input_ids']) < 40960
    return result


def status(stage, **values):
    data = dict(stage=stage, pid=os.getpid(), updated=time.strftime('%Y-%m-%d %H:%M:%S'), **values)
    atomic(ROOT / 'status.json', data)
    print(json.dumps(data, ensure_ascii=False), flush=True)


def freeze():
    baseline = json.loads((PARENT / 'source_baseline/manifest.json').read_text())
    source = Path(baseline['original_vllm_source'])
    changed = [name for name, digest in baseline['source_sha256'].items()
               if not (source / name).is_file() or sha(source / name) != digest]
    assert not changed, f'Upstream source changed: {changed}'
    paths = [*(ROOT / 'src').glob('*.py'), ROOT / 'experiment.json',
             PARENT / 'inputs/cases.jsonl', PARENT / 'inputs/manifest.json',
             PARENT / 'run_manifest.json', PARENT / 'source_baseline/manifest.json',
             PARENT / 'src/sampling.py', PARENT / 'src/score.py',
             LEGACY / 'src/score.py', LEGACY / 'vendor/official_sources.json',
             MODEL / 'config.json', MODEL / 'generation_config.json']
    value = dict(config=CONFIG, source_hashes={str(p): sha(p) for p in sorted(paths)},
                 model=str(MODEL), upstream_files_checked=len(baseline['source_sha256']))
    fingerprint = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    value['fingerprint'] = fingerprint
    path = ROOT / 'run_manifest.json'
    if path.exists():
        assert json.loads(path.read_text()) == value, 'Quality extension sources changed after freeze'
    else:
        atomic(path, value)
    return value
