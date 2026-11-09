"""Durable case failures and checkpoints; no placeholder successful records."""
import json
import os
import time
import traceback
from common import ROOT, atomic, sha

CASE_FAILURE_EXIT = 85


def give(path):
    if os.geteuid() == 0:
        owner = (ROOT / 'config.json').stat()
        os.chown(path, owner.st_uid, owner.st_gid)


def checkpoint(identity, **value):
    path = ROOT / 'checkpoints' / (identity + '.json')
    atomic(path, dict(run=identity, pid=os.getpid(),
                     updated=time.strftime('%Y-%m-%d %H:%M:%S'), **value))
    give(path)


def failures(identity):
    return sorted((ROOT / 'failures' / identity).glob('[0-9]*.json'))


def record_failure(identity, row, *, error, exit_code=None, execution_id=None,
                   profile_base=None, source_log=None):
    path = ROOT / 'failures' / identity / f"{row['index']:06d}.json"
    if path.exists():return path
    partial = []
    if profile_base:
        from pathlib import Path
        base = Path(profile_base)
        partial = [str(p) for p in base.parent.glob(base.name + '.*')]
    atomic(path, dict(run=identity, index=row['index'], case_id=row['case_id'],
                     dataset=row['dataset'], input_ids_sha256=row['input_ids_sha256'],
                     sampling_seed=row['sampling_seed'], execution_id=execution_id,
                     status='failed', error=str(error), exit_code=exit_code,
                     partial_artifacts=partial, source_log=source_log,
                     complete_physical_traffic=False, excluded_from_successful_results=True,
                     action='skip this failed case and continue in a new inference process',
                     failed_at=time.strftime('%Y-%m-%d %H:%M:%S')))
    give(path)
    for p in partial:
        give(p)
    return path


def verify_import(path, record, frozen):
    if record['run_fingerprint'] == frozen['fingerprint']:return
    imports = json.loads((ROOT / 'parent_imports.json').read_text())
    assert record['run_fingerprint'] == imports['parent_fingerprint']
    relative = str(path.relative_to(ROOT))
    assert sha(path) == imports['imported_result_hashes'][relative], 'Imported result changed'
    assert record['input_tokens'] + record['generated_tokens'] <= 40832
