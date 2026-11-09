"""One persistent vLLM engine per codec; counters cover every evaluated answer."""
import argparse
import os
import uuid
from common import ROOT, atomic, rows, sha, status, run_id
from provenance import freeze


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--codec', required=True)
    args = p.parse_args()
    assert os.geteuid() == 0, 'Physical counters must be enabled as root'
    frozen = freeze()
    assert args.codec in frozen['queue']
    owner = (ROOT / 'config.json').stat()
    identity = run_id(args.codec, 'fixed', 4)
    dest = ROOT / 'results' / identity
    profiles = ROOT / 'profiles' / identity
    for directory in (dest, dest / 'scores', dest / 'traffic_totals', profiles):
        directory.mkdir(parents=True, exist_ok=True)
        os.chown(directory, owner.st_uid, owner.st_gid)
    from vllm_runner import VllmRunner
    runner = VllmRunner(args.codec)
    try:
        for row in rows(ROOT / 'inputs/cases.jsonl'):
            name = f"{row['index']:06d}.json"
            output = dest / name
            if output.exists():
                import json
                record = json.loads(output.read_text())
                assert record['run_fingerprint'] == frozen['fingerprint']
                assert record['input_ids_sha256'] == row['input_ids_sha256']
                assert record['backend'] == 'vllm'
            else:
                execution_id = uuid.uuid4().hex
                base = profiles / f"{row['index']:06d}_{execution_id}"
                status('vllm_request_start', run=identity, index=row['index'], case=row['case_id'],
                       input_tokens=row['input_tokens'], execution_id=execution_id)
                atomic(base.with_suffix('.invocation.json'), dict(execution_id=execution_id,
                       case_id=row['case_id'], input_ids_sha256=row['input_ids_sha256'],
                       sampling_seed=row['sampling_seed'], run_fingerprint=frozen['fingerprint'],
                       actual_generation_executions=1))
                record = runner.generate(row, base, execution_id, with_counters=True)
                record['run_fingerprint'] = frozen['fingerprint']
                atomic(output, record)
                os.chown(output, owner.st_uid, owner.st_gid)
                for path in profiles.glob(base.name + '.*'):
                    os.chown(path, owner.st_uid, owner.st_gid)
            measured = record['physical_traffic']
            assert measured['status'] == 'measured' and measured['execution_id'] == record['execution_id']
            assert sha(measured['raw_csv']) == measured['raw_csv_sha256']
            receipt = dict(**measured, case_id=row['case_id'], dataset=row['dataset'],
                           generated_tokens=record['generated_tokens'], generation_sha256=sha(output),
                           generation_path=str(output), complete=True, actual_generation_executions=1,
                           scope='complete live prefill and decode, including thinking, weights, KV and sampling; not KV-only')
            atomic(dest / 'traffic_totals' / name, receipt)
            os.chown(dest / 'traffic_totals' / name, owner.st_uid, owner.st_gid)
            status('vllm_request_complete', run=identity, index=row['index'], case=row['case_id'],
                   generated_tokens=record['generated_tokens'], finish_reason=record['finish_reason'],
                   elapsed_seconds=record['elapsed_seconds'], dram_total_bytes=measured['dram_total_bytes'])
    finally:
        runner.close()


if __name__ == '__main__':main()
