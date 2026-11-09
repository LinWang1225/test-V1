"""One persistent vLLM engine per codec; counters cover every evaluated answer."""
import argparse
import os
import uuid
import json
import traceback
from common import ROOT, atomic, rows, sha, status, run_id
from provenance import freeze
from failure_policy import checkpoint, record_failure, verify_import, CASE_FAILURE_EXIT


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
    checkpoint(identity, stage='model_initializing')
    from vllm_runner import VllmRunner
    runner = VllmRunner(args.codec)
    try:
        for row in rows(ROOT / 'inputs/cases.jsonl'):
            name = f"{row['index']:06d}.json"
            output = dest / name
            if (ROOT/'failures'/identity/name).exists():
                continue
            if output.exists():
                record = json.loads(output.read_text())
                verify_import(output, record, frozen)
                assert record['input_ids_sha256'] == row['input_ids_sha256']
                assert record['backend'] == 'vllm'
            else:
                execution_id = uuid.uuid4().hex
                base = profiles / f"{row['index']:06d}_{execution_id}"
                checkpoint(identity, stage='request_active', row={k:row[k] for k in
                           ('index','case_id','dataset','input_ids_sha256','sampling_seed')},
                           execution_id=execution_id, profile_base=str(base))
                status('vllm_request_start', run=identity, index=row['index'], case=row['case_id'],
                       input_tokens=row['input_tokens'], execution_id=execution_id)
                atomic(base.with_suffix('.invocation.json'), dict(execution_id=execution_id,
                       case_id=row['case_id'], input_ids_sha256=row['input_ids_sha256'],
                       sampling_seed=row['sampling_seed'], run_fingerprint=frozen['fingerprint'],
                       actual_generation_executions=1))
                try:
                    record = runner.generate(row, base, execution_id, with_counters=True)
                except Exception:
                    error=traceback.format_exc()
                    record_failure(identity,row,error=error,exit_code=CASE_FAILURE_EXIT,
                                   execution_id=execution_id,profile_base=base,
                                   source_log=str(ROOT/'logs'/(identity+'.log')))
                    print(error,flush=True)
                    # A profiler/CUDA error can poison this context. Exit the
                    # worker without GPU cleanup; the parent starts a new one.
                    os._exit(CASE_FAILURE_EXIT)
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
            checkpoint(identity, stage='request_complete', index=row['index'])
        checkpoint(identity, stage='codec_finished')
    finally:
        runner.close()


if __name__ == '__main__':main()
