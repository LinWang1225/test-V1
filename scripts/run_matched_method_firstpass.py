"""Run a selected KV method against an immutable selection of completed FP16 points."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import threading
import time
import traceback

from run_operator_experiment import ROOT, Controller, aggregate, atomic, file_sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--method', choices=['turboquant', 'fp8'], required=True)
    args = parser.parse_args()
    method = args.method
    source, run = args.source.resolve(), args.run_dir.resolve()
    lock = (ROOT / 'locks/operator-experiment.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    run.mkdir(parents=True, exist_ok=True)
    original = json.loads((source / 'identity.json').read_text())
    assert file_sha(ROOT / 'configs/operator_profile.json') == original['config_sha256']
    assert file_sha(ROOT / 'data/operator_profile/manifest.json') == original['data_manifest_sha256']
    for path in [ROOT / 'scripts/run_operator_experiment.py', ROOT / 'scripts/summarize_operator_trace.py',
                 ROOT / 'scripts/operator_hooks/operator_hooks/__init__.py']:
        assert file_sha(path) == original['code_sha256'][path.name], path
    cfg = json.loads((source / 'config.json').read_text())
    source_plan = json.loads((source / 'plan.json').read_text())
    if (run / 'plan.json').exists():
        points = json.loads((run / 'plan.json').read_text())['points']
    else:
        points = []
        for point in source_plan['points']:
            statepath = source / 'points/fp16' / point['id'] / 'state.json'
            if statepath.exists():
                state = json.loads(statepath.read_text())
                if state['status'] == 'complete':
                    assert state['point'] == point
                    points.append(point)
    points = [point for point in points if point['repeat'] == 1]
    assert points
    provenance = {}
    for point in points:
        src = source / 'points/fp16' / point['id']
        state = json.loads((src / 'state.json').read_text())
        assert state['status'] == 'complete' and state['point'] == point
        assert (Path(state['attempt']) / 'traces/kernel_summary.json').exists()
        provenance[point['id']] = file_sha(src / 'state.json')
        dest = run / 'points/fp16' / point['id']
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.symlink_to(src, target_is_directory=True)
        assert dest.resolve() == src
    identity = {'source': str(source), 'source_identity': original,
                'fp16_state_sha256': provenance, 'runner_sha256': file_sha(Path(__file__))}
    if (run / 'identity.json').exists():
        assert json.loads((run / 'identity.json').read_text()) == identity
    atomic(run / 'identity.json', identity)
    atomic(run / 'config.json', cfg)
    unique = {cid for p in points for cid in p['case_ids']}
    atomic(run / 'plan.json', {'methods': ['fp16', method], 'points': points,
        'points_per_method': len(points), 'full_samples_per_method': len(unique),
        'main_tokens_per_request': 1024,
        'total_profiled_requests': 2 * sum(len(p['case_ids']) for p in points),
        'selection': 'Completed FP16 first-pass points only; all extra repeats cancelled by user; baseline linked, not rerun.'})
    manifest = json.loads((ROOT / 'data/operator_profile/manifest.json').read_text())
    lookup = {}
    for name, info in manifest['datasets'].items():
        path = ROOT / 'data/operator_profile' / f'{name}.jsonl'
        assert file_sha(path) == info['sha256']
        for line in path.read_text().splitlines():
            row = json.loads(line)
            lookup[row['case_id']] = row
    ctrl = Controller(run, cfg)
    thread = threading.Thread(target=ctrl.heartbeat, daemon=True)
    thread.start()
    (run / 'controller.pid').write_text(str(os.getpid()) + '\n')
    def stop(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    done = len(points)
    needed = []
    for point in points:
        path = run / 'points' / method / point['id'] / 'state.json'
        if path.exists() and json.loads(path.read_text())['status'] == 'complete':
            done += 1
        else:
            needed.append(point)
    ctrl.state['completed_points'] = done
    ctrl.update(status='starting', total_points=len(points) * 2, run_dir=str(run),
                baseline_source=str(source), unique_samples_per_method=len(unique))
    try:
        preflight = json.loads((source / 'preflight' / f'{method}.json').read_text())
        assert preflight['passed']
        atomic(run / 'preflight' / f'{method}.json', {**preflight, 'reused_from': str(source)})
        if needed:
            traces = ctrl.start(method, 'matched_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
            for point in needed:
                ctrl.point(method, point, [lookup[c] for c in point['case_ids']], traces, 1024)
                aggregate(run, ['fp16', method])
        aggregate(run, ['fp16', method])
        ctrl.update(status='complete', completed_unix=time.time())
    except BaseException as exc:
        with (run / 'failure.log').open('a') as f:
            traceback.print_exc(file=f)
        ctrl.update(status='failed', error=str(exc))
        raise
    finally:
        ctrl.stop()
        ctrl.stop_heartbeat.set()
        thread.join(timeout=20)


if __name__ == '__main__':
    main()
