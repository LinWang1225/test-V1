"""Resumable EvalScope + worker-profiler controller; one method owns GPU 0."""
import argparse
import concurrent.futures
import contextlib
import csv
import datetime
import fcntl
import hashlib
import json
import os
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from collections import defaultdict
from pathlib import Path

from summarize_operator_trace import summarize

ROOT = Path(__file__).resolve().parents[1]
HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def api(base, endpoint, payload=None, timeout=1800):
    req = urllib.request.Request(base + endpoint,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'})
    with HTTP.open(req, timeout=timeout) as r:
        body = r.read()
        return json.loads(body) if body else None


class Controller:
    def __init__(self, run, cfg):
        self.run, self.cfg = run, cfg
        self.server = None
        self.server_log = None
        self.base = f"http://127.0.0.1:{cfg['execution']['port']}"
        self.state = {'status': 'starting', 'pid': os.getpid(), 'started_unix': time.time(),
                      'completed_points': 0, 'errors': []}
        self.state_lock = threading.Lock()
        self.stop_heartbeat = threading.Event()

    def update(self, **kw):
        with self.state_lock:
            self.state.update(kw)
            self.state['updated_unix'] = time.time()
            atomic(self.run / 'status.json', self.state)
        print(datetime.datetime.now().isoformat(timespec='seconds'), json.dumps(kw, ensure_ascii=False), flush=True)
        with (self.run / 'events.jsonl').open('a') as f:
            f.write(json.dumps({'time': time.time(), **kw}) + '\n')

    def heartbeat(self):
        while not self.stop_heartbeat.wait(15):
            free = shutil.disk_usage(self.run).free / 2**30
            gpu = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw',
                                  '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
            with self.state_lock:
                snapshot = dict(self.state)
            snapshot.update(heartbeat_unix=time.time(), disk_free_gib=free,
                            server_pid=self.server.pid if self.server else None,
                            gpu_csv=gpu.stdout.strip())
            atomic(self.run / 'heartbeat.json', snapshot)
            with (self.run / 'telemetry.jsonl').open('a') as f:
                f.write(json.dumps(snapshot) + '\n')

    def stop(self):
        if self.server is not None:
            proc = self.server
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=20)
            self.server = None
            self.server_log.close()
            self.server_log = None

    def start(self, method, label):
        self.stop()
        with socket.socket() as sock:
            if sock.connect_ex(('127.0.0.1', self.cfg['execution']['port'])) == 0:
                raise RuntimeError('Experiment port is occupied; refusing to kill unrelated process')
        apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                              capture_output=True, text=True, timeout=10).stdout.strip()
        if apps:
            raise RuntimeError(f'GPU has active compute processes {apps}; refusing to contaminate comparison')
        logdir = self.run / 'servers'
        logdir.mkdir(exist_ok=True)
        traces = self.run / 'incoming' / f'{method}_{label}'
        traces.mkdir(parents=True, exist_ok=True)
        profiler = {'profiler': 'torch', 'torch_profiler_dir': str(traces),
                    'torch_profiler_with_stack': False, 'torch_profiler_record_shapes': False,
                    'torch_profiler_with_memory': False, 'torch_profiler_use_gzip': True,
                    'torch_profiler_dump_cuda_time_total': False, 'ignore_frontend': True}
        e = self.cfg['execution']
        cmd = [str(ROOT / '.venv/bin/vllm'), 'serve', self.cfg['model']['path'],
               '--served-model-name', 'operator-profile', '--host', '127.0.0.1', '--port', str(e['port']),
               '--dtype', 'float16', '--kv-cache-dtype', self.cfg['methods'][method]['kv_cache_dtype'],
               '--max-model-len', str(e['max_model_len']), '--max-num-seqs', '8',
               '--max-num-batched-tokens', str(e['max_num_batched_tokens']),
               '--gpu-memory-utilization', str(e['gpu_memory_utilization']), '--block-size', '128',
               '--enforce-eager', '--no-enable-prefix-caching', '--seed', '42',
               '--profiler-config', json.dumps(profiler)]
        if method == 'fp8':
            cmd += ['--calculate-kv-scales', '--attention-backend', 'FLASHINFER']
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                   OPERATOR_PROFILE_SCOPES='1', VLLM_PLUGINS='operator_profile_scopes',
                   VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS='0', VLLM_NO_USAGE_STATS='1',
                   VLLM_CUSTOM_SCOPES_FOR_PROFILING='1', PYTHONUNBUFFERED='1',
                   no_proxy='127.0.0.1,localhost', NO_PROXY='127.0.0.1,localhost',
                   OMP_NUM_THREADS='8', TOKENIZERS_PARALLELISM='false')
        atomic(logdir / f'{method}_{label}.command.json', {'argv': cmd, 'profiler': profiler,
            'hook_source_sha256': file_sha(ROOT / 'scripts/operator_hooks/operator_hooks/__init__.py')})
        self.server_log = (logdir / f'{method}_{label}.log').open('a', buffering=1)
        self.server = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=self.server_log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        self.update(status='server_starting', method=method, server_log=str(logdir / f'{method}_{label}.log'))
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            if self.server.poll() is not None:
                raise RuntimeError(f'Server exited {self.server.returncode}; inspect {method}_{label}.log')
            try:
                api(self.base, '/health', timeout=3)
                self.update(status='server_ready', method=method)
                return traces
            except Exception:
                time.sleep(3)
        raise TimeoutError('Server startup timeout')

    def evalscope(self, rows, folder, tokens):
        folder.mkdir(parents=True, exist_ok=True)
        bodyfile = folder / 'requests.jsonl'
        with bodyfile.open('w') as f:
            for r in rows:
                body = dict(r['body'], max_tokens=tokens, min_tokens=tokens)
                f.write(json.dumps(body, ensure_ascii=False) + '\n')
        cmd = [str(ROOT / '.venv-eval/bin/evalscope'), 'perf', '--url', self.base + '/v1/chat/completions',
               '--api', 'openai', '--api-key', 'EMPTY', '--model', 'operator-profile',
               '--dataset', 'line_by_line', '--dataset-path', str(bodyfile),
               '--tokenizer-path', self.cfg['model']['path'], '--parallel', '8', '--number', str(len(rows)),
               '--stream', '--no-test-connection', '--warmup-num', '0', '--seed', '42',
               '--total-timeout', '7200', '--read-timeout', '7200', '--connect-timeout', '30',
               '--outputs-dir', str(folder / 'evalscope'), '--no-timestamp', '--name', 'capture',
               '--log-every-n-query', '1', '--max-prompt-length', '32768', '--min-prompt-length', '0',
               '--max-tokens', str(tokens), '--min-tokens', str(tokens), '--temperature', '0']
        atomic(folder / 'command.json', {'argv': cmd, 'case_ids': [r['case_id'] for r in rows]})
        env = dict(os.environ, HF_HUB_OFFLINE='1', no_proxy='localhost,127.0.0.1', NO_PROXY='localhost,127.0.0.1',
                   TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
        t0 = time.perf_counter()
        with (folder / 'evalscope.log').open('w') as log:
            result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=9000)
        if result.returncode:
            raise RuntimeError(f'EvalScope exited {result.returncode}; inspect {folder}/evalscope.log')
        databases = list(folder.rglob('benchmark_data.db'))
        if len(databases) != 1:
            raise RuntimeError(f'Expected one EvalScope database, found {databases}')
        with sqlite3.connect(databases[0]) as db:
            db.row_factory = sqlite3.Row
            result_rows = [dict(r) for r in db.execute('SELECT request_id,success,prompt_tokens,completion_tokens,latency,first_chunk_latency,time_per_output_token FROM result')]
        if len(result_rows) != len(rows) or not all(r['success'] and r['completion_tokens'] == tokens for r in result_rows):
            atomic(folder / 'request_validation_failure.json', result_rows)
            raise RuntimeError('Request count/success/generated token count mismatch')
        if sorted(r['prompt_tokens'] for r in result_rows) != sorted(r['input_tokens'] for r in rows):
            atomic(folder / 'prompt_validation_failure.json', result_rows)
            raise RuntimeError('Server prompt-token counts differ from frozen inputs')
        atomic(folder / 'request_metrics.json', {'wall_s': time.perf_counter() - t0, 'rows': result_rows,
            'interpretation': 'Profiler-on request latencies are diagnostic, not production benchmark results.'})
        return result_rows

    def point(self, method, point, rows, traces, tokens, warm=True):
        folder = self.run / 'points' / method / point['id']
        statepath = folder / 'state.json'
        if statepath.exists() and json.loads(statepath.read_text()).get('status') == 'complete':
            return
        folder.mkdir(parents=True, exist_ok=True)
        # Never overwrite partial evidence on resume.
        attempts = folder / 'attempts'
        attempts.mkdir(exist_ok=True)
        attempt = attempts / datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        attempt.mkdir()
        self.update(status='warming_up' if warm else 'capturing', method=method, point=point['id'],
                    dataset=point['dataset'], attempt=str(attempt), point_samples=len(rows))
        atomic(statepath, {'status': 'running', 'point': point, 'attempt': str(attempt), 'started': time.time()})
        if shutil.disk_usage(self.run).free < 25 * 2**30:
            raise RuntimeError('Less than 25 GiB disk free; preserving evidence and stopping before exhaustion')
        if warm:
            # Exercise the same prompts and at least one 128-token KVarN tile before capture.
            self.evalscope(rows, attempt / 'warmup', min(128, tokens))
        t0 = time.perf_counter()
        before = set(traces.rglob('*'))
        metrics_before = urllib.request.Request(self.base + '/metrics')
        with HTTP.open(metrics_before, timeout=30) as r:
            (attempt / 'metrics.before.txt').write_bytes(r.read())
        api(self.base, '/start_profile', {})
        self.update(status='capturing', method=method, point=point['id'], capture_started_unix=time.time())
        try:
            request_metrics = self.evalscope(rows, attempt / 'measured', tokens)
        except BaseException:
            # Preserve a partial trace when requests fail; do not mark it valid.
            try:
                api(self.base, '/stop_profile', {}, timeout=600)
            except Exception:
                pass
            raise
        self.update(status='exporting_trace', method=method, point=point['id'])
        export_start = time.perf_counter()
        api(self.base, '/stop_profile', {}, timeout=3600)
        export_s = time.perf_counter() - export_start
        with HTTP.open(self.base + '/metrics', timeout=30) as r:
            (attempt / 'metrics.after.txt').write_bytes(r.read())
        dest = attempt / 'traces'
        dest.mkdir()
        fresh = [p for p in traces.rglob('*') if p.is_file() and p not in before]
        for p in fresh:
            shutil.move(str(p), str(dest / p.name))
        self.update(status='summarizing_trace', method=method, point=point['id'])
        summary_start = time.perf_counter()
        summary = summarize(dest)
        if summary['phase_matched_fraction'] < .95:
            raise RuntimeError(f"Only {summary['phase_matched_fraction']:.1%} GPU events matched phase scopes")
        state = {'status': 'complete', 'point': point, 'attempt': str(attempt), 'finished': time.time(),
                 'capture_export_summary_s': time.perf_counter() - t0, 'export_s': export_s,
                 'summary_s': time.perf_counter() - summary_start,
                 'gpu_event_count': summary['gpu_event_count'], 'phase_matched_fraction': summary['phase_matched_fraction'],
                 'generated_tokens': sum(r['completion_tokens'] for r in request_metrics),
                 'trace_compressed_bytes': sum(r['compressed_bytes'] for r in summary['trace_files'])}
        atomic(statepath, state)
        self.update(status='point_complete', method=method, point=point['id'],
                    completed_points=self.state['completed_points'] + 1,
                    last_point=state)


def make_plan(data):
    points = []
    for dataset, rows in data.items():
        batches = [rows[i:i + 8] for i in range(0, len(rows), 8)]
        for i, batch in enumerate(batches):
            points.append({'id': f'{dataset}_b{i:03d}_r1', 'dataset': dataset, 'batch': i,
                           'repeat': 1, 'case_ids': [r['case_id'] for r in batch]})
        full = [i for i, batch in enumerate(batches) if len(batch) == 8]
        chosen = sorted({full[0], full[len(full) // 2], full[-1]})
        for rep in [2, 3]:
            for i in chosen:
                points.append({'id': f'{dataset}_b{i:03d}_r{rep}', 'dataset': dataset, 'batch': i,
                               'repeat': rep, 'case_ids': [r['case_id'] for r in batches[i]]})
    return points


def aggregate(run, methods):
    rows = []
    for method in methods:
        for statepath in sorted((run / 'points' / method).glob('*/state.json')):
            state = json.loads(statepath.read_text())
            if state['status'] != 'complete':
                continue
            point = state['point']
            summary = json.loads((Path(state['attempt']) / 'traces/kernel_summary.json').read_text())
            for c in summary['categories']:
                rows.append({'method': method, 'dataset': point['dataset'], 'point': point['id'],
                             'repeat': point['repeat'], 'requests': len(point['case_ids']), **c})
    if not rows:
        return
    dest = run / 'summary'
    dest.mkdir(exist_ok=True)
    with (dest / 'point_categories.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    totals = defaultdict(float)
    for r in rows:
        if r['repeat'] == 1:
            totals[(r['method'], r['dataset'], r['phase'], r['category'])] += r['total_gpu_ms']
    comparisons = []
    for (method, dataset, phase, category), value in totals.items():
        if method == 'fp16':
            continue
        base = totals.get(('fp16', dataset, phase, category))
        comparisons.append({'method': method, 'dataset': dataset, 'phase': phase, 'category': category,
            'gpu_ms': value, 'fp16_category_gpu_ms': base,
            'delta_gpu_ms': value - (base or 0), 'ratio_to_fp16': value / base if base else None,
            'note': 'Cumulative GPU durations, not critical-path savings. Category absence in FP16 is not isolated quantization cost.'})
    comparisons.sort(key=lambda r: r['delta_gpu_ms'], reverse=True)
    atomic(dest / 'fp16_category_comparison.json', comparisons)
    candidates = [{'priority': i + 1, **r} for i, r in enumerate(comparisons) if r['delta_gpu_ms'] > 0]
    atomic(dest / 'optimization_candidates.json', candidates)
    text = ['算子分析阶段结果', '', '以下为累计 GPU 时间的排查候选，不能直接视为可节省的端到端时间。',
            '融合读 KV / 反量化 / attention 不拆成虚构的独立时间；不同后端与存储位宽均是解释变量。', '',
            '代表批次 r2/r3 保存在 point_categories.csv，可检查稳定性；首轮完整数据用于下面的增量排序。', '']
    for r in candidates[:20]:
        text.append(f"{r['priority']}. {r['method']} / {r['dataset']} / {r['phase']} / {r['category']}: delta={r['delta_gpu_ms']:.3f} GPU-ms")
    (dest / 'optimization_readout.txt').write_text('\n'.join(text) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--aggregate-only', action='store_true')
    args = parser.parse_args()
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    cfgpath = ROOT / 'configs/operator_profile.json'
    cfg = json.loads(cfgpath.read_text())
    methods = list(cfg['methods'])
    if args.aggregate_only:
        aggregate(run, methods)
        return
    lockfile = (ROOT / 'locks/operator-experiment.lock').open('a')
    fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifestpath = ROOT / 'data/operator_profile/manifest.json'
    manifest = json.loads(manifestpath.read_text())
    assert manifest['config_sha256'] == file_sha(cfgpath)
    data = {}
    for name, info in manifest['datasets'].items():
        path = manifestpath.parent / f'{name}.jsonl'
        assert file_sha(path) == info['sha256']
        data[name] = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    points = make_plan(data)
    identity = {'config_sha256': file_sha(cfgpath), 'data_manifest_sha256': file_sha(manifestpath),
                'code_sha256': {p.name: file_sha(p) for p in [Path(__file__), ROOT / 'scripts/summarize_operator_trace.py', ROOT / 'scripts/operator_hooks/operator_hooks/__init__.py']}}
    if (run / 'identity.json').exists():
        assert json.loads((run / 'identity.json').read_text()) == identity, 'Run inputs/code changed; use a new run directory'
    atomic(run / 'identity.json', identity)
    atomic(run / 'config.json', cfg)
    atomic(run / 'plan.json', {'points_per_method': len(points), 'methods': methods, 'points': points,
        'full_samples_per_method': 964, 'main_tokens_per_request': 1024,
        'total_profiled_requests': len(methods) * sum(len(p['case_ids']) for p in points)})
    lookup = {r['case_id']: r for rows in data.values() for r in rows}
    ctrl = Controller(run, cfg)
    thread = threading.Thread(target=ctrl.heartbeat, daemon=True)
    thread.start()
    (run / 'controller.pid').write_text(str(os.getpid()) + '\n')
    ctrl.update(status='preflight', total_points=len(points) * len(methods), run_dir=str(run))

    def on_signal(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    try:
        # A trace-backed eight-request check for every method before the long run.
        pilot_rows = data['math500'][:8]
        for method in methods:
            marker = run / 'preflight' / f'{method}.json'
            if marker.exists() and json.loads(marker.read_text()).get('passed'):
                continue
            traces = ctrl.start(method, 'preflight')
            point = {'id': f'preflight_{method}', 'dataset': 'preflight', 'repeat': 0,
                     'case_ids': [r['case_id'] for r in pilot_rows]}
            ctrl.point(method, point, pilot_rows, traces, 128)
            ctrl.stop()
            atomic(marker, {'passed': True, 'tokens': 128, 'concurrency': 8, 'finished': time.time()})
        ctrl.update(status='preflight_passed')
        if args.preflight_only:
            return
        done = sum(1 for p in (run / 'points').glob('*/*/state.json')
                   if not p.parent.name.startswith('preflight') and json.loads(p.read_text()).get('status') == 'complete')
        ctrl.state['completed_points'] = done
        for method in methods:
            needed = [p for p in points if not (run / 'points' / method / p['id'] / 'state.json').exists()
                      or json.loads((run / 'points' / method / p['id'] / 'state.json').read_text()).get('status') != 'complete']
            if not needed:
                continue
            traces = ctrl.start(method, 'full_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
            for point in needed:
                rows = [lookup[cid] for cid in point['case_ids']]
                ctrl.point(method, point, rows, traces, 1024)
                aggregate(run, methods)
            ctrl.stop()
        aggregate(run, methods)
        ctrl.update(status='complete', completed_unix=time.time())
    except BaseException as exc:
        with (run / 'failure.log').open('a') as f:
            traceback.print_exc(file=f)
        ctrl.update(status='failed', error=str(exc), failure_log=str(run / 'failure.log'))
        raise
    finally:
        ctrl.stop()
        ctrl.stop_heartbeat.set()
        thread.join(timeout=20)


if __name__ == '__main__':
    main()
