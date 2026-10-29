"""Freeze the agreed diagnostic workload, without answer scoring."""
import hashlib
import json
import random
import time
import zipfile
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
OLD = Path('/home/wanglin/kvbench-3090-cu126')
TASKS = ['qasper', 'hotpotqa', 'gov_report', 'trec', 'passage_retrieval_en', 'lcc']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    started = time.perf_counter()
    cfg = json.loads((ROOT / 'configs/operator_profile.json').read_text())
    dest = ROOT / 'data/operator_profile'
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'manifest.json').exists():
        manifest = json.loads((dest / 'manifest.json').read_text())
        assert manifest['config_sha256'] == sha(ROOT / 'configs/operator_profile.json')
        for name, info in manifest['datasets'].items():
            assert sha(dest / f'{name}.jsonl') == info['sha256']
        print('Frozen inputs verified; no resampling.', flush=True)
        return
    tok = AutoTokenizer.from_pretrained(cfg['model']['path'], local_files_only=True)
    tasks_file = OLD / 'kvbench_asym/longbench_prompts.json'
    tasks = json.loads(tasks_file.read_text())
    archive = OLD / 'data_sources/longbench_5e628be/data.zip'
    assert sha(archive) == 'cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64'
    manifest = {'config_sha256': sha(ROOT / 'configs/operator_profile.json'),
                'longbench_archive_sha256': sha(archive),
                'longbench_template_file_sha256': sha(tasks_file), 'datasets': {},
                'prompt_policy': 'existing task prompt, chat template with enable_thinking=False; diagnostic only',
                'human_math_source_manifest': json.loads((OLD / 'data/real_v5/manifest.json').read_text())}

    def length(messages):
        return len(tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False))

    def freeze(name, records):
        rows = []
        for rec in records:
            t0 = time.perf_counter()
            messages = rec.pop('messages')
            original = length(messages)
            original_messages_sha = hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode()).hexdigest()
            budget = cfg['execution']['max_model_len'] - cfg['execution']['fixed_generated_tokens']
            effective = original
            while effective > budget:
                # Preserve both ends, including the task/question suffix.
                content = messages[-1]['content']
                ids = tok.encode(content, add_special_tokens=False)
                keep = max(1, len(ids) - (effective - budget) - 16)
                messages[-1]['content'] = tok.decode(ids[:keep // 2], skip_special_tokens=False) + '\n...\n' + tok.decode(ids[-(keep - keep // 2):], skip_special_tokens=False)
                effective = length(messages)
            body = {'model': 'operator-profile', 'messages': messages,
                    'temperature': 0, 'seed': 42, 'max_tokens': 1024,
                    'min_tokens': 1024, 'ignore_eos': True,
                    'chat_template_kwargs': {'enable_thinking': False},
                    'request_id': rec['case_id']}
            rows.append({**rec, 'dataset': name, 'body': body,
                         'input_tokens': effective, 'original_input_tokens': original,
                         'truncated': original != effective,
                         'original_messages_sha256': original_messages_sha,
                         'prepare_and_tokenize_s': time.perf_counter() - t0})
        rows.sort(key=lambda r: (r['input_tokens'], r['case_id']))
        path = dest / f'{name}.jsonl'
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        manifest['datasets'][name] = {'count': len(rows), 'sha256': sha(path),
            'case_ids': [r['case_id'] for r in rows], 'truncated': sum(r['truncated'] for r in rows),
            'min_input_tokens': min(r['input_tokens'] for r in rows),
            'max_input_tokens': max(r['input_tokens'] for r in rows)}
        print(name, manifest['datasets'][name]['count'], 'frozen', flush=True)

    with zipfile.ZipFile(archive) as z:
        for task in TASKS:
            source = z.read(f'data/{task}.jsonl')
            raw = [json.loads(line) for line in source.decode().splitlines() if line.strip()]
            expanded = []
            for i, row in enumerate(raw):
                template = tasks[task]
                if isinstance(template, dict):
                    template = template['prompt']
                prompt = template.format(**row)
                expanded.append({'case_id': f'longbench/{task}/{i}', 'source_index': i,
                    'source_row_sha256': hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                    'messages': [{'role': 'user', 'content': prompt}]})
            expanded.sort(key=lambda r: (length(r['messages']), r['source_index']))
            rng = random.Random(42)
            selected = [rng.choice(expanded[i * len(expanded) // 50:(i + 1) * len(expanded) // 50]) for i in range(50)]
            freeze(task, selected)
    for filename, name, count in [('humaneval', 'humaneval', 164), ('math_500', 'math500', 500)]:
        source = OLD / f'data/real_v5/{filename}.jsonl'
        raw = [json.loads(s) for s in source.read_text().splitlines() if s.strip()]
        assert len(raw) == count and len({r['case_id'] for r in raw}) == count
        expected = manifest['human_math_source_manifest']['datasets'][filename]['sha256']
        assert sha(source) == expected
        freeze(name, [{'case_id': name + '/' + str(r['case_id']), 'source_index': r['source_index'],
                       'messages': r['messages'], 'source_record': r['original_record']} for r in raw])
    manifest['prepare_wall_s'] = time.perf_counter() - started
    manifest['total_samples_per_method'] = sum(v['count'] for v in manifest['datasets'].values())
    assert manifest['total_samples_per_method'] == 964
    (dest / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
