"""Freeze 464 IDs, stratified samples and full (untruncated) prompts."""
from collections import Counter, defaultdict
import gzip
import zipfile
import numpy as np
from common import *
from transformers import AutoTokenizer

def stratified(records, total, key, rng):
    groups = defaultdict(list)
    for r in records:
        groups[key(r)].append(r)
    keys = sorted(groups, key=str)
    exact = {k: total * len(groups[k]) / len(records) for k in keys}
    quotas = {k: int(exact[k]) for k in keys}
    for k in sorted(keys, key=lambda k: (-(exact[k] - quotas[k]), str(k)))[:total-sum(quotas.values())]:
        quotas[k] += 1
    output = []
    for k in keys:
        order = rng.permutation(len(groups[k]))[:quotas[k]]
        output.extend(groups[k][int(i)] for i in order)
    assert len(output) == total
    return output

def main():
    manifest_path = ROOT / 'inputs/manifest.json'
    if manifest_path.exists():
        m = json.loads(manifest_path.read_text())
        assert sha(ROOT / 'inputs/cases.jsonl') == m['cases_sha256']
        assert len(list(rows(ROOT / 'inputs/cases.jsonl'))) == 464
        print('frozen manifest already verified', flush=True)
        return
    tok = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    native = json.loads((MODEL / 'config.json').read_text())['max_position_embeddings']
    original = json.loads((BASE / 'inputs/manifest.json').read_text())
    calibration = {r['case_id'] for r in original if r['split'] == 'calibration'}
    explored = {r['case_id'] for r in original}
    templates = json.loads((LEGACY / 'vendor/longbench/config/dataset2prompt.json').read_text())
    datasets = sorted(templates)
    assert len(datasets) == 21
    rng = np.random.default_rng(SEED)
    selected = []
    no_chat = {'trec', 'triviaqa', 'samsum', 'lsht', 'lcc', 'repobench-p'}
    def freeze(dataset, source_index, record, prompt, raw=False):
        ids = tok.encode(prompt, add_special_tokens=False) if raw else tok.apply_chat_template(
            [{'role': 'user', 'content': prompt}], tokenize=True, add_generation_prompt=True, enable_thinking=False)
        if len(ids) >= native:
            return None
        case = record.get('task_id', record.get('unique_id', f'longbench/{dataset}/{source_index}'))
        return dict(case_id=case, dataset=dataset, source_index=source_index,
                    input_ids=ids, input_tokens=len(ids), input_ids_sha256=digest(ids),
                    prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), source_row=record,
                    input_truncated=False, used_in_codec_calibration=case in calibration,
                    previous_exploratory_case=case in explored)
    with gzip.open(LEGACY / 'inputs/HumanEval.jsonl.gz', 'rt') as f:
        human = [json.loads(x) for x in f if x.strip()]
    for i, r in enumerate(human):
        selected.append(freeze('humaneval', i, r, 'Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.\n' + r['prompt']))
    maths = list(enumerate(rows(LEGACY / 'inputs/math500.jsonl')))
    math_sample = stratified(maths, 100, lambda x: (x[1]['subject'], x[1]['level']), rng)
    for i, r in sorted(math_sample):
        selected.append(freeze('math500', i, r, r['problem'] + '\nPlease reason step by step, and put your final answer within \\boxed{}.'))
    archive = Path('/home/wanglin/kvbench-3090-cu126/data_sources/longbench_5e628be/data.zip')
    # 11 tasks receive 10 cases, 10 tasks receive 9; the seed fixes extra quotas.
    extra = set(rng.choice(datasets, size=11, replace=False).tolist())
    quotas = {d: 9 + int(d in extra) for d in datasets}
    exclusions = {}
    with zipfile.ZipFile(archive) as z:
        for dataset in datasets:
            candidates = []
            counts = Counter()
            raw_records = [json.loads(x) for x in z.read(f'data/{dataset}.jsonl').decode().splitlines() if x.strip()]
            for i, r in enumerate(raw_records):
                case = f'longbench/{dataset}/{i}'
                if case in calibration:
                    counts['calibration_overlap'] += 1
                    continue
                frozen = freeze(dataset, i, r, templates[dataset].format(**r), dataset in no_chat)
                if frozen is None:
                    counts['input_exceeds_native_context'] += 1
                    continue
                candidates.append(frozen)
            # Four within-task length strata, based on this tokenizer's full prompt.
            candidates.sort(key=lambda r: (r['input_tokens'], r['source_index']))
            ranks = {r['case_id']: min(3, i * 4 // len(candidates)) for i, r in enumerate(candidates)}
            picked = stratified(candidates, quotas[dataset], lambda r: ranks[r['case_id']], rng)
            selected.extend(sorted(picked, key=lambda r: r['source_index']))
            exclusions[dataset] = dict(counts)
    assert len(selected) == 464 and all(r is not None for r in selected)
    assert len({r['case_id'] for r in selected}) == 464
    assert not any(r['used_in_codec_calibration'] for r in selected)
    for i, r in enumerate(selected):
        r['index'] = i
    path = ROOT / 'inputs/cases.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in selected))
    eos = json.loads((MODEL / 'generation_config.json').read_text())['eos_token_id']
    eos = [eos] if isinstance(eos, int) else eos
    atomic(manifest_path, dict(schema_version=1, seed=SEED, total_cases=464,
        datasets=dict(Counter(r['dataset'] for r in selected)), longbench_quotas=quotas,
        math_subject_level_counts=dict(Counter(f"{r['source_row']['subject']}/L{r['source_row']['level']}" for r in selected if r['dataset']=='math500')),
        cases_sha256=sha(path), native_context_length=native, artificial_output_limit=None,
        eos_token_ids=sorted(set([tok.eos_token_id, *eos])), input_truncation=False,
        source_hashes={str(p): sha(p) for p in [archive, LEGACY/'inputs/math500.jsonl', LEGACY/'inputs/HumanEval.jsonl.gz', LEGACY/'vendor/longbench/config/dataset2prompt.json']},
        exclusions=exclusions, historical_exploratory_overlap=sum(r['previous_exploratory_case'] for r in selected),
        calibration_overlap=0, interpretation='Frozen development evaluation panel; repeated method selection is exploratory.'))
    status('samples_frozen', total=464, longbench=200, native_context=native)

if __name__ == '__main__':
    main()
