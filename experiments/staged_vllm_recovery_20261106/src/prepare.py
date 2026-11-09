"""Retokenize the same frozen 464 IDs; complete chat prompts, thinking enabled."""
from common import *
from transformers import AutoTokenizer


def main():
    source = EXPERIMENTS/'staged_algorithm_20261102/inputs'
    old_manifest = json.loads((source/'manifest.json').read_text())
    assert sha(source/'cases.jsonl') == old_manifest['cases_sha256']
    old = list(rows(source/'cases.jsonl'))
    dest = ROOT/'inputs/cases.jsonl'
    if dest.exists():
        manifest = json.loads((ROOT/'inputs/manifest.json').read_text())
        assert manifest['generation'] == GENERATION
        assert sha(dest) == manifest['cases_sha256']
        assert manifest['parent_cases_sha256'] == old_manifest['cases_sha256']
        print('Frozen sampling inputs verified', flush=True)
        return
    tok = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    templates = json.loads((LEGACY/'vendor/longbench/config/dataset2prompt.json').read_text())
    native = json.loads((MODEL/'config.json').read_text())['max_position_embeddings']
    selected = []
    for prior in old:
        row = dict(prior)
        dataset, raw = row['dataset'], row['source_row']
        if dataset == 'humaneval':
            prompt = 'Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.\n'+raw['prompt']
        elif dataset == 'math500':
            prompt = raw['problem']+'\nPlease reason step by step, and put your final answer within \\boxed{}.'
        else:
            prompt = templates[dataset].format(**raw)
        assert hashlib.sha256(prompt.encode()).hexdigest() == prior['prompt_sha256']
        ids = tok.apply_chat_template([dict(role='user', content=prompt)], tokenize=True,
                                      add_generation_prompt=True, enable_thinking=True)
        if len(ids) >= native:
            raise ValueError(f"Full thinking input exceeds model boundary: {row['case_id']}")
        row.update(parent_input_ids_sha256=prior['input_ids_sha256'], input_ids=ids,
                   input_tokens=len(ids), input_ids_sha256=digest(ids), input_truncated=False,
                   prompt_policy='Qwen chat template, enable_thinking=True for all tasks',
                   sampling_seed=int(digest([GENERATION['seed'], row['case_id']])[:16],16) % (2**63-1))
        selected.append(row)
    assert len(selected) == 464
    assert [r['case_id'] for r in selected] == [r['case_id'] for r in old]
    dest.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in selected))
    manifest = dict(old_manifest)
    manifest.update(schema_version=2, cases_sha256=sha(dest), parent_cases_sha256=old_manifest['cases_sha256'],
                    generation=GENERATION, native_context_length=native, artificial_output_limit=None,
                    input_truncation=False, all_tasks_chat_template=True,
                    sample_identity='Same frozen IDs and source prompts; all retokenized with thinking enabled',
                    physical_measurement_required=True,
                    physical_scope='layer17 first actual decode attention per request, warm and evicted, not whole generation')
    atomic(ROOT/'inputs/manifest.json', manifest)
    atomic(ROOT/'inputs/codebook_provenance.json', dict(
        reused_frozen_calibration=True, parent=str(EXPERIMENTS/'staged_algorithm_20261102/codebooks'),
        note='Historical 60-request calibration is not refit on thinking test trajectories.',
        files={str(p.relative_to(ROOT/'codebooks')):sha(p) for p in sorted((ROOT/'codebooks').glob('*/*.npz'))}))
    print('Frozen 464 complete thinking prompts, including 200 LongBench', flush=True)


if __name__ == '__main__':
    main()
