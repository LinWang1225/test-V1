"""Read-only paired error audit; keep the official scores unchanged."""
import ast
from collections import Counter
import json

from common import ROOT, PARENT, atomic, cases, sha


OBSERVATIONS = {
    'HumanEval/57': 'Uses strict monotonic comparisons; the task permits equal adjacent elements.',
    'HumanEval/126': 'Checks sorting but omits the restriction against three or more equal elements.',
    'HumanEval/128': 'Final code names the function prod_signss instead of the required prod_signs.',
    'HumanEval/153': 'Counts all non-uppercase characters as lowercase, including punctuation and digits.',
    'HumanEval/156': 'Returns uppercase Roman numerals; the task requires lowercase.',
    'test/algebra/2486.json': 'Interval is numerically equivalent, but 72/27 is unreduced; the task explicitly requires reduced fractions and the pinned official grader rejects them.'
}


def main():
    rows = {r['index']:r for r in cases()}
    summaries = []
    details = []
    for dataset in ('humaneval', 'math500'):
        counts = Counter()
        for sp in sorted((PARENT / 'results/Jacobian_Tree__fixed4/scores').glob('[0-9]*.json')):
            score = json.loads(sp.read_text())
            if score['dataset'] != dataset:
                continue
            gp = sp.parent.parent / sp.name
            bp = PARENT / 'results/fp16' / sp.name
            bs = bp.parent / 'scores' / sp.name
            generation = json.loads(gp.read_text())
            baseline = json.loads(bp.read_text())
            base_score = json.loads(bs.read_text())
            assert score['generation_sha256'] == sha(gp)
            assert base_score['generation_sha256'] == sha(bp)
            counts['n'] += 1
            counts['jacobian_correct'] += int(score['score'])
            counts['fp16_correct'] += int(base_score['score'])
            if score['score'] > base_score['score']:
                counts['gains'] += 1
            if score['score'] >= base_score['score']:
                continue
            counts['losses'] += 1
            counts['losses_finished_thinking' if generation['thinking_complete'] else 'losses_unfinished_thinking'] += 1
            case_id = generation['case_id']
            value = dict(case_id=case_id, index=generation['index'], dataset=dataset,
                category='unfinished_thinking' if not generation['thinking_complete'] else 'completed_but_failed',
                generated_tokens=generation['generated_tokens'], fp16_generated_tokens=baseline['generated_tokens'],
                finish_reason=generation['finish_reason'], thinking_complete=generation['thinking_complete'],
                final_response=generation['text'], fp16_final_response=baseline['text'],
                observation=OBSERVATIONS.get(case_id),
                generation_path=str(gp), generation_sha256=sha(gp),
                fp16_generation_path=str(bp), fp16_generation_sha256=sha(bp))
            if dataset == 'humaneval':
                completion = score['completion']
                try:
                    parsed = ast.parse(rows[generation['index']]['source_row']['prompt'] + completion)
                    names = [node.name for node in parsed.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
                except SyntaxError:
                    names = []
                value.update(expected_entry_point=rows[generation['index']]['source_row']['entry_point'],
                             parsed_function_names=names, execution_result=score['official_execution_result'])
            else:
                value.update(extracted_answer=score['extracted_answer'],
                             gold_answer=rows[generation['index']]['source_row']['answer'])
            details.append(value)
        summaries.append(dict(dataset=dataset, **counts))
    atomic(ROOT / 'analysis/jacobian_error_audit.json', dict(
        summaries=summaries, losses=details, official_scores_unchanged=True,
        numerical_causes_not_yet_isolated=[
            'Frozen calibration from short non-thinking trajectories may not match long thinking.',
            'Local attention-output surrogate does not optimize task correctness over the entire closed loop.',
            'All-layer quantization and altered logits can change the sampled trajectory even with an identical seed.',
            'A sampled synthetic layer0 writer/reader equivalence check is not a full 36-layer numerical audit.'
        ]))
    print(json.dumps(summaries, ensure_ascii=False))


if __name__ == '__main__':
    main()
