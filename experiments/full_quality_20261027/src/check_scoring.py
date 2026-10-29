"""Verify official scorer execution and extraction on canonical answers."""
from common import *
from score import *


def main():
    sandbox.full_namespace_check();sandbox.sandbox_self_test(Path('/usr/bin/python3'))
    humans=list(jsonlines(ROOT/'inputs/frozen/humaneval.jsonl'))
    for i,row in enumerate(humans):
        p=row['source_row']
        canonical=p['prompt']+p['canonical_solution']
        result=human_score('```python\n'+canonical+'\n```',p)
        assert result['score']==1,(row['case_id'],result)
        if i%32==0:print('official canonical HumanEval',i+1,'/164',flush=True)
    # Wrong code must fail; passing every candidate would invalidate the run.
    p=humans[0]['source_row'];bad=human_score(f'def {p["entry_point"]}(*args, **kwargs):\n    return None',p)
    assert bad['score']==0
    maths=list(jsonlines(ROOT/'inputs/frozen/math500.jsonl'))
    canonical_failures=[]
    for i,row in enumerate(maths):
        problem=row['source_row'];result=math_score(problem['solution'],problem)
        if result['score']!=1:canonical_failures.append(dict(case=row['case_id'],extracted=result,answer=problem['answer']))
        assert math_score('\\boxed{'+problem['answer']+'}',problem)['score']==1
    assert not canonical_failures,canonical_failures
    assert math_score('\\boxed{2}',{'answer':'3'})['score']==0
    assert math_score('\\boxed{0.5}',{'answer':'\\frac{1}{2}'})['score']==1
    official=longbench_module();counts={}
    for name in official.dataset2metric:
        rs=list(jsonlines(ROOT/'inputs/frozen'/f'{name}.jsonl'))[:2]
        predictions=[r['source_row']['answers'][0] for r in rs]
        ours=[longbench_score(p,r['source_row'],name)['score'] for p,r in zip(predictions,rs)]
        expected=official.scorer(name,predictions,[r['source_row']['answers'] for r in rs],rs[-1]['source_row']['all_classes'])
        assert round(100*sum(ours)/len(ours),2)==expected,(name,ours,expected)
        counts[name]=len(ours)
    atomic(ROOT/'scoring_checks.json',dict(passed=True,official_human_canonical_cases=164,human_wrong_code_rejected=True,math_canonical_solution_cases=500,math_boxed_gold_cases=500,math_equivalence_fraction_decimal=True,longbench_official_aggregate_parity=counts,sandbox_selftest=True))
    print('All official scoring gates passed',flush=True)


if __name__=='__main__':main()
