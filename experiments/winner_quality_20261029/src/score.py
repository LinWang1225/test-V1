"""Pinned official task scorers, raw outputs retained, no answer-conditioned fixes."""
import argparse,ast,contextlib,importlib.util,re,signal,sys
from common import *
sys.path.insert(0,str(ROOT/'vendor'))
import sandbox
from grading import grader


def final_math(text):
    # Freeze one response-only extraction rule; never inspect the gold answer.
    text=text.split('</think>')[-1]
    positions=list(re.finditer(r'\\(?:boxed|fbox)\s*\{',text))
    for m in reversed(positions):
        start=m.end();depth=1
        for i in range(start,len(text)):
            if text[i]=='{':depth+=1
            elif text[i]=='}':
                depth-=1
                if depth==0:return text[start:i].strip(),'last_boxed'
    # Non-boxed final line remains scoreable but is explicitly flagged.
    lines=[s.strip() for s in text.strip().splitlines() if s.strip()]
    if not lines:return None,'empty'
    value=re.sub(r'^(?:final\s+answer|answer)\s*(?:is|:|=)\s*','',lines[-1],flags=re.I)
    return value.strip().strip('$').strip(),'last_nonempty_line'


def human_completion(text,problem):
    text=text.split('</think>')[-1].strip('\n')
    blocks=re.findall(r'```(?:python|py)?\s*\n(.*?)```',text,re.S|re.I)
    if blocks:text=blocks[0]
    elif text.lstrip().startswith('```'):text=re.sub(r'^```[^\n]*\n','',text.lstrip()).split('```')[0]
    # Full-function chat outputs become valid completions after the original
    # empty stub. No model function body is edited or selected using tests.
    if re.search(r'^\s*(?:async\s+)?def\s+'+re.escape(problem['entry_point'])+r'\s*\(',text,re.M):
        return '    pass\n\n'+text.strip()+'\n','full_function_after_stub'
    if text and not text[0].isspace():
        return '\n'.join('    '+s if s else '' for s in text.splitlines())+'\n','indent_body'
    return text+'\n','body_completion'


def human_score(text,problem):
    completion,rule=human_completion(text,problem)
    official=(ROOT/'vendor/human_eval/execution.py').read_text()
    # Official unsafe_execute includes exact prompt+completion+test composition,
    # reliability guard, swallowed I/O and official time limit; outer sandbox
    # supplies filesystem/network/process isolation and a hard wall deadline.
    program='\n'.join([
        'namespace = {}',f'exec(compile({official!r}, "official_humaneval_execution.py", "exec"), namespace)',
        'results = []',f'namespace["unsafe_execute"]({problem!r}, {completion!r}, 3.0, results)',
        'assert results == ["passed"], repr(results)',
    ])
    result=sandbox.sandbox_run(program,Path('/usr/bin/python3'),8)
    return dict(score=float(bool(result.get('passed'))),grading_status='graded',extraction_rule=rule,completion=completion,official_execution_result=result)


@contextlib.contextmanager
def time_limit(seconds):
    def timeout(signum,frame):raise TimeoutError('grader wall time limit')
    previous=signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,seconds)
    try:yield
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)


def math_score(text,problem):
    answer,rule=final_math(text)
    with time_limit(30):correct=grader.grade_answer(answer,problem['answer'])
    return dict(score=float(bool(correct)),grading_status='graded',extraction_rule=rule,extracted_answer=answer)


def longbench_module():
    # Avoid collisions with sandbox or unrelated modules called "metrics".
    spec=importlib.util.spec_from_file_location('official_longbench_metrics',ROOT/'vendor/longbench/metrics.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    saved=sys.modules.get('metrics');sys.modules['metrics']=module
    try:
        spec=importlib.util.spec_from_file_location('official_longbench_eval',ROOT/'vendor/longbench/eval.py')
        evaluator=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluator)
    finally:
        if saved is None:sys.modules.pop('metrics',None)
        else:sys.modules['metrics']=saved
    return evaluator


_LB=None
def longbench_score(text,problem,dataset):
    global _LB
    if _LB is None:_LB=longbench_module()
    prediction=text
    if dataset in ['trec','triviaqa','samsum','lsht']:prediction=prediction.lstrip('\n').split('\n')[0]
    score=max((_LB.dataset2metric[dataset](prediction,a,all_classes=problem['all_classes']) for a in problem['answers']),default=0.)
    return dict(score=float(score),grading_status='graded',official_task_metric=_LB.dataset2metric[dataset].__name__,scoring_prediction=prediction)


def score_dataset(policy,dataset,pilot=False):
    manifest=json.loads((ROOT/'inputs/manifest.json').read_text())
    assert sha(ROOT/'inputs/frozen'/f'{dataset}.jsonl')==manifest['datasets'][dataset]['sha256']
    frozen=list(jsonlines(ROOT/'inputs/frozen'/f'{dataset}.jsonl'))
    paths=sorted((ROOT/'results'/('pilot' if pilot else 'full')/policy/dataset).glob('[0-9]*.json'))
    if not pilot:assert len(paths)==len(frozen),(policy,dataset,len(paths),len(frozen))
    results=[];generations=[]
    for path in paths:
        record=json.loads(path.read_text());row=frozen[record['index']]
        generations.append(record)
        assert record['input_ids_sha256']==row['input_ids_sha256']
        dest=path.parent/'scores'/path.name
        if dest.exists():
            result=json.loads(dest.read_text());assert result['generation_sha256']==sha(path)
        else:
            try:
                if dataset=='humaneval':graded=human_score(record['text'],row['source_row'])
                elif dataset=='math500':graded=math_score(record['text'],row['source_row'])
                else:graded=longbench_score(record['text'],row['source_row'],dataset)
            except (TimeoutError,Exception) as exc:
                # Infrastructure/parse exceptions are unknown, not silent zeros.
                # Preserve evidence and stop the queue for repair.
                atomic(path.parent/'grading_error.json',dict(case_id=row['case_id'],error=repr(exc),generation_sha256=sha(path)))
                raise
            result=dict(case_id=row['case_id'],dataset=dataset,index=row['index'],policy=policy,generation_sha256=sha(path),**graded)
            atomic(dest,result)
        results.append(result)
    assert all(r['grading_status']=='graded' for r in results)
    mean=sum(r['score'] for r in results)/max(len(results),1)
    summary=dict(complete=not pilot,n_cases=len(results),expected_cases=len(frozen),policy=policy,dataset=dataset,score=mean,score_percent=100*mean,metric='pass@1' if dataset=='humaneval' else 'answer_accuracy' if dataset=='math500' else 'official_longbench_task_metric',pilot=pilot,truncated_prompts=sum(json.loads(p.read_text())['input_truncated'] for p in paths),hit_output_limit=sum(json.loads(p.read_text())['hit_output_limit'] for p in paths))
    summary['mean_generated_tokens']=sum(r['generated_tokens'] for r in generations)/max(len(generations),1)
    summary['natural_eos_stops']=sum(r['finish_reason']=='eos' for r in generations)
    summary['generation_wall_seconds']=sum(r['elapsed_seconds'] for r in generations)
    not_cal=[r for r,g in zip(results,generations) if not g.get('used_in_codec_calibration',False)]
    summary['non_calibration_cases']=len(not_cal)
    summary['non_calibration_score_percent']=100*sum(r['score'] for r in not_cal)/len(not_cal) if not_cal else None
    atomic(ROOT/'results'/('pilot' if pilot else 'full')/policy/dataset/'score_summary.json',summary)
    print('scored',summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--policy',choices=POLICIES,required=True);p.add_argument('--dataset',required=True);p.add_argument('--pilot',action='store_true');a=p.parse_args();score_dataset(a.policy,a.dataset,a.pilot)
