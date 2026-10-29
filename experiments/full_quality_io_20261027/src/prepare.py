"""Freeze complete official datasets and model-specific shared prompt token IDs."""
import argparse,gzip,json,zipfile
from common import *
from transformers import AutoTokenizer

NO_CHAT={'trec','triviaqa','samsum','lsht','lcc','repobench-p'}


def prepare():
    tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True)
    config=json.loads((ROOT/'config.json').read_text())
    templates=json.loads((ROOT/'vendor/longbench/config/dataset2prompt.json').read_text())
    limits=json.loads((ROOT/'vendor/longbench/config/dataset2maxlen.json').read_text())
    archive=OLD/'data_sources/longbench_5e628be/data.zip'
    math=list(jsonlines(ROOT/'inputs/math500.jsonl'))
    with gzip.open(ROOT/'inputs/HumanEval.jsonl.gz','rt') as f:human=[json.loads(s) for s in f if s.strip()]
    assert len(human)==164 and len(math)==500
    assert len({r['task_id'] for r in human})==164
    # Local copies used in earlier profiling are cross-checked for provenance.
    oldhuman={r['original_record']['task_id']:r['original_record'] for r in jsonlines(OLD/'data/real_v5/humaneval.jsonl')}
    assert all(all(oldhuman[r['task_id']][k]==r[k] for k in ('prompt','test','entry_point')) for r in human)
    oldmath={r['original_record']['unique_id']:r['original_record'] for r in jsonlines(OLD/'data/real_v5/math_500.jsonl')}
    assert all(r['unique_id'] in oldmath and oldmath[r['unique_id']]['problem']==r['problem'] and oldmath[r['unique_id']]['answer']==r['answer'] for r in math)
    frozen={};records=[]
    def freeze(dataset,source,prompt_fn,max_new,no_chat=False):
        rows=[]
        for i,r in enumerate(source):
            prompt=prompt_fn(r)
            if no_chat:ids=tok.encode(prompt,add_special_tokens=False)
            else:ids=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
            original=len(ids);budget=config['max_sequence_length']-max_new
            if original>budget:
                assert dataset not in ('humaneval','math500'),'Unexpected short-task prompt overflow'
                half=budget//2;ids=ids[:half]+ids[-(budget-half):]
            case=r.get('task_id',r.get('unique_id',f'longbench/{dataset}/{i}'))
            row=dict(case_id=case,dataset=dataset,index=i,input_ids=ids,input_tokens=len(ids),original_input_tokens=original,input_truncated=original!=len(ids),input_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),max_new_tokens=max_new,source_row=r,prompt_policy='raw official fewshot/code prompt' if no_chat else 'Qwen chat template, enable_thinking=False',prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
            rows.append(row)
        path=ROOT/'inputs/frozen'/f'{dataset}.jsonl'
        write_jsonlines(path,rows)
        frozen[dataset]=dict(count=len(rows),path=str(path.relative_to(ROOT)),sha256=sha(path),max_new_tokens=max_new,truncated_prompts=sum(r['input_truncated'] for r in rows),min_input_tokens=min(r['input_tokens'] for r in rows),max_input_tokens=max(r['input_tokens'] for r in rows))
        records.extend(rows)
        print('frozen',dataset,frozen[dataset],flush=True)
    freeze('humaneval',human,lambda r:'Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.\n'+r['prompt'],config['max_new_tokens']['humaneval'])
    freeze('math500',math,lambda r:r['problem']+'\nPlease reason step by step, and put your final answer within \\boxed{}.',config['max_new_tokens']['math500'])
    with zipfile.ZipFile(archive) as z:
        for dataset in limits:
            raw=[json.loads(line) for line in z.read(f'data/{dataset}.jsonl').decode().splitlines() if line.strip()]
            freeze(dataset,raw,lambda r:templates[dataset].format(**r),limits[dataset],dataset in NO_CHAT)
    assert len(limits)==21 and sum(frozen[n]['count'] for n in limits)==4750
    assert len(records)==5414
    eos=sorted(set([tok.eos_token_id,*json.loads((MODEL/'generation_config.json').read_text()).get('eos_token_id',[])]))
    atomic(ROOT/'inputs/manifest.json',dict(complete=True,datasets=frozen,total_cases_per_policy=len(records),policies=list(POLICIES),total_generations=len(records)*len(POLICIES),longbench_subtasks=21,longbench_archive_sha256=sha(archive),official_sources=load_sources(),eos_token_ids=eos,model_revision='1cfa9a7208912126459214e8b04321603b3df60c',prompt_tokenization_frozen=True,all_original_human_math_rows_crosschecked=True))
    status('complete_dataset_freeze',cases_per_policy=len(records),total_generations=len(records)*len(POLICIES))


def load_sources():return json.loads((ROOT/'vendor/official_sources.json').read_text())


def annotate():
    previous=json.loads((ROOT.parent/'encoding_ablation_20261024/inputs/manifest.json').read_text())
    calibration={r['case_id'] for r in previous if r['split']=='calibration'}
    explored={r['case_id'] for r in previous}
    manifest=json.loads((ROOT/'inputs/manifest.json').read_text());ncal=0;nseen=0
    for dataset,info in manifest['datasets'].items():
        path=ROOT/info['path'];data=list(jsonlines(path))
        for row in data:
            row['used_in_codec_calibration']=row['case_id'] in calibration
            row['previous_exploratory_case']=row['case_id'] in explored
        write_jsonlines(path,data);info['sha256']=sha(path)
        info['codec_calibration_overlap']=sum(r['used_in_codec_calibration'] for r in data)
        info['previous_exploratory_overlap']=sum(r['previous_exploratory_case'] for r in data)
        ncal+=info['codec_calibration_overlap'];nseen+=info['previous_exploratory_overlap']
    assert ncal==60 and nseen==180,(ncal,nseen)
    manifest['codec_calibration_overlap_cases']=ncal;manifest['previous_exploratory_overlap_cases']=nseen
    atomic(ROOT/'inputs/manifest.json',manifest)
    print('annotated calibration/exploratory overlap',ncal,nseen,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--annotate-only',action='store_true');a=p.parse_args()
    if not a.annotate_only:prepare()
    annotate()
