"""Audit sampling distribution, seeded reproducibility, parsing and frozen panel."""
import torch
from transformers import AutoTokenizer, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper
from common import *
from sampling import sample_token, split_response


def main():
    torch.set_num_threads(4)
    torch.manual_seed(127)
    logits=torch.randn(3,1,1000)
    scores=logits[:,-1].float()
    for warper in [TemperatureLogitsWarper(.6),TopKLogitsWarper(20),TopPLogitsWarper(.95)]:
        scores=warper(torch.zeros(3,1,dtype=torch.long),scores)
    values,indices=torch.topk(logits[:,-1]/.6,20,sorted=True)
    probs=values.softmax(-1)
    values=values.masked_fill(probs.cumsum(-1)-probs>=.95,-torch.inf)
    actual=torch.zeros_like(scores).scatter(-1,indices,values.softmax(-1))
    torch.testing.assert_close(actual,scores.softmax(-1),rtol=2e-6,atol=1e-7)
    gen=torch.Generator().manual_seed(123)
    sequence=[sample_token(logits,gen) for _ in range(16)]
    gen2=torch.Generator().manual_seed(123)
    first=[sample_token(logits,gen2) for _ in range(7)]
    continued=[sample_token(logits,gen2) for _ in range(9)]
    assert all(torch.equal(a,b) for a,b in zip(sequence,first+continued))
    tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True)
    finished=split_response(tok,tok.encode('reasoning\n</think>\n42',add_special_tokens=False))
    assert finished['thinking_complete'] and finished['text'].strip()=='42'
    incomplete=split_response(tok,tok.encode('uncompleted reasoning',add_special_tokens=False))
    assert not incomplete['thinking_complete'] and incomplete['text']==''
    manifest=json.loads((ROOT/'inputs/manifest.json').read_text())
    panel=list(rows(ROOT/'inputs/cases.jsonl'))
    assert len(panel)==464 and manifest['generation']==GENERATION
    assert len({r['sampling_seed'] for r in panel})==464
    assert all(not r['input_truncated'] and r['input_tokens']<manifest['native_context_length'] for r in panel)
    assert len(manifest['longbench_quotas'])==21 and sum(manifest['longbench_quotas'].values())==200
    close=tok.convert_tokens_to_ids('</think>')
    assert all(close not in r['input_ids'][-5:] for r in panel)
    atomic(ROOT/'checks_sampling.json',dict(passed=True,hf_probability_equivalence=True,
        request_local_seed_reproducibility=True,unfinished_thinking_empty_answer=True,
        same_464_ids=True,all_21_longbench=True,no_task_length_caps=True))
    print('Sampling, seeded reproducibility, thinking extraction, complete panel audits passed',flush=True)


if __name__=='__main__':main()
