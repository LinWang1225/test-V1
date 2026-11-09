"""Explicit Qwen sampling with request-local RNG and response-only extraction."""
import torch
from common import GENERATION


def sample_token(logits, generator):
    scores = logits[:, -1, :].float() / GENERATION['temperature']
    values, indices = torch.topk(scores, GENERATION['top_k'], dim=-1, sorted=True)
    probs = values.softmax(-1)
    # Keep the first token crossing top-p, matching HF's nucleus boundary.
    remove = probs.cumsum(-1) - probs >= GENERATION['top_p']
    probs = values.masked_fill(remove, -torch.inf).softmax(-1)
    chosen = torch.multinomial(probs, 1, generator=generator)
    return indices.gather(-1, chosen)


def split_response(tokenizer, ids):
    close = tokenizer.convert_tokens_to_ids('</think>')
    positions = [i for i, token in enumerate(ids) if token == close]
    raw = tokenizer.decode(ids, skip_special_tokens=False)
    if not positions:
        return dict(text='', thinking_text=tokenizer.decode(ids, skip_special_tokens=True),
                    raw_text=raw, thinking_complete=False, thinking_tokens=len(ids), final_tokens=0)
    boundary = positions[-1]
    return dict(text=tokenizer.decode(ids[boundary+1:], skip_special_tokens=True),
                thinking_text=tokenizer.decode(ids[:boundary], skip_special_tokens=True),
                raw_text=raw, thinking_complete=True, thinking_tokens=boundary+1,
                final_tokens=len(ids)-boundary-1)
