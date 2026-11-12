import random
from datasets import load_dataset
from config import *

TLDR = "TL;DR:"

def clean_post(raw_prompt):
    """Keep SUBREDDIT / TITLE /POST, drop anything from 'TL;DR:' on,
    and end with one fixed marker the model learns to continue."""
    post = raw_prompt.split(TLDR)[0].strip()
    return post + "\n" + TLDR

def clean_summary(raw):
    """Strip a leading 'TL;DR:', cut stray text after <|endoftext|>,
    and start with exactly one space (it follows 'TL;DR:')."""
    text = raw.split("<|endoftext|>")[0].strip()
    if text.startswith(TLDR):
        text = text[len(TLDR):].strip()
    return " " + text

def _sample_valid(rows, n, tok, make, seed):
    """Shuffle rows with a fixed seed and keep the first n that pass
    the length filter. Each kept item is a dict of token-id lists."""
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    kept = []

    for start in range(0, len(order), 2000):
        for i in order[start:start + 2000]:
            item = make(rows[i], tok)
            if item is not None:
                kept.append(item)
            if len(kept) == n:
                return kept

    return kept

def _make_sft(row, tok):
    prompt = clean_post(row["prompt"])
    summary = clean_summary(row["label"])

    p = tok(prompt, add_special_tokens=False)["input_ids"]
    s = tok(summary, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]

    if len(summary.strip()) == 0 or len(p) > MAX_PROMPT_TOKENS or len(s) > MAX_SUMMARY_TOKENS:
        return None

    return {
        "prompt_ids": p,
        "summary_ids": s,
        "prompt": prompt,
        "summary": summary
    }

def _make_pref(row, tok):
    prompt = clean_post(row["prompt"])
    ch = clean_summary(row["chosen"])
    rj = clean_summary(row["rejected"])

    if ch == rj or not ch.strip() or not rj.strip():
        return None

    p = tok(prompt, add_special_tokens=False)["input_ids"]
    c = tok(ch, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    r = tok(rj, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]

    if len(p) > MAX_PROMPT_TOKENS or max(len(c), len(r)) > MAX_SUMMARY_TOKENS:
        return None

    return {
        "prompt_ids": p,
        "chosen_ids": c,
        "rejected_ids": r,
        "prompt": prompt,
        "chosen": ch,
        "rejected": rj
    }

def load_sft_splits(tok):
    ds = load_dataset(SFT_DATA)
    train = _sample_valid(ds["train"], N_SFT_TRAIN, tok, _make_sft, SEED)
    val = _sample_valid(ds["valid"], N_SFT_VAL, tok, _make_sft, SEED)
    return train, val

def load_preference_splits(tok):
    ds = load_dataset(PREF_DATA)
    train = _sample_valid(ds["train"], N_PREF_TRAIN, tok, _make_pref, SEED)
    val = _sample_valid(ds["valid1"], N_PREF_VAL, tok, _make_pref, SEED)
    return train, val

def load_eval_posts(tok):
    """Held-out posts (with their human reference summary) used to
    compare SFT, PPO and DPO at the end. Never train on these."""
    ds = load_dataset(SFT_DATA)
    return _sample_valid(ds["test"], N_EVAL_POSTS, tok, _make_sft, SEED)