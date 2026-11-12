from functools import partial
from torch.utils.data import DataLoader
from models import load_tokenizer
from data import load_sft_splits, load_preference_splits, load_eval_posts
from batching import collate_sft, collate_pref, collate_prompts

tok = load_tokenizer()

sft_train, sft_val = load_sft_splits(tok)
pref_train, pref_val = load_preference_splits(tok)
eval_posts = load_eval_posts(tok)

print("SFT train/val:", len(sft_train), len(sft_val))
print("Pref train/val:", len(pref_train), len(pref_val))
print("Eval posts:", len(eval_posts))

ex = sft_train[0]
print("\n--- one SFT example ---")
print(ex["prompt"][-300:])
print("SUMMARY:", repr(ex["summary"]))

pad = tok.pad_token_id

b = next(iter(DataLoader(
    sft_train,
    8,
    shuffle=True,
    collate_fn=partial(collate_sft, pad_id=pad)
)))

print("\nSFT batch:", tuple(b["input_ids"].shape), "prompt_lens", b["prompt_lens"].tolist())

b = next(iter(DataLoader(
    pref_train,
    8,
    shuffle=True,
    collate_fn=partial(collate_pref, pad_id=pad)
)))

print("Pref batch:", tuple(b["chosen_ids"].shape), tuple(b["rejected_ids"].shape))

b = next(iter(DataLoader(
    eval_posts,
    8,
    collate_fn=partial(collate_prompts, pad_id=pad)
)))

print("Prompt batch:", tuple(b["input_ids"].shape), "(left-padded)")