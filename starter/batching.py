import torch

def _pad(seqs, pad_id, side):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)

    for i, s in enumerate(seqs):
        t = torch.tensor(s, dtype=torch.long)

        if side == "right":
            ids[i, :len(s)], mask[i, :len(s)] = t, 1
        else:  # left padding
            ids[i, L - len(s):], mask[i, L - len(s):] = t, 1

    return ids, mask


def collate_sft(batch, pad_id):
    """Right-padded prompt+summary. prompt_lens[i] = number of prompt
    tokens in row i, so summary tokens are positions prompt_lens[i] .. end."""
    seqs = [b["prompt_ids"] + b["summary_ids"] for b in batch]
    ids, mask = _pad(seqs, pad_id, "right")
    prompt_lens = torch.tensor([len(b["prompt_ids"]) for b in batch])

    return {
        "input_ids": ids,
        "attention_mask": mask,
        "prompt_lens": prompt_lens
    }


def collate_pref(batch, pad_id):
    """Two right-padded batches (prompt+chosen, prompt+rejected) that
    share the same prompt_lens."""
    ch, ch_mask = _pad(
        [b["prompt_ids"] + b["chosen_ids"] for b in batch],
        pad_id,
        "right"
    )

    rj, rj_mask = _pad(
        [b["prompt_ids"] + b["rejected_ids"] for b in batch],
        pad_id,
        "right"
    )

    prompt_lens = torch.tensor([len(b["prompt_ids"]) for b in batch])

    return {
        "chosen_ids": ch,
        "chosen_mask": ch_mask,
        "rejected_ids": rj,
        "rejected_mask": rj_mask,
        "prompt_lens": prompt_lens
    }


def collate_prompts(batch, pad_id):
    """LEFT-padded prompts only, for model.generate() in PPO rollouts
    and evaluation. Left padding keeps every prompt's last token at
    the same position, which is where generation starts."""
    ids, mask = _pad(
        [b["prompt_ids"] for b in batch],
        pad_id,
        "left"
    )

    return {
        "input_ids": ids,
        "attention_mask": mask
    }