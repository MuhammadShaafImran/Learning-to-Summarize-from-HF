import random

import numpy as np
import torch


def set_seed(seed=42):
    """Make the data order and training as reproducible as possible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def token_logprobs(model, input_ids, attention_mask, prompt_lens):
    """
    Compute per-token log-probabilities and the summary mask.

    Args:
        model:          causal language model.
        input_ids:      (B, L) right-padded prompt + summary token ids.
        attention_mask: (B, L) 1 for real tokens, 0 for padding.
        prompt_lens:    (B,) number of prompt tokens in each row.

    Returns:
        log_probs:    (B, L - 1)
            log_probs[b, j] = log P(input_ids[b, j + 1] | input_ids[b, :j + 1])
        summary_mask: (B, L - 1) bool
            1 exactly on summary targets (not prompt, not padding).

    This helper assumes the right padding produced by starter.batching.collate_sft.
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )

    # logits[:, j] predicts token j + 1.
    logits = outputs.logits[:, :-1, :].float()
    targets = input_ids[:, 1:]

    # Log probabilities of the actual next tokens.
    log_probs_all = torch.log_softmax(logits, dim=-1)
    log_probs = log_probs_all.gather(
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)

    # Target position j corresponds to input position j + 1.
    # The summary starts at input position prompt_lens[b], so target j is a
    # summary token exactly when j + 1 >= prompt_lens[b].
    positions = torch.arange(
        1, targets.size(1) + 1, device=input_ids.device
    ).unsqueeze(0)

    prompt_lens = torch.as_tensor(prompt_lens, device=input_ids.device)
    summary_mask = positions >= prompt_lens.unsqueeze(1)

    # Remove padding positions.
    summary_mask = summary_mask & attention_mask[:, 1:].bool()

    # Free the large (B, L-1, vocab) tensors as soon as they are not needed.
    del outputs, logits, log_probs_all

    return log_probs, summary_mask


@torch.no_grad()
def generate_summary(model, tokenizer, prompt_ids, device, max_new_tokens=48):
    """
    Greedily generate one summary from one prompt.

    Only the newly generated tokens are decoded, so the returned string is
    the summary text alone (special tokens such as EOS are skipped).
    """
    was_training = model.training
    model.eval()

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    new_tokens = generated[0, input_ids.shape[1]:].cpu()
    summary = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    if was_training:
        model.train()

    return summary
