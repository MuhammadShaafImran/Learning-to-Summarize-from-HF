
import random
import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def token_logprobs(model, input_ids, attention_mask, prompt_lens):
    """
    Returns:
        log_probs: (B, L-1)
        summary_mask: (B, L-1)

    Each log probability at position t represents
    log P(input_ids[t+1] | input_ids[:t+1]).

    The summary mask selects summary targets only.
    """

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
    )

    # Position t predicts token t+1
    logits = outputs.logits[:, :-1, :].float()
    targets = input_ids[:, 1:]

    # Log probabilities for the actual next tokens
    log_probs_all = torch.log_softmax(logits, dim=-1)

    log_probs = log_probs_all.gather(
        dim=-1,
        index=targets.unsqueeze(-1)
    ).squeeze(-1)

    batch_size, seq_len = targets.shape
    positions = torch.arange(
        1, seq_len + 1, device=input_ids.device
    ).unsqueeze(0)

    # Target positions at or after prompt length are summary tokens
    summary_mask = positions >= prompt_lens.unsqueeze(1)

    # Remove padding positions
    summary_mask = summary_mask & attention_mask[:, 1:].bool()

    return log_probs, summary_mask