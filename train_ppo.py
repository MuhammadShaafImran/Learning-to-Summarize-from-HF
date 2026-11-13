"""
Task 4: Stage 3 - PPO.

Run from the repository root after SFT and the reward model are done:

    python train_ppo.py

Inputs:
    checkpoints/sft/                 policy + reference, and the tokenizer
    checkpoints/reward/              frozen reward model

Outputs:
    checkpoints/ppo/                 PPO policy (no value head) + tokenizer
    results/training_logs/ppo_training.csv
    results/samples_ppo.md           a few summaries from the trained policy

Models in memory (assignment section 2.4):
    policy  pi_theta   checkpoints/sft/ + a new linear value head   trained
    reference pi_ref   checkpoints/sft/                             frozen
    reward  r_psi      checkpoints/reward/                          frozen

One PPO iteration = 16 prompts -> 1 sampled summary each -> K = 4 update epochs.

Settings (assignment section 2.4, 4.5) - do not change:
    clip epsilon 0.2      KL coefficient 0.05
    value coeff  0.1      entropy coeff  0.01
    discount     1.0      PPO epochs K   4
    AdamW lr 1e-5         gradient-norm clipping 1.0
    16 prompts per rollout, 200 iterations
"""

import csv
import os
import random
import time
from pathlib import Path

# Helps the T4 allocator reuse fragmented blocks.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from starter.config import SEED
from starter.models import load_tokenizer, load_causal_lm
from starter.data import load_sft_splits, load_eval_posts
from starter.batching import collate_prompts

from common import set_seed, generate_summary
from train_reward import load_reward_model


# --------------------------------------------------
# 1. Configuration (assignment settings)
# --------------------------------------------------

PROMPTS_PER_ROLLOUT = 16
ITERATIONS = 200

PPO_EPOCHS = 4
CLIP_EPS = 0.2
KL_COEF = 0.05
VALUE_COEF = 0.1
ENTROPY_COEF = 0.01
DISCOUNT = 1.0

LEARNING_RATE = 1e-5
GRAD_CLIP = 1.0
MAX_NEW_TOKENS = 48

# The assignment allows splitting a rollout in two halves when the T4 runs
# out of memory: gradients are accumulated, one optimizer.step() per epoch.
MICRO_BATCH_SIZE = 8
SAVE_EVERY = 50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT = Path(__file__).resolve().parent

SFT_CHECKPOINT = ROOT / "checkpoints" / "sft"
REWARD_CHECKPOINT = ROOT / "checkpoints" / "reward"
PPO_CHECKPOINT = ROOT / "checkpoints" / "ppo"
LOG_DIR = ROOT / "results" / "training_logs"
SAMPLE_FILE = ROOT / "results" / "samples_ppo.md"

PPO_CHECKPOINT.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Policy / reference forward: per-token log-probs and values
# --------------------------------------------------

def token_logprobs_and_values(
    model,
    value_head,
    input_ids,
    attention_mask,
    prompt_lens,
    gen_lens,
):
    """
    Per-token statistics for the generated tokens of every row.

    Args:
        model:          causal LM (policy or reference).
        value_head:     linear value head, or None for the reference.
        input_ids:      (B, L) RIGHT-padded prompt + generated summary.
        attention_mask: (B, L) 1 for real tokens, 0 for padding.
        prompt_lens:    (B,) number of prompt tokens in each row.
        gen_lens:       (B,) number of generated tokens in each row.

    Returns:
        log_probs: (B, Tg) log pi(a_t | s_t), zero on padding columns.
        values:    (B, Tg) V(s_t) read from the position that predicts a_t,
                   or None when value_head is None.
        entropy:   (B, Tg) token entropy, or None when value_head is None.
        gen_mask:  (B, Tg) 1 on generated tokens, 0 on padding columns.

    Right padding + prompt_lens is used instead of left padding on purpose:
    the hidden state that predicts token t+1 is at index t, and prompt_lens
    tells us exactly where each summary starts.
    """
    batch_size, seq_len = input_ids.shape
    max_gen = int(gen_lens.max().item())

    hidden_states = model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    ).last_hidden_state

    # The k-th generated token is predicted from target index prompt_len + k.
    columns = torch.arange(max_gen, device=input_ids.device).unsqueeze(0)
    target_index = (prompt_lens.unsqueeze(1) + columns).clamp(max=seq_len - 2)
    gen_mask = columns < gen_lens.unsqueeze(1)

    positions = target_index.unsqueeze(-1).expand(
        -1, -1, hidden_states.size(-1)
    )
    selected_hidden = hidden_states.gather(1, positions)   # (B, Tg, H)

    # The LM head is applied only at the generated positions, so the
    # (B, L, vocab) logits tensor is never materialised.
    logits = model.lm_head(selected_hidden)                # (B, Tg, vocab)
    log_probs_all = torch.log_softmax(logits.float(), dim=-1)

    targets = input_ids.gather(1, target_index + 1)        # (B, Tg)
    log_probs = log_probs_all.gather(
        -1, targets.unsqueeze(-1)
    ).squeeze(-1)                                          # (B, Tg)

    log_probs = log_probs.masked_fill(~gen_mask, 0.0)

    if value_head is None:
        values = None
        entropy = None
    else:
        values = value_head(selected_hidden).squeeze(-1)   # (B, Tg)
        entropy = -(log_probs_all.exp() * log_probs_all).sum(dim=-1)

    del logits, log_probs_all, selected_hidden, hidden_states

    return log_probs, values, entropy, gen_mask


# --------------------------------------------------
# 3. Rollout: sample one summary per prompt
# --------------------------------------------------

def collect_rollout(policy, tokenizer, examples):
    """
    Sample one summary per prompt and cut it at its first end-of-sequence.

    Prompts are LEFT-padded (collate_prompts), so every prompt ends at the
    same column and the generated region is the same slice for every row.
    """
    prompts = [example["prompt_ids"] for example in examples]

    batch = collate_prompts(examples, pad_id=tokenizer.pad_token_id)
    input_ids = batch["input_ids"].to(DEVICE)
    attention_mask = batch["attention_mask"].to(DEVICE)
    prompt_width = input_ids.size(1)

    policy.eval()
    policy.config.use_cache = True

    with torch.no_grad():
        generated = policy.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    policy.config.use_cache = False

    generated = generated[:, prompt_width:]

    gen_ids = []
    for row in generated:
        token_ids = row.tolist()

        # A summary ends at its first EOS; ignore everything after it.
        if tokenizer.eos_token_id in token_ids:
            token_ids = token_ids[
                : token_ids.index(tokenizer.eos_token_id) + 1
            ]

        gen_ids.append(token_ids)

    return prompts, gen_ids


def build_scoring_batch(prompts, gen_ids, pad_id):
    """
    Right-pad prompt + generated tokens for the scoring forward passes.

    Returns input_ids, attention_mask, prompt_lens, gen_lens, plus the
    decoded summaries for logging.
    """
    sequences = [p + g for p, g in zip(prompts, gen_ids)]
    max_len = max(len(sequence) for sequence in sequences)

    input_ids = torch.full(
        (len(sequences), max_len), pad_id, dtype=torch.long
    )
    attention_mask = torch.zeros(
        (len(sequences), max_len), dtype=torch.long
    )

    for index, sequence in enumerate(sequences):
        length = len(sequence)
        input_ids[index, :length] = torch.tensor(sequence, dtype=torch.long)
        attention_mask[index, :length] = 1

    prompt_lens = torch.tensor([len(p) for p in prompts], dtype=torch.long)
    gen_lens = torch.tensor([len(g) for g in gen_ids], dtype=torch.long)

    return {
        "input_ids": input_ids.to(DEVICE),
        "attention_mask": attention_mask.to(DEVICE),
        "prompt_lens": prompt_lens.to(DEVICE),
        "gen_lens": gen_lens.to(DEVICE),
    }


# --------------------------------------------------
# 4. Per-token reward, returns and advantages
# --------------------------------------------------

def compute_rewards(log_probs_old, log_probs_ref, reward_scores, gen_lens):
    """
    Eq. 41 of the lecture notes:

        r_t = -beta_KL ( log pi_old(a_t|s_t) - log pi_ref(a_t|s_t) )
              + 1[t = T] r_psi(x, y)

    The reward-model score is added at the last generated token T.
    """
    rewards = -KL_COEF * (log_probs_old - log_probs_ref)

    last_index = (gen_lens - 1).clamp(min=0).unsqueeze(1)
    rewards = rewards.scatter_add(1, last_index, reward_scores.unsqueeze(1))

    return rewards


def compute_returns(rewards, gen_mask, discount=DISCOUNT):
    """
    Eq. 19: R_t = sum_{k >= 0} gamma^k r_{t+k}, computed backwards.

    With gamma = 1 this is the suffix sum of the per-token rewards.
    """
    returns = torch.zeros_like(rewards)
    running = torch.zeros(
        rewards.size(0), device=rewards.device, dtype=rewards.dtype
    )

    for step in range(rewards.size(1) - 1, -1, -1):
        keep = gen_mask[:, step].to(rewards.dtype)
        running = running + discount * rewards[:, step] * keep
        returns[:, step] = running

    return returns


def compute_advantages(returns, values, gen_mask):
    """
    Eq. 20: A_t = R_t - V_old(s_t), then normalised over all generated
    tokens of the batch.
    """
    advantages = returns - values

    valid = advantages[gen_mask]
    mean = valid.mean()
    std = valid.std(unbiased=False)

    advantages = (advantages - mean) / (std + 1e-8)
    advantages = advantages.masked_fill(~gen_mask, 0.0)

    return advantages


# --------------------------------------------------
# 5. PPO update: K epochs on the same rollout
# --------------------------------------------------

def ppo_update(policy, value_head, optimizer, rollout):
    """
    Minimise, over all generated tokens of the rollout:

        L = -L_PPO + c_V * E[(V(s_t) - R_t)^2] - c_H * E[H(pi(.|s_t))]

    with L_PPO = E[min(ratio * A_t, clip(ratio, 1-eps, 1+eps) * A_t)].

    The rollout is processed in micro-batches; gradients are accumulated and
    a single optimizer.step() is taken per epoch, so the update is identical
    to the full-batch one.
    """
    input_ids = rollout["input_ids"]
    attention_mask = rollout["attention_mask"]
    prompt_lens = rollout["prompt_lens"]
    gen_lens = rollout["gen_lens"]

    old_log_probs = rollout["old_log_probs"]
    advantages = rollout["advantages"]
    returns = rollout["returns"]

    gen_mask = rollout["gen_mask"]
    total_valid = gen_mask.sum().item()

    parameters = list(policy.parameters()) + list(value_head.parameters())

    stats = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "clip_fraction": 0.0,
        "ratio": 0.0,
        "epoch1_clip_fraction": 0.0,
    }

    for epoch in range(PPO_EPOCHS):
        policy.train()
        optimizer.zero_grad(set_to_none=True)

        epoch_clip_hits = 0
        epoch_ratio_sum = 0.0
        epoch_valid = 0
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_entropy = 0.0

        for start in range(0, input_ids.size(0), MICRO_BATCH_SIZE):
            stop = start + MICRO_BATCH_SIZE
            rows = slice(start, stop)

            log_probs, values, entropy, mask = token_logprobs_and_values(
                policy,
                value_head,
                input_ids[rows],
                attention_mask[rows],
                prompt_lens[rows],
                gen_lens[rows],
            )

            log_ratio = log_probs - old_log_probs[rows]
            ratio = torch.exp(log_ratio)

            batch_advantages = advantages[rows]

            unclipped = ratio * batch_advantages
            clipped = torch.clamp(
                ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS
            ) * batch_advantages

            policy_loss = -torch.min(unclipped, clipped)[mask].mean()
            value_loss = ((values - returns[rows]) ** 2)[mask].mean()
            entropy_mean = entropy[mask].mean()

            loss = (
                policy_loss
                + VALUE_COEF * value_loss
                - ENTROPY_COEF * entropy_mean
            )

            # Weight by the token count so the accumulated gradient equals
            # the mean over the whole rollout.
            chunk_valid = mask.sum().item()
            weight = chunk_valid / max(total_valid, 1)
            (loss * weight).backward()

            epoch_policy_loss += policy_loss.item() * weight
            epoch_value_loss += value_loss.item() * weight
            epoch_entropy += entropy_mean.item() * weight

            outside = (ratio < 1.0 - CLIP_EPS) | (ratio > 1.0 + CLIP_EPS)
            epoch_clip_hits += (outside & mask).sum().item()
            epoch_ratio_sum += (ratio * mask).sum().item()
            epoch_valid += chunk_valid

        clip_fraction = epoch_clip_hits / max(epoch_valid, 1)

        # Report the last of the K epochs: that is the update actually taken.
        stats["policy_loss"] = epoch_policy_loss
        stats["value_loss"] = epoch_value_loss
        stats["entropy"] = epoch_entropy
        stats["clip_fraction"] = clip_fraction
        stats["ratio"] = epoch_ratio_sum / max(epoch_valid, 1)
        if epoch == 0:
            stats["epoch1_clip_fraction"] = clip_fraction

        clip_grad_norm_(parameters, GRAD_CLIP)
        optimizer.step()

    return stats


# --------------------------------------------------
# 6. Helpers
# --------------------------------------------------

def save_policy(policy, tokenizer, path):
    """
    Save the policy only - the value head is not needed after training.

    Weights are written in bfloat16 so this checkpoint matches the SFT and
    DPO checkpoints and stays small; the in-memory model is left untouched.
    """
    path.mkdir(parents=True, exist_ok=True)

    policy.config.use_cache = True
    state_dict = {
        key: value.to(torch.bfloat16)
        for key, value in policy.state_dict().items()
    }
    policy.save_pretrained(path, state_dict=state_dict)
    tokenizer.save_pretrained(path)
    policy.config.use_cache = False


def format_report(stats, mean_reward, mean_kl, mean_length):
    return (
        f"Reward: {mean_reward:+.4f} | "
        f"KL: {mean_kl:+.4f} | "
        f"Len: {mean_length:.2f} | "
        f"Clip: {stats['clip_fraction']:.4f} | "
        f"Policy: {stats['policy_loss']:+.4f} | "
        f"Value: {stats['value_loss']:.4f} | "
        f"Entropy: {stats['entropy']:.4f}"
    )


# --------------------------------------------------
# 7. Main training function
# --------------------------------------------------

def main():
    set_seed(SEED)

    print("Device:", DEVICE)
    if DEVICE == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    for name, path in (("SFT", SFT_CHECKPOINT), ("reward", REWARD_CHECKPOINT)):
        if not (path / "config.json").exists():
            raise FileNotFoundError(
                f"Missing {name} checkpoint: {path}\n"
                "Run the earlier stages first, or copy the checkpoint here."
            )

    print("\nLoading tokenizer...")
    tokenizer = load_tokenizer(SFT_CHECKPOINT)

    print("Loading SFT prompts...")
    sft_train, sft_val = load_sft_splits(tokenizer)
    print("SFT train / validation pairs:", len(sft_train), "/", len(sft_val))

    # Fixed prompt order: each of the 3,200 rollout prompts is used once
    # (the list is cycled only if the SFT subset is smaller than that).
    required_prompts = ITERATIONS * PROMPTS_PER_ROLLOUT
    order = list(range(len(sft_train)))
    random.Random(SEED).shuffle(order)
    prompt_order = [order[i % len(order)] for i in range(required_prompts)]

    print("Rollout prompts:", len(prompt_order), "over", ITERATIONS, "iterations")

    print("\nLoading policy, reference and reward model...")
    policy = load_causal_lm(name_or_path=SFT_CHECKPOINT, device=DEVICE).float()
    reference = load_causal_lm(
        name_or_path=SFT_CHECKPOINT, device=DEVICE
    ).float()
    reward_model = load_reward_model(REWARD_CHECKPOINT, DEVICE)

    policy.config.use_cache = False
    reference.config.use_cache = False

    # Frozen reference.
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad = False

    # Frozen reward model.
    reward_model.eval()
    for parameter in reward_model.parameters():
        parameter.requires_grad = False

    print(
        "Reference parameters requiring grad:",
        sum(p.requires_grad for p in reference.parameters()),
    )
    print(
        "Reward parameters requiring grad:   ",
        sum(p.requires_grad for p in reward_model.parameters()),
    )

    assert not any(p.requires_grad for p in reference.parameters())
    assert not any(p.requires_grad for p in reward_model.parameters())

    # New value head on the last hidden layer (assignment section 2.4).
    hidden_size = policy.config.hidden_size
    value_head = nn.Linear(hidden_size, 1).to(DEVICE)
    nn.init.normal_(value_head.weight, std=0.01)
    nn.init.zeros_(value_head.bias)
    print("Value head:", hidden_size, "-> 1")

    # Recompute layer activations during backward instead of storing them.
    policy.model.gradient_checkpointing_enable()
    print("Gradient checkpointing: enabled")

    optimizer = torch.optim.AdamW(
        list(policy.parameters()) + list(value_head.parameters()),
        lr=LEARNING_RATE,
    )

    log_path = LOG_DIR / "ppo_training.csv"

    start_time = time.time()
    first_iteration = True
    final_mean_reward = float("nan")

    with open(log_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "iteration",
            "mean_reward",
            "mean_kl",
            "mean_length",
            "clip_fraction",
            "policy_loss",
            "value_loss",
            "entropy",
            "mean_ratio",
            "epoch1_clip_fraction",
            "elapsed_seconds",
        ])

        for iteration in range(1, ITERATIONS + 1):
            start = (iteration - 1) * PROMPTS_PER_ROLLOUT
            examples = [
                sft_train[index]
                for index in prompt_order[start:start + PROMPTS_PER_ROLLOUT]
            ]

            # 4.1 Rollout.
            prompts, gen_ids = collect_rollout(policy, tokenizer, examples)
            rollout = build_scoring_batch(
                prompts, gen_ids, tokenizer.pad_token_id
            )

            # 4.2 Store log pi_old, log pi_ref, V_old and r_psi without grads.
            with torch.no_grad():
                old_log_probs, old_values, _, gen_mask = (
                    token_logprobs_and_values(
                        policy,
                        value_head,
                        rollout["input_ids"],
                        rollout["attention_mask"],
                        rollout["prompt_lens"],
                        rollout["gen_lens"],
                    )
                )
                log_probs_ref, _, _, _ = token_logprobs_and_values(
                    reference,
                    None,
                    rollout["input_ids"],
                    rollout["attention_mask"],
                    rollout["prompt_lens"],
                    rollout["gen_lens"],
                )
                reward_scores = reward_model(
                    rollout["input_ids"], rollout["attention_mask"]
                )

            # 4.3 Per-token reward.
            rewards = compute_rewards(
                old_log_probs,
                log_probs_ref,
                reward_scores,
                rollout["gen_lens"],
            )

            # 4.4 Returns and advantages.
            returns = compute_returns(rewards, gen_mask)
            advantages = compute_advantages(returns, old_values, gen_mask)

            rollout.update({
                "old_log_probs": old_log_probs,
                "advantages": advantages,
                "returns": returns,
                "gen_mask": gen_mask,
            })

            # KL to the reference, summed over the summary tokens per row.
            mean_kl = (
                ((old_log_probs - log_probs_ref) * gen_mask)
                .sum(dim=1)
                .mean()
                .item()
            )
            mean_reward = reward_scores.mean().item()
            mean_length = rollout["gen_lens"].float().mean().item()

            if first_iteration:
                print(
                    f"\nCorrectness check (iteration 1): "
                    f"KL to the reference = {mean_kl:+.6f} "
                    "(policy still equals SFT)"
                )
                assert abs(mean_kl) < 1e-3, (
                    "KL to the reference must be 0 on the first iteration: "
                    "the policy must start from the SFT checkpoint."
                )

            # 4.5 Update for K epochs.
            stats = ppo_update(policy, value_head, optimizer, rollout)
            final_mean_reward = mean_reward

            if first_iteration:
                print(
                    f"Correctness check (iteration 1): "
                    f"epoch-1 clip fraction = "
                    f"{stats['epoch1_clip_fraction']:.6f} "
                    f"(ratio is 1 in the first epoch)"
                )
                assert stats["epoch1_clip_fraction"] < 1e-6
                first_iteration = False

            elapsed = time.time() - start_time

            writer.writerow([
                iteration,
                mean_reward,
                mean_kl,
                mean_length,
                stats["clip_fraction"],
                stats["policy_loss"],
                stats["value_loss"],
                stats["entropy"],
                stats["ratio"],
                stats["epoch1_clip_fraction"],
                elapsed,
            ])
            csvfile.flush()

            print(
                f"Iter {iteration:3d}/{ITERATIONS} | "
                f"{format_report(stats, mean_reward, mean_kl, mean_length)} | "
                f"{elapsed / 60:.2f} min"
            )

            if iteration % SAVE_EVERY == 0:
                save_policy(policy, tokenizer, PPO_CHECKPOINT)
                print(f"  intermediate checkpoint saved ({iteration} iterations)")

    total_minutes = (time.time() - start_time) / 60

    print("\nTraining completed.")
    print(f"Final mean reward-model score: {final_mean_reward:+.4f}")
    print(f"Total time: {total_minutes:.2f} minutes")

    # 4.7 Save the policy without the value head.
    print("\nSaving PPO checkpoint...")
    policy.model.gradient_checkpointing_disable()
    save_policy(policy, tokenizer, PPO_CHECKPOINT)
    print("Checkpoint saved at:", PPO_CHECKPOINT)

    # A few before/after style samples for the report.
    print("\nGenerating sample summaries...")
    eval_posts = load_eval_posts(tokenizer)[:5]

    # Reward-model scores for these summaries are computed properly, on the
    # 300 evaluation posts, by evaluate.py (Task 6).
    with open(SAMPLE_FILE, "w", encoding="utf-8") as file:
        file.write("# PPO sample summaries\n\n")

        for index, post in enumerate(eval_posts):
            summary = generate_summary(
                policy,
                tokenizer,
                post["prompt_ids"],
                DEVICE,
                max_new_tokens=MAX_NEW_TOKENS,
            )

            file.write(f"## Post {index + 1}\n\n")
            file.write(f"**Human reference:** {post['summary']}\n\n")
            file.write(f"**PPO summary:** {summary}\n\n")
            file.write("---\n\n")

    print("Samples saved at:", SAMPLE_FILE)
    print("PPO stage finished successfully.")


if __name__ == "__main__":
    main()
