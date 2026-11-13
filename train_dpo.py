"""
Task 5: Stage 4 — Direct Preference Optimisation (DPO).

Run from the repository root after SFT has finished:

    python train_dpo.py

Required input:
    checkpoints/sft/                 SFT policy + tokenizer

Outputs:
    checkpoints/dpo/                 final DPO policy + tokenizer
    results/training_logs/dpo_training.csv

DPO settings from the assignment:
    beta = 0.1
    AdamW, lr = 2e-6
    batch size = 8 pairs
    1 epoch over the 20,000 preference pairs
    log every 250 steps on the 1,000 validation pairs
"""

import csv
import math
import time
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from starter.config import SEED
from starter.models import load_tokenizer, load_causal_lm
from starter.data import load_preference_splits
from starter.batching import collate_pref

from common import set_seed, token_logprobs


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

BETA = 0.1
BATCH_SIZE = 8
LEARNING_RATE = 2e-6
EPOCHS = 1
LOG_INTERVAL = 250

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT = Path(__file__).resolve().parent

SFT_CHECKPOINT = ROOT / "checkpoints" / "sft"
DPO_CHECKPOINT = ROOT / "checkpoints" / "dpo"
LOG_DIR = ROOT / "results" / "training_logs"

DPO_CHECKPOINT.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Sequence log-probability over summary tokens only
# --------------------------------------------------

def sequence_logprob(model, input_ids, attention_mask, prompt_lens):
    """
    Return log pi(y | x) for every row in the batch.

    Only summary tokens contribute. Prompt tokens and padding are excluded by
    the summary mask returned by token_logprobs.
    """
    log_probs, summary_mask = token_logprobs(
        model,
        input_ids,
        attention_mask,
        prompt_lens,
    )

    # Shape: (B,)
    return log_probs.masked_fill(~summary_mask, 0.0).sum(dim=1)


# --------------------------------------------------
# 3. DPO margin and loss
# --------------------------------------------------

def dpo_margin(policy, reference, batch, device):
    """
    The expression inside sigma(.):
        beta * [(log pi(y_w|x) - log pi_ref(y_w|x))
                - (log pi(y_l|x) - log pi_ref(y_l|x))]
    """
    chosen_ids = batch["chosen_ids"].to(device)
    chosen_mask = batch["chosen_mask"].to(device)
    rejected_ids = batch["rejected_ids"].to(device)
    rejected_mask = batch["rejected_mask"].to(device)
    prompt_lens = batch["prompt_lens"].to(device)

    # Trainable policy: build the graph for backpropagation.
    policy_chosen = sequence_logprob(
        policy, chosen_ids, chosen_mask, prompt_lens
    )
    policy_rejected = sequence_logprob(
        policy, rejected_ids, rejected_mask, prompt_lens
    )

    # Frozen reference: no gradients and no graph.
    with torch.no_grad():
        reference_chosen = sequence_logprob(
            reference, chosen_ids, chosen_mask, prompt_lens
        )
        reference_rejected = sequence_logprob(
            reference, rejected_ids, rejected_mask, prompt_lens
        )

    chosen_advantage = policy_chosen - reference_chosen
    rejected_advantage = policy_rejected - reference_rejected

    return BETA * (chosen_advantage - rejected_advantage)


def calculate_dpo_loss(policy, reference, batch, device):
    margin = dpo_margin(policy, reference, batch, device)
    return -F.logsigmoid(margin).mean()


# --------------------------------------------------
# 4. Validation: loss and implicit reward accuracy
# --------------------------------------------------

@torch.no_grad()
def evaluate_dpo(policy, reference, val_loader, device):
    """
    Returns:
        validation loss
        implicit reward accuracy = fraction of pairs where the DPO margin > 0
    """
    was_training = policy.training
    policy.eval()
    reference.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch in val_loader:
        margin = dpo_margin(policy, reference, batch, device)

        total_loss += -F.logsigmoid(margin).sum().item()
        correct += (margin > 0).sum().item()
        total += margin.numel()

    if was_training:
        policy.train()

    return total_loss / max(total, 1), correct / max(total, 1)


# --------------------------------------------------
# 5. Data statistics
# --------------------------------------------------

def print_pref_stats(rows, name):
    prompt_lengths = [len(row["prompt_ids"]) for row in rows]
    chosen_lengths = [len(row["chosen_ids"]) for row in rows]
    rejected_lengths = [len(row["rejected_ids"]) for row in rows]

    print(f"{name}: {len(rows)} pairs")
    print(
        f"  prompt tokens   mean/max: "
        f"{sum(prompt_lengths) / len(prompt_lengths):.2f} / {max(prompt_lengths)}"
    )
    print(
        f"  chosen tokens   mean/max: "
        f"{sum(chosen_lengths) / len(chosen_lengths):.2f} / {max(chosen_lengths)}"
        " (includes EOS)"
    )
    print(
        f"  rejected tokens mean/max: "
        f"{sum(rejected_lengths) / len(rejected_lengths):.2f} / {max(rejected_lengths)}"
        " (includes EOS)"
    )


# --------------------------------------------------
# 6. Main training function
# --------------------------------------------------

def main():
    set_seed(SEED)

    print("Device:", DEVICE)
    if DEVICE == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    if not (SFT_CHECKPOINT / "config.json").exists():
        raise FileNotFoundError(
            f"Missing SFT checkpoint: {SFT_CHECKPOINT}\n"
            "Run train_sft.py first, or copy checkpoints/sft/ here."
        )

    # Tokenizer and data.
    print("\nLoading tokenizer...")
    tokenizer = load_tokenizer(SFT_CHECKPOINT)

    print("Loading preference data...")
    train_data, val_data = load_preference_splits(tokenizer)

    print("Preference train / validation pairs:", len(train_data), "/", len(val_data))
    print_pref_stats(train_data, "Preference train")
    print_pref_stats(val_data, "Preference validation")

    collate_fn = partial(
        collate_pref,
        pad_id=tokenizer.pad_token_id,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=(DEVICE == "cuda"),
    )

    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=(DEVICE == "cuda"),
    )

    print("Train batches per epoch:", len(train_loader))
    print("Validation batches:     ", len(val_loader))
    print("Optimizer steps:        ", len(train_loader) * EPOCHS)

    # Policy and frozen reference, both from the SFT checkpoint.
    print("\nLoading SFT policy and frozen reference...")
    policy = load_causal_lm(name_or_path=SFT_CHECKPOINT, device=DEVICE)
    reference = load_causal_lm(name_or_path=SFT_CHECKPOINT, device=DEVICE)

    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad = False

    assert all(
        not parameter.requires_grad
        for parameter in reference.parameters()
    ), "Reference model must be frozen."

    print(
        "Reference parameters requiring grad:",
        sum(p.requires_grad for p in reference.parameters()),
    )

    # Correctness check: at initialisation pi_theta == pi_ref, so the DPO
    # argument is zero and the loss must be log(2) ~= 0.6931.
    first_batch = next(iter(train_loader))

    policy.eval()
    with torch.no_grad():
        initial_loss = calculate_dpo_loss(
            policy, reference, first_batch, DEVICE
        )

    expected_initial_loss = math.log(2.0)
    print(
        f"Initial DPO loss: {initial_loss.item():.6f} "
        f"(expected {expected_initial_loss:.6f})"
    )
    assert abs(initial_loss.item() - expected_initial_loss) < 1e-3, (
        "DPO initial-loss check failed: policy and reference must be identical "
        "at the start."
    )

    policy.train()

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=LEARNING_RATE,
    )

    log_path = LOG_DIR / "dpo_training.csv"

    running_loss = 0.0
    running_steps = 0
    epoch_loss_sum = 0.0
    epoch_steps = 0

    global_step = 0
    start_time = time.time()

    policy.train()

    with open(log_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "step",
            "train_loss",
            "validation_loss",
            "validation_accuracy",
            "elapsed_seconds",
        ])

        for _epoch in range(EPOCHS):
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)

                loss = calculate_dpo_loss(
                    policy, reference, batch, DEVICE
                )
                loss.backward()
                optimizer.step()

                loss_value = loss.item()
                running_loss += loss_value
                running_steps += 1
                epoch_loss_sum += loss_value
                epoch_steps += 1
                global_step += 1

                if global_step % LOG_INTERVAL == 0:
                    train_loss = running_loss / running_steps
                    val_loss, val_accuracy = evaluate_dpo(
                        policy, reference, val_loader, DEVICE
                    )
                    elapsed = time.time() - start_time

                    writer.writerow([
                        global_step,
                        train_loss,
                        val_loss,
                        val_accuracy,
                        elapsed,
                    ])
                    csvfile.flush()

                    print(
                        f"Step {global_step} | "
                        f"Train Loss: {train_loss:.4f} | "
                        f"Val Loss: {val_loss:.4f} | "
                        f"Val Accuracy: {val_accuracy:.4f} | "
                        f"Time: {elapsed / 60:.2f} min"
                    )

                    running_loss = 0.0
                    running_steps = 0

        final_train_loss = epoch_loss_sum / max(epoch_steps, 1)
        final_val_loss, final_val_accuracy = evaluate_dpo(
            policy, reference, val_loader, DEVICE
        )
        elapsed = time.time() - start_time

        writer.writerow([
            "final",
            final_train_loss,
            final_val_loss,
            final_val_accuracy,
            elapsed,
        ])
        csvfile.flush()

    print("\nTraining completed.")
    print(f"Final train loss:        {final_train_loss:.4f}")
    print(f"Final validation loss:   {final_val_loss:.4f}")
    print(f"Final implicit accuracy: {final_val_accuracy:.4f}")
    print(f"Total time: {elapsed / 60:.2f} minutes")

    # Save the DPO policy only (the reference is not needed later).
    print("\nSaving DPO checkpoint...")
    policy.save_pretrained(DPO_CHECKPOINT)
    tokenizer.save_pretrained(DPO_CHECKPOINT)
    print("Checkpoint saved at:", DPO_CHECKPOINT)
    print("DPO stage finished successfully.")


if __name__ == "__main__":
    main()
