"""
Task 2: Stage 1 supervised fine-tuning (SFT) on the TL;DR data.

Run from the repository root:

    python train_sft.py

Outputs:
    checkpoints/sft/                 final model + tokenizer
    results/training_logs/sft_training.csv
    results/samples_sft.md           before/after examples
"""

import csv
import time
from functools import partial
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from starter.config import SEED
from starter.models import load_tokenizer, load_causal_lm
from starter.data import load_sft_splits, load_eval_posts
from starter.batching import collate_sft

from common import set_seed, token_logprobs, generate_summary


# --------------------------------------------------
# 1. Configuration (assignment settings)
# --------------------------------------------------

BATCH_SIZE = 16
LEARNING_RATE = 5e-5
EPOCHS = 1
GRAD_CLIP = 1.0
LOG_INTERVAL = 250
MAX_NEW_TOKENS = 48

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT = Path(__file__).resolve().parent

CHECKPOINT_DIR = ROOT / "checkpoints" / "sft"
LOG_DIR = ROOT / "results" / "training_logs"
SAMPLE_FILE = ROOT / "results" / "samples_sft.md"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_FILE.parent.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Summary-only SFT loss
# --------------------------------------------------

def calculate_sft_loss(model, batch, device):
    """
    Mean negative log-likelihood of the summary tokens only.

    Prompt tokens, padding tokens and tokens after the summary are excluded
    by summary_mask.
    """
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    prompt_lens = batch["prompt_lens"].to(device)

    log_probs, summary_mask = token_logprobs(
        model,
        input_ids,
        attention_mask,
        prompt_lens,
    )

    selected_log_probs = log_probs[summary_mask]

    # This branch should never be taken for valid SFT data, but it prevents
    # a NaN loss if an empty summary ever slips through.
    if selected_log_probs.numel() == 0:
        return log_probs.sum() * 0.0

    return -selected_log_probs.mean()


# --------------------------------------------------
# 3. Validation
# --------------------------------------------------

@torch.no_grad()
def evaluate(model, val_loader, device):
    """Token-weighted mean summary NLL over the validation set."""
    was_training = model.training
    model.eval()

    total_nll = 0.0
    total_tokens = 0

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        prompt_lens = batch["prompt_lens"].to(device)

        log_probs, summary_mask = token_logprobs(
            model,
            input_ids,
            attention_mask,
            prompt_lens,
        )

        selected_log_probs = log_probs[summary_mask]
        total_nll += -selected_log_probs.sum().item()
        total_tokens += selected_log_probs.numel()

    if was_training:
        model.train()

    return total_nll / max(total_tokens, 1)


# --------------------------------------------------
# 4. Save five before/after examples
# --------------------------------------------------

def save_samples(model, tokenizer, examples, device, before_outputs):
    with open(SAMPLE_FILE, "w", encoding="utf-8") as file:
        file.write("# SFT Before and After Examples\n\n")

        for index, example in enumerate(examples):
            after = generate_summary(
                model,
                tokenizer,
                example["prompt_ids"],
                device,
                max_new_tokens=MAX_NEW_TOKENS,
            )

            file.write(f"## Example {index + 1}\n\n")
            file.write(f"**Prompt:**\n\n{example['prompt']}\n\n")
            file.write(f"**Human reference:**\n\n{example['summary']}\n\n")
            file.write(f"**Before SFT:**\n\n{before_outputs[index]}\n\n")
            file.write(f"**After SFT:**\n\n{after}\n\n")
            file.write("---\n\n")


# --------------------------------------------------
# 5. Main training function
# --------------------------------------------------

def print_length_stats(rows, name):
    """Report mean/max prompt and summary lengths for Table 1."""
    prompt_lengths = [len(row["prompt_ids"]) for row in rows]
    summary_lengths = [len(row["summary_ids"]) for row in rows]

    print(
        f"{name} prompt tokens  mean/max: "
        f"{sum(prompt_lengths) / len(prompt_lengths):.2f} / {max(prompt_lengths)}"
    )
    print(
        f"{name} summary tokens mean/max: "
        f"{sum(summary_lengths) / len(summary_lengths):.2f} / {max(summary_lengths)}"
        " (includes EOS)"
    )


def main():
    set_seed(SEED)

    print("Device:", DEVICE)
    if DEVICE == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # Load tokenizer and fixed SFT subsets.
    print("\nLoading tokenizer...")
    tokenizer = load_tokenizer()

    print("Loading SFT dataset...")
    train_data, val_data = load_sft_splits(tokenizer)

    print("Training examples:  ", len(train_data))
    print("Validation examples:", len(val_data))
    print_length_stats(train_data, "SFT train")
    print_length_stats(val_data, "SFT validation")

    # DataLoaders. collate_sft right-pads and returns prompt_lens, which is
    # what token_logprobs needs to build the summary mask.
    collate_fn = partial(
        collate_sft,
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

    # Load the base model.
    print("\nLoading base model...")
    model = load_causal_lm(device=DEVICE)

    # Five held-out evaluation posts, generated before and after SFT.
    eval_examples = load_eval_posts(tokenizer)[:5]

    print("\nGenerating before-SFT summaries...")
    before_outputs = [
        generate_summary(
            model,
            tokenizer,
            example["prompt_ids"],
            DEVICE,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        for example in eval_examples
    ]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    log_path = LOG_DIR / "sft_training.csv"

    running_loss = 0.0
    running_steps = 0
    epoch_loss_sum = 0.0
    epoch_steps = 0

    global_step = 0
    start_time = time.time()

    model.train()

    with open(log_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "step",
            "train_loss",
            "validation_loss",
            "elapsed_seconds",
        ])

        for _epoch in range(EPOCHS):
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)

                loss = calculate_sft_loss(model, batch, DEVICE)
                loss.backward()

                clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

                loss_value = loss.item()
                running_loss += loss_value
                running_steps += 1
                epoch_loss_sum += loss_value
                epoch_steps += 1
                global_step += 1

                if global_step % LOG_INTERVAL == 0:
                    train_loss = running_loss / running_steps
                    val_loss = evaluate(model, val_loader, DEVICE)
                    elapsed = time.time() - start_time

                    writer.writerow([
                        global_step,
                        train_loss,
                        val_loss,
                        elapsed,
                    ])
                    csvfile.flush()

                    print(
                        f"Step {global_step} | "
                        f"Train Loss: {train_loss:.4f} | "
                        f"Validation Loss: {val_loss:.4f} | "
                        f"Time: {elapsed / 60:.2f} min"
                    )

                    running_loss = 0.0
                    running_steps = 0

        # Final metrics: epoch-average training loss and final validation loss.
        final_train_loss = epoch_loss_sum / max(epoch_steps, 1)
        final_val_loss = evaluate(model, val_loader, DEVICE)
        elapsed = time.time() - start_time

        writer.writerow([
            "final",
            final_train_loss,
            final_val_loss,
            elapsed,
        ])
        csvfile.flush()

    print("\nTraining completed.")
    print(f"Final train loss:      {final_train_loss:.4f}")
    print(f"Final validation loss: {final_val_loss:.4f}")
    print(f"Total time: {elapsed / 60:.2f} minutes")

    # Save final model and tokenizer.
    print("\nSaving SFT checkpoint...")
    model.save_pretrained(CHECKPOINT_DIR)
    tokenizer.save_pretrained(CHECKPOINT_DIR)
    print("Checkpoint saved at:", CHECKPOINT_DIR)

    # Generate the same five examples after SFT.
    print("\nGenerating after-SFT summaries...")
    save_samples(
        model,
        tokenizer,
        eval_examples,
        DEVICE,
        before_outputs,
    )

    print("Samples saved at:", SAMPLE_FILE)
    print("SFT stage finished successfully.")


if __name__ == "__main__":
    main()
