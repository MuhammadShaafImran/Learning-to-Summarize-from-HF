
import os
import csv
import time
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from starter.config import SEED
from starter.models import load_tokenizer, load_causal_lm
from starter.data import load_sft_splits, load_eval_posts
from starter.batching import collate_sft

from common import set_seed, token_logprobs


# --------------------------------------------------
# 1. Configuration
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

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    prompt_lens = batch["prompt_lens"].to(device)

    log_probs, summary_mask = token_logprobs(
        model,
        input_ids,
        attention_mask,
        prompt_lens
    )

    # Negative log likelihood of summary tokens only
    selected_log_probs = log_probs * summary_mask.float()

    token_count = summary_mask.sum().clamp(min=1)

    loss = -selected_log_probs.sum() / token_count

    return loss


# --------------------------------------------------
# 3. Validation
# --------------------------------------------------

@torch.no_grad()
def evaluate(model, val_loader, device):

    model.eval()

    total_loss = 0.0
    total_tokens = 0

    for batch in val_loader:

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        prompt_lens = batch["prompt_lens"].to(device)

        log_probs, summary_mask = token_logprobs(
            model,
            input_ids,
            attention_mask,
            prompt_lens
        )

        total_loss += -(
            log_probs * summary_mask.float()
        ).sum().item()

        total_tokens += summary_mask.sum().item()

    model.train()

    return total_loss / max(total_tokens, 1)


# --------------------------------------------------
# 4. Generate a summary
# --------------------------------------------------

@torch.no_grad()
def generate_summary(model, tokenizer, example, device):

    model.eval()

    input_ids = torch.tensor(
        [example["prompt_ids"]],
        dtype=torch.long,
        device=device
    )

    attention_mask = torch.ones_like(input_ids)

    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    # Remove the original prompt from generated output
    new_tokens = generated[0, input_ids.shape[1]:]

    summary = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    )

    model.train()

    return summary.strip()


# --------------------------------------------------
# 5. Save five before/after examples
# --------------------------------------------------

def save_samples(model, tokenizer, examples, device, before_outputs):

    model.eval()

    with open(SAMPLE_FILE, "w", encoding="utf-8") as file:

        file.write("# SFT Before and After Examples\n\n")

        for index, example in enumerate(examples[:5]):

            after = generate_summary(
                model, tokenizer, example, device
            )

            file.write(f"## Example {index + 1}\n\n")
            file.write(f"**Prompt:**\n\n{example['prompt']}\n\n")
            file.write(f"**Human reference:**\n\n{example['summary']}\n\n")
            file.write(f"**Before SFT:**\n\n{before_outputs[index]}\n\n")
            file.write(f"**After SFT:**\n\n{after}\n\n")
            file.write("---\n\n")

    model.train()


# --------------------------------------------------
# 6. Main training function
# --------------------------------------------------

def main():

    set_seed(SEED)

    print("Device:", DEVICE)

    if DEVICE == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = load_tokenizer()

    # Load fixed SFT subsets
    print("Loading SFT dataset...")
    train_data, val_data = load_sft_splits(tokenizer)

    print("Training examples:", len(train_data))
    print("Validation examples:", len(val_data))

    # Create DataLoaders
    collate_fn = partial(
        collate_sft,
        pad_id=tokenizer.pad_token_id
    )

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Load pretrained model
    print("\nLoading base model...")
    model = load_causal_lm(device=DEVICE)

    # Generate five examples before training
    eval_examples = load_eval_posts(tokenizer)[:5]

    print("\nGenerating before-SFT summaries...")

    before_outputs = [
        generate_summary(model, tokenizer, example, DEVICE)
        for example in eval_examples
    ]

    # AdamW optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    running_loss = 0.0
    running_steps = 0

    # CSV training log
    log_path = LOG_DIR / "sft_training.csv"

    with open(log_path, "w", newline="", encoding="utf-8") as csvfile:

        writer = csv.writer(csvfile)
        writer.writerow([
            "step",
            "train_loss",
            "validation_loss",
            "elapsed_seconds"
        ])

        global_step = 0
        start_time = time.time()

        model.train()

        for epoch in range(EPOCHS):

            for batch in train_loader:

                optimizer.zero_grad(set_to_none=True)

                loss = calculate_sft_loss(
                    model,
                    batch,
                    DEVICE
                )

                loss.backward()
                
                running_loss += loss.item()
                running_steps += 1

                clip_grad_norm_(
                    model.parameters(),
                    GRAD_CLIP
                )

                optimizer.step()

                global_step += 1

                if global_step % LOG_INTERVAL == 0:

                    # train_loss = loss.item()
                    train_loss = running_loss / running_steps
                    running_loss = 0.0
                    running_steps = 0
                    
                    val_loss = evaluate(
                        model,
                        val_loader,
                        DEVICE
                    )

                    elapsed = time.time() - start_time

                    writer.writerow([
                        global_step,
                        train_loss,
                        val_loss,
                        elapsed
                    ])

                    csvfile.flush()

                    print(
                        f"Step {global_step} | "
                        f"Train Loss: {train_loss:.4f} | "
                        f"Validation Loss: {val_loss:.4f} | "
                        f"Time: {elapsed / 60:.2f} min"
                    )

        # Final validation loss
        final_val_loss = evaluate(
            model,
            val_loader,
            DEVICE
        )

        elapsed = time.time() - start_time

        print("\nTraining completed.")
        print("Final validation loss:", final_val_loss)
        print(f"Total time: {elapsed / 60:.2f} minutes")

        writer.writerow([
            global_step,
            "",
            final_val_loss,
            elapsed
        ])

    # Save final model and tokenizer
    print("\nSaving SFT checkpoint...")

    model.save_pretrained(CHECKPOINT_DIR)
    tokenizer.save_pretrained(CHECKPOINT_DIR)

    print("Checkpoint saved at:", CHECKPOINT_DIR)

    # Generate five examples after training
    print("\nGenerating after-SFT summaries...")

    save_samples(
        model,
        tokenizer,
        eval_examples,
        DEVICE,
        before_outputs
    )

    print("Samples saved at:", SAMPLE_FILE)
    print("SFT stage finished successfully.")


if __name__ == "__main__":
    main()
