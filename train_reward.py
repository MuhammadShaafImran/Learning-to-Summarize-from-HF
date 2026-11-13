"""
Task 3: Stage 2 - reward model r_psi(x, y).

Run from the repository root after SFT has finished:

    python train_reward.py

Input:
    checkpoints/sft/                 SFT transformer + tokenizer

Outputs:
    checkpoints/reward/              reward model + tokenizer
    results/training_logs/reward_training.csv
    results/reward_model_errors.md   three misclassified validation pairs

Architecture (assignment 3.1):
    backbone = model.model from checkpoints/sft/   (no language-model head)
    head     = nn.Linear(hidden_size, 1)
    reward   = head applied to the final hidden state of the LAST REAL
               (non-padding) token of prompt + summary

Loss (assignment 3.2, Bradley-Terry):
    L = -E[ log sigmoid( r(x, y_w) - r(x, y_l) ) ]

Settings (assignment 3.3):
    AdamW, lr = 1e-5, batch size = 8 pairs, 1 epoch over 20,000 pairs
"""

import csv
import time
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel

from starter.config import SEED
from starter.models import load_tokenizer, load_causal_lm
from starter.data import load_preference_splits
from starter.batching import collate_pref

from common import set_seed


# --------------------------------------------------
# 1. Configuration (assignment settings)
# --------------------------------------------------

BATCH_SIZE = 8
LEARNING_RATE = 1e-5
EPOCHS = 1
GRAD_CLIP = 1.0
LOG_INTERVAL = 250

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT = Path(__file__).resolve().parent

SFT_CHECKPOINT = ROOT / "checkpoints" / "sft"
REWARD_CHECKPOINT = ROOT / "checkpoints" / "reward"
LOG_DIR = ROOT / "results" / "training_logs"
ERROR_FILE = ROOT / "results" / "reward_model_errors.md"

REWARD_CHECKPOINT.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Reward model
# --------------------------------------------------

class RewardModel(nn.Module):
    """
    Transformer body from the SFT checkpoint plus a scalar head.

    The head reads the final hidden state of the last real token. The
    tokenizer's padding token is the EOS token, so real tokens are located
    with the attention mask and never with `input_ids != pad_id`.
    """

    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone
        hidden_size = backbone.config.hidden_size

        self.value_head = nn.Linear(hidden_size, 1)

        # Small random init: the reward starts near 0, so the Bradley-Terry
        # loss starts near log(2) and the scores grow smoothly.
        nn.init.normal_(self.value_head.weight, std=0.01)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids:      (B, L) right-padded prompt + summary.
            attention_mask: (B, L) 1 for real tokens, 0 for padding.

        Returns:
            rewards: (B,) one scalar per prompt + summary.
        """
        hidden_states = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).last_hidden_state

        # Index of the last real token in every row.
        last_index = attention_mask.sum(dim=1) - 1
        rows = torch.arange(hidden_states.size(0), device=hidden_states.device)
        last_hidden = hidden_states[rows, last_index]

        return self.value_head(last_hidden).squeeze(-1)

    @classmethod
    def from_sft_checkpoint(cls, checkpoint, device):
        """Load checkpoints/sft/ and keep only the transformer body."""
        causal_lm = load_causal_lm(name_or_path=checkpoint, device=device)

        # Upcast: the SFT config stores bfloat16, and bf16 is both slow and
        # numerically noisy on a T4.
        causal_lm = causal_lm.float()

        # `model.model` is the transformer body; the LM head is dropped.
        model = cls(causal_lm.model)
        del causal_lm

        return model.to(device)


def save_reward_model(reward_model, tokenizer, path):
    """Save backbone + head + config so the model can be rebuilt anywhere."""
    path.mkdir(parents=True, exist_ok=True)

    state_dict = {}
    for key, value in reward_model.backbone.state_dict().items():
        state_dict[f"backbone.{key}"] = value
    for key, value in reward_model.value_head.state_dict().items():
        state_dict[f"value_head.{key}"] = value

    torch.save(state_dict, path / "reward_model.pt")
    reward_model.backbone.config.save_pretrained(path)
    tokenizer.save_pretrained(path)


def load_reward_model(path=REWARD_CHECKPOINT, device="cpu"):
    """
    Rebuild the reward model saved by save_reward_model().

    This is what train_ppo.py and evaluate.py use, so the PPO stage never has
    to depend on the SFT checkpoint being present.
    """
    config = AutoConfig.from_pretrained(path)
    backbone = AutoModel.from_config(config)

    reward_model = RewardModel(backbone)

    state_dict = torch.load(
        path / "reward_model.pt",
        map_location="cpu",
        weights_only=True,
    )

    backbone_state = {
        key[len("backbone."):]: value
        for key, value in state_dict.items()
        if key.startswith("backbone.")
    }
    head_state = {
        key[len("value_head."):]: value
        for key, value in state_dict.items()
        if key.startswith("value_head.")
    }

    reward_model.backbone.load_state_dict(backbone_state)
    reward_model.value_head.load_state_dict(head_state)

    return reward_model.to(device).float().eval()


# --------------------------------------------------
# 3. Bradley-Terry loss and validation accuracy
# --------------------------------------------------

def calculate_reward_loss(reward_model, batch, device):
    """
    -E[ log sigmoid( r(x, y_w) - r(x, y_l) ) ]
    """
    chosen_ids = batch["chosen_ids"].to(device)
    chosen_mask = batch["chosen_mask"].to(device)
    rejected_ids = batch["rejected_ids"].to(device)
    rejected_mask = batch["rejected_mask"].to(device)

    chosen_rewards = reward_model(chosen_ids, chosen_mask)
    rejected_rewards = reward_model(rejected_ids, rejected_mask)

    return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()


@torch.no_grad()
def evaluate_reward(reward_model, val_loader, device):
    """
    Returns:
        mean Bradley-Terry loss
        pairwise accuracy = share of pairs with r(x, y_w) > r(x, y_l)
    """
    was_training = reward_model.training
    reward_model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch in val_loader:
        chosen_ids = batch["chosen_ids"].to(device)
        chosen_mask = batch["chosen_mask"].to(device)
        rejected_ids = batch["rejected_ids"].to(device)
        rejected_mask = batch["rejected_mask"].to(device)

        chosen_rewards = reward_model(chosen_ids, chosen_mask)
        rejected_rewards = reward_model(rejected_ids, rejected_mask)

        total_loss += -F.logsigmoid(
            chosen_rewards - rejected_rewards
        ).sum().item()
        correct += (chosen_rewards > rejected_rewards).sum().item()
        total += chosen_rewards.numel()

    if was_training:
        reward_model.train()

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def save_error_examples(reward_model, val_data, tokenizer, path, n=3):
    """
    Save the first n validation pairs the reward model gets wrong.

    Task 7.1 asks what kinds of pairs it gets wrong; this writes them out.
    """
    reward_model.eval()
    collate_fn = partial(collate_pref, pad_id=tokenizer.pad_token_id)

    lines = [
        "# Reward model: misclassified validation pairs\n\n",
        "Pairs where the reward model scored the human-rejected summary "
        "higher than the chosen one.\n\n",
    ]

    found = 0

    for start in range(0, len(val_data), BATCH_SIZE):
        chunk = val_data[start:start + BATCH_SIZE]
        batch = collate_fn(chunk)

        chosen = reward_model(
            batch["chosen_ids"].to(DEVICE), batch["chosen_mask"].to(DEVICE)
        )
        rejected = reward_model(
            batch["rejected_ids"].to(DEVICE), batch["rejected_mask"].to(DEVICE)
        )

        for index, row in enumerate(chunk):
            if chosen[index].item() > rejected[index].item():
                continue

            found += 1
            lines.append(f"## Wrong pair {found}\n\n")
            lines.append(
                "**Post (tail):**\n\n```\n"
                f"{row['prompt'][-400:]}\n```\n\n"
            )
            lines.append(
                f"**Chosen (human preferred), r = {chosen[index].item():.3f}**"
                f"\n\n{row['chosen']}\n\n"
            )
            lines.append(
                f"**Rejected, r = {rejected[index].item():.3f}**"
                f"\n\n{row['rejected']}\n\n"
            )

            if found == n:
                break

        if found == n:
            break

    if found == 0:
        lines.append("No misclassified validation pairs found.\n")

    path.write_text("".join(lines), encoding="utf-8")


# --------------------------------------------------
# 4. Main training function
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

    print("\nLoading tokenizer...")
    tokenizer = load_tokenizer(SFT_CHECKPOINT)

    print("Loading preference data...")
    train_data, val_data = load_preference_splits(tokenizer)

    print("Preference train / validation pairs:", len(train_data), "/", len(val_data))

    collate_fn = partial(collate_pref, pad_id=tokenizer.pad_token_id)

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

    print("\nLoading SFT transformer as the reward backbone...")
    reward_model = RewardModel.from_sft_checkpoint(SFT_CHECKPOINT, DEVICE)

    print("Backbone dtype:", next(reward_model.parameters()).dtype)
    print("Hidden size:   ", reward_model.backbone.config.hidden_size)

    optimizer = torch.optim.AdamW(
        reward_model.parameters(),
        lr=LEARNING_RATE,
    )

    log_path = LOG_DIR / "reward_training.csv"

    running_loss = 0.0
    running_steps = 0
    epoch_loss_sum = 0.0
    epoch_steps = 0

    global_step = 0
    start_time = time.time()

    reward_model.train()

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

                loss = calculate_reward_loss(reward_model, batch, DEVICE)
                loss.backward()

                clip_grad_norm_(reward_model.parameters(), GRAD_CLIP)
                optimizer.step()

                loss_value = loss.item()
                running_loss += loss_value
                running_steps += 1
                epoch_loss_sum += loss_value
                epoch_steps += 1
                global_step += 1

                if global_step % LOG_INTERVAL == 0:
                    train_loss = running_loss / running_steps
                    val_loss, val_accuracy = evaluate_reward(
                        reward_model, val_loader, DEVICE
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
                        f"Val Pairwise Acc: {val_accuracy:.4f} | "
                        f"Time: {elapsed / 60:.2f} min"
                    )

                    running_loss = 0.0
                    running_steps = 0

        final_train_loss = epoch_loss_sum / max(epoch_steps, 1)
        final_val_loss, final_val_accuracy = evaluate_reward(
            reward_model, val_loader, DEVICE
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
    print(f"Final train loss:          {final_train_loss:.4f}")
    print(f"Final validation loss:     {final_val_loss:.4f}")
    print(f"Final pairwise accuracy:   {final_val_accuracy:.4f}")
    print("(chance level is 0.5000)")
    print(f"Total time: {elapsed / 60:.2f} minutes")

    print("\nSaving reward model checkpoint...")
    save_reward_model(reward_model, tokenizer, REWARD_CHECKPOINT)
    print("Checkpoint saved at:", REWARD_CHECKPOINT)

    print("\nSaving three misclassified validation pairs...")
    save_error_examples(reward_model, val_data, tokenizer, ERROR_FILE)
    print("Errors saved at:", ERROR_FILE)

    print("Reward-model stage finished successfully.")


if __name__ == "__main__":
    main()
