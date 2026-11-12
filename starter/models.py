import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .config import MODEL_NAME

def load_tokenizer(name=MODEL_NAME):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:  # SmolLM2 has no pad token
        tok.pad_token = tok.eos_token
    return tok

def load_causal_lm(name_or_path=MODEL_NAME, device=None):
    """Plain pretrained causal LM (no heads added). Use it for the
    SFT policy, the frozen reference model, and as the backbone of the
    reward model and of the PPO value head."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(name_or_path)
    return model.to(device)