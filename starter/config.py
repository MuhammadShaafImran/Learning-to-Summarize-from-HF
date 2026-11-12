MODEL_NAME = "HuggingFaceTB/SmolLM2-135M" # base model for every stage
SFT_DATA = "CarperAI/openai_summarize_tldr"
PREF_DATA = "CarperAI/openai_summarize_comparisons"

MAX_PROMPT_TOKENS = 448 # post + "TL;DR :" longer than this is dropped
MAX_SUMMARY_TOKENS = 48 # summaries longer than this are dropped

SEED = 42
N_SFT_TRAIN, N_SFT_VAL = 20000 , 500
N_PREF_TRAIN, N_PREF_VAL = 20000 , 1000
N_EVAL_POSTS = 300