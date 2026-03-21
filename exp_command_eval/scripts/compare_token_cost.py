import os
import json
from pathlib import Path
from statistics import mean
from datasets import load_dataset
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# ========= PATH CONFIG =========
FT_EVAL_JSON = PROJECT_ROOT / "exp_command_eval" / "results" / "small_ft_eval.json"
SKILL_EVAL_JSON = PROJECT_ROOT / "exp_command_eval" / "results" / "small_skill_eval.json"

TRAIN_FILE = PROJECT_ROOT / "data" / "browser_command_train.jsonl"
SKILL_PROMPT_FILE = PROJECT_ROOT / "prompts" / "browser_skill_prompt.txt"

OUT_JSON = PROJECT_ROOT / "exp_command_eval" / "results" / "token_cost_summary.json"

NUM_EPOCHS = 3

# ========= FIND LOCAL BASE MODEL =========
snapshot_root = os.getenv(
    "QWEN25_SNAPSHOT_DIR",
    "/root/autodl-tmp/cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots"
)
snapshots_dir = Path(snapshot_root)
subdirs = [p for p in snapshots_dir.iterdir() if p.is_dir()]
assert len(subdirs) >= 1, "No local Qwen2.5-7B-Instruct snapshot found."
BASE_MODEL = str(subdirs[0])

print("Using tokenizer from:", BASE_MODEL)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

# ========= LOAD FILES =========
with open(FT_EVAL_JSON, "r", encoding="utf-8") as f:
    ft_eval = json.load(f)

with open(SKILL_EVAL_JSON, "r", encoding="utf-8") as f:
    skill_eval = json.load(f)

with open(SKILL_PROMPT_FILE, "r", encoding="utf-8") as f:
    skill_prompt = f.read().strip()

# ========= TOKEN COUNT HELPERS =========
def count_tokens_text(text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False).input_ids)

def build_ft_input(query: str) -> str:
    messages = [{"role": "user", "content": query}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

def build_skill_input(query: str) -> str:
    messages = [
        {"role": "system", "content": skill_prompt},
        {"role": "user", "content": query},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

# ========= INFERENCE TOKEN COST =========
ft_input_tokens = []
ft_output_tokens = []
ft_total_tokens = []

for r in ft_eval["results"]:
    query = r["query"]
    pred_raw = r["pred_raw"]
    inp = build_ft_input(query)
    in_tok = count_tokens_text(inp)
    out_tok = count_tokens_text(pred_raw)
    ft_input_tokens.append(in_tok)
    ft_output_tokens.append(out_tok)
    ft_total_tokens.append(in_tok + out_tok)

skill_input_tokens = []
skill_output_tokens = []
skill_total_tokens = []

for r in skill_eval["results"]:
    query = r["query"]
    pred_raw = r["pred_raw"]
    inp = build_skill_input(query)
    in_tok = count_tokens_text(inp)
    out_tok = count_tokens_text(pred_raw)
    skill_input_tokens.append(in_tok)
    skill_output_tokens.append(out_tok)
    skill_total_tokens.append(in_tok + out_tok)

# ========= TRAINING TOKEN COST =========
train_ds = load_dataset("json", data_files=TRAIN_FILE, split="train")

train_example_tokens = []
for ex in train_ds:
    text = tokenizer.apply_chat_template(
        ex["messages"],
        tokenize=False,
        add_generation_prompt=False
    )
    train_example_tokens.append(count_tokens_text(text))

train_tokens_per_epoch = sum(train_example_tokens)
train_tokens_total = train_tokens_per_epoch * NUM_EPOCHS

# ========= SUMMARY =========
avg_ft_in = mean(ft_input_tokens)
avg_ft_out = mean(ft_output_tokens)
avg_ft_total = mean(ft_total_tokens)

avg_skill_in = mean(skill_input_tokens)
avg_skill_out = mean(skill_output_tokens)
avg_skill_total = mean(skill_total_tokens)

avg_saved_per_query = avg_skill_total - avg_ft_total

if avg_saved_per_query > 0:
    break_even_queries = train_tokens_total / avg_saved_per_query
else:
    break_even_queries = None

summary = {
    "base_model": BASE_MODEL,
    "num_epochs": NUM_EPOCHS,
    "test_size": len(ft_eval["results"]),
    "finetuned_only_query": {
        "avg_input_tokens": avg_ft_in,
        "avg_output_tokens": avg_ft_out,
        "avg_total_tokens": avg_ft_total,
        "sum_total_tokens_on_test30": sum(ft_total_tokens),
    },
    "base_plus_skill_prompt": {
        "avg_input_tokens": avg_skill_in,
        "avg_output_tokens": avg_skill_out,
        "avg_total_tokens": avg_skill_total,
        "sum_total_tokens_on_test30": sum(skill_total_tokens),
    },
    "delta": {
        "avg_saved_input_tokens_per_query": avg_skill_in - avg_ft_in,
        "avg_saved_output_tokens_per_query": avg_skill_out - avg_ft_out,
        "avg_saved_total_tokens_per_query": avg_saved_per_query,
    },
    "training_cost": {
        "train_examples": len(train_ds),
        "train_tokens_per_epoch": train_tokens_per_epoch,
        "train_tokens_total_all_epochs": train_tokens_total,
        "break_even_queries": break_even_queries,
    }
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(json.dumps(summary, ensure_ascii=False, indent=2))
print("Saved to:", OUT_JSON)