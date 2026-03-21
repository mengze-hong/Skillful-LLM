import os
import re
import json
from pathlib import Path
from collections import Counter, defaultdict

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# =========================
# CONFIG
# =========================

TEST_FILE = PROJECT_ROOT / "data" / "browser_command_test.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "exp_command_eval" / "results"
OUTPUT_JSON = OUTPUT_DIR / "base_skill_test30_eval.json“

ADAPTER_DIR = None
PROMPT_FILE = PROJECT_ROOT / "prompts" / "browser_skill_prompt.txt"

MAX_NEW_TOKENS = 160
DO_SAMPLE = False
TEMPERATURE = 1.0
REPETITION_PENALTY = 1.05

# =========================
# UTIL
# =========================

def get_base_model_path():
    snapshot_root = os.getenv(
        "QWEN25_SNAPSHOT_DIR",
        "/root/autodl-tmp/cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots"
    )
    snapshots_dir = Path(snapshot_root)
    subdirs = [p for p in snapshots_dir.iterdir() if p.is_dir()]
    if not subdirs:
        raise RuntimeError("No local Qwen2.5-7B-Instruct snapshot found.")
    return str(subdirs[0])

def normalize_quotes(text: str) -> str:
    return (
        text.replace("“", '"')
            .replace("”", '"')
            .replace("‘", "'")
            .replace("’", "'")
    )

def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())

def normalize_line(line: str) -> str:
    line = normalize_quotes(line)
    line = line.strip()
    line = re.sub(r"\s+", " ", line)
    return line

def extract_assistant_target(example):
    msgs = example["messages"]
    for m in msgs:
        if m["role"] == "assistant":
            return m["content"].strip()
    raise ValueError("No assistant message found")

def extract_user_query(example):
    msgs = example["messages"]
    for m in msgs:
        if m["role"] == "user":
            return m["content"].strip()
    raise ValueError("No user message found")

def extract_actionable_lines(text: str):
    """
    只提取我们关心的“目标输出行”：
    - browser ...
    - ASK_USER "..."
    - CANNOT_EXECUTE "..."
    """
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    out = []
    for line in lines:
        nl = normalize_line(line)
        if nl.startswith("browser "):
            out.append(nl)
        elif nl.startswith("ASK_USER "):
            out.append(nl)
        elif nl.startswith("CANNOT_EXECUTE "):
            out.append(nl)
    return out

def detect_route(action_lines):
    if not action_lines:
        return "unknown"
    first = action_lines[0]
    if first.startswith("browser "):
        return "execute"
    if first.startswith("ASK_USER "):
        return "clarify"
    if first.startswith("CANNOT_EXECUTE "):
        return "cannot_execute"
    return "unknown"

def extract_question_from_ask_user(line: str):
    m = re.match(r'^ASK_USER\s+"(.*)"$', normalize_line(line))
    return m.group(1).strip() if m else ""

def extract_reason_from_cannot(line: str):
    m = re.match(r'^CANNOT_EXECUTE\s+"(.*)"$', normalize_line(line))
    return m.group(1).strip() if m else ""

def multiset_f1(pred_lines, gold_lines):
    pred = Counter(pred_lines)
    gold = Counter(gold_lines)
    overlap = sum((pred & gold).values())
    pred_n = sum(pred.values())
    gold_n = sum(gold.values())
    if pred_n == 0 and gold_n == 0:
        return 1.0
    if pred_n == 0 or gold_n == 0:
        return 0.0
    precision = overlap / pred_n
    recall = overlap / gold_n
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def has_pathological_repetition(lines, gold_lines):
    """
    粗暴但实用地抓“明显坏掉的重复”：
    - 行数远超 gold
    - 某一条命令重复 >= 3 次
    """
    if not lines:
        return False
    c = Counter(lines)
    if any(v >= 3 for v in c.values()):
        return True
    if len(lines) > max(12, len(gold_lines) * 2 + 2):
        return True
    return False

def has_leading_chatter(raw_output: str, action_lines):
    if not action_lines:
        return True
    first_action = action_lines[0]
    idx = raw_output.find(first_action)
    if idx == -1:
        return True
    prefix = raw_output[:idx].strip()
    return len(prefix) > 0

def load_prompt(prompt_file):
    if not prompt_file:
        return None
    with open(prompt_file, "r", encoding="utf-8") as f:
        return f.read().strip()

# =========================
# LOAD MODEL
# =========================

os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_MODEL = get_base_model_path()
SKILL_PROMPT = load_prompt(PROMPT_FILE)

print("BASE_MODEL =", BASE_MODEL)
print("ADAPTER_DIR =", ADAPTER_DIR)
print("PROMPT_FILE =", PROMPT_FILE)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer_source = ADAPTER_DIR if ADAPTER_DIR else BASE_MODEL
print("loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)

print("loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

if ADAPTER_DIR:
    print("loading adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
else:
    model = base_model

model.eval()

# =========================
# LOAD DATA
# =========================

dataset = load_dataset("json", data_files=TEST_FILE, split="train")
print("test size =", len(dataset))

# =========================
# EVAL
# =========================

summary = {
    "total": 0,
    "route_acc": 0,
    "actionable_exact_acc": 0,
    "chatter_free_acc": 0,
    "avg_execute_line_f1": 0.0,
    "avg_execute_cmd_count_diff": 0.0,
    "no_pathological_repetition_acc": 0,
    "by_route": {
        "execute": {"total": 0, "route_acc": 0, "exact_acc": 0, "first_cmd_acc": 0, "avg_line_f1": 0.0},
        "clarify": {"total": 0, "route_acc": 0, "exact_acc": 0, "nonempty_question_acc": 0},
        "cannot_execute": {"total": 0, "route_acc": 0, "exact_acc": 0, "reason_acc": 0},
        "unknown": {"total": 0}
    }
}

execute_f1s = []
execute_cmd_count_diffs = []

results = []

for i, ex in enumerate(dataset, 1):
    query = extract_user_query(ex)
    gold_output = extract_assistant_target(ex)
    gold_lines = extract_actionable_lines(gold_output)
    gold_route = detect_route(gold_lines)

    # 构造输入
    if SKILL_PROMPT is None:
        messages = [{"role": "user", "content": query}]
    else:
        messages = [
            {"role": "system", "content": SKILL_PROMPT},
            {"role": "user", "content": query},
        ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen = outputs[0][inputs["input_ids"].shape[1]:]
    pred_raw = tokenizer.decode(gen, skip_special_tokens=True).strip()
    pred_lines = extract_actionable_lines(pred_raw)
    pred_route = detect_route(pred_lines)

    route_ok = (pred_route == gold_route)
    actionable_exact = (pred_lines == gold_lines)
    chatter_free = not has_leading_chatter(pred_raw, pred_lines)
    pathological_repetition = has_pathological_repetition(pred_lines, gold_lines)

    record = {
        "id": ex.get("id", f"sample_{i}"),
        "category": ex.get("category"),
        "source": ex.get("source"),
        "query_language": ex.get("query_language"),
        "query": query,
        "gold_output": gold_output,
        "pred_raw": pred_raw,
        "gold_lines": gold_lines,
        "pred_lines": pred_lines,
        "gold_route": gold_route,
        "pred_route": pred_route,
        "route_ok": route_ok,
        "actionable_exact": actionable_exact,
        "chatter_free": chatter_free,
        "pathological_repetition": pathological_repetition,
    }

    summary["total"] += 1
    summary["by_route"].setdefault(gold_route, {"total": 0})
    summary["by_route"][gold_route]["total"] += 1

    if route_ok:
        summary["route_acc"] += 1
        summary["by_route"][gold_route]["route_acc"] = summary["by_route"][gold_route].get("route_acc", 0) + 1

    if actionable_exact:
        summary["actionable_exact_acc"] += 1
        summary["by_route"][gold_route]["exact_acc"] = summary["by_route"][gold_route].get("exact_acc", 0) + 1

    if chatter_free:
        summary["chatter_free_acc"] += 1

    if not pathological_repetition:
        summary["no_pathological_repetition_acc"] += 1

    if gold_route == "execute":
        line_f1 = multiset_f1(pred_lines, gold_lines)
        execute_f1s.append(line_f1)
        cmd_count_diff = abs(len(pred_lines) - len(gold_lines))
        execute_cmd_count_diffs.append(cmd_count_diff)

        first_cmd_ok = False
        if pred_lines and gold_lines:
            first_cmd_ok = (pred_lines[0] == gold_lines[0])

        record["execute_line_f1"] = line_f1
        record["execute_cmd_count_diff"] = cmd_count_diff
        record["execute_first_cmd_ok"] = first_cmd_ok

        if first_cmd_ok:
            summary["by_route"]["execute"]["first_cmd_acc"] += 1

    elif gold_route == "clarify":
        gold_q = extract_question_from_ask_user(gold_lines[0]) if gold_lines else ""
        pred_q = extract_question_from_ask_user(pred_lines[0]) if pred_lines else ""
        nonempty_question_ok = pred_route == "clarify" and len(pred_q.strip()) > 0
        record["gold_question"] = gold_q
        record["pred_question"] = pred_q
        record["nonempty_question_ok"] = nonempty_question_ok
        if nonempty_question_ok:
            summary["by_route"]["clarify"]["nonempty_question_acc"] += 1

    elif gold_route == "cannot_execute":
        gold_reason = extract_reason_from_cannot(gold_lines[0]) if gold_lines else ""
        pred_reason = extract_reason_from_cannot(pred_lines[0]) if pred_lines else ""
        reason_ok = (gold_reason == pred_reason)
        record["gold_reason"] = gold_reason
        record["pred_reason"] = pred_reason
        record["reason_ok"] = reason_ok
        if reason_ok:
            summary["by_route"]["cannot_execute"]["reason_acc"] += 1

    results.append(record)

    print("=" * 80)
    print(f"[{i}/{len(dataset)}] {record['id']} | gold_route={gold_route} | pred_route={pred_route} | route_ok={route_ok}")
    print("QUERY:", query)
    print("PRED:")
    print(pred_raw[:1000])
    print("-" * 80)

# =========================
# FINAL AGG
# =========================

def safe_div(x, y):
    return x / y if y else 0.0

summary["route_acc"] = safe_div(summary["route_acc"], summary["total"])
summary["actionable_exact_acc"] = safe_div(summary["actionable_exact_acc"], summary["total"])
summary["chatter_free_acc"] = safe_div(summary["chatter_free_acc"], summary["total"])
summary["no_pathological_repetition_acc"] = safe_div(summary["no_pathological_repetition_acc"], summary["total"])

if execute_f1s:
    summary["avg_execute_line_f1"] = sum(execute_f1s) / len(execute_f1s)
if execute_cmd_count_diffs:
    summary["avg_execute_cmd_count_diff"] = sum(execute_cmd_count_diffs) / len(execute_cmd_count_diffs)

for route in ["execute", "clarify", "cannot_execute"]:
    total = summary["by_route"][route]["total"]
    if total == 0:
        continue
    if "route_acc" in summary["by_route"][route]:
        summary["by_route"][route]["route_acc"] = safe_div(summary["by_route"][route]["route_acc"], total)
    if "exact_acc" in summary["by_route"][route]:
        summary["by_route"][route]["exact_acc"] = safe_div(summary["by_route"][route]["exact_acc"], total)

summary["by_route"]["execute"]["avg_line_f1"] = safe_div(sum(execute_f1s), len(execute_f1s))
summary["by_route"]["execute"]["first_cmd_acc"] = safe_div(
    summary["by_route"]["execute"]["first_cmd_acc"],
    summary["by_route"]["execute"]["total"]
)
summary["by_route"]["clarify"]["nonempty_question_acc"] = safe_div(
    summary["by_route"]["clarify"]["nonempty_question_acc"],
    summary["by_route"]["clarify"]["total"]
)
summary["by_route"]["cannot_execute"]["reason_acc"] = safe_div(
    summary["by_route"]["cannot_execute"]["reason_acc"],
    summary["by_route"]["cannot_execute"]["total"]
)

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 100)
print("FINAL SUMMARY")
print(json.dumps(summary, ensure_ascii=False, indent=2))
print("=" * 100)
print("Saved to:", OUTPUT_JSON)