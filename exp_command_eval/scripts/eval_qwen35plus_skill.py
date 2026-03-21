import os
import re
import json
import time
from collections import Counter
from datasets import load_dataset
from openai import OpenAI
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# =========================
# CONFIG
# =========================

TEST_FILE = PROJECT_ROOT / "data" / "browser_command_test.jsonl"
PROMPT_FILE = PROJECT_ROOT / "prompts" / "browser_skill_prompt.txt"
OUTPUT_DIR = PROJECT_ROOT / "exp_command_eval" / "results"
OUTPUT_JSON = OUTPUT_DIR / "qwen35plus_skill_eval.json"

MODEL_NAME = "qwen3.5-plus"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

TEMPERATURE = 0
MAX_TOKENS = 512
MAX_RETRIES = 3
SLEEP_BETWEEN_CALLS = 1.0

# =========================
# UTIL
# =========================

def normalize_quotes(text: str) -> str:
    return (
        text.replace("“", '"')
            .replace("”", '"')
            .replace("‘", "'")
            .replace("’", "'")
    )

def normalize_line(line: str) -> str:
    line = normalize_quotes(line)
    line = line.strip()
    line = re.sub(r"\s+", " ", line)
    return line

def extract_user_query(example):
    for m in example["messages"]:
        if m["role"] == "user":
            return m["content"].strip()
    raise ValueError("No user message found")

def extract_assistant_target(example):
    for m in example["messages"]:
        if m["role"] == "assistant":
            return m["content"].strip()
    raise ValueError("No assistant message found")

def extract_actionable_lines(text: str):
    """
    只提取我们关心的目标行：
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
    粗暴但实用地抓明显坏掉的重复：
    - 某一条命令重复 >= 3 次
    - 行数远超 gold
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

def safe_div(x, y):
    return x / y if y else 0.0

# =========================
# LOAD
# =========================

os.makedirs(OUTPUT_DIR, exist_ok=True)

api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    raise RuntimeError("DASHSCOPE_API_KEY not set")

with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    skill_prompt = f.read().strip()

dataset = load_dataset("json", data_files=TEST_FILE, split="train")

client = OpenAI(
    api_key=api_key,
    base_url=BASE_URL,
)

print("MODEL =", MODEL_NAME)
print("TEST_FILE =", TEST_FILE)
print("PROMPT_FILE =", PROMPT_FILE)
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
    "avg_prompt_tokens": 0.0,
    "avg_completion_tokens": 0.0,
    "avg_total_tokens": 0.0,
    "by_route": {
        "execute": {"total": 0, "route_acc": 0, "exact_acc": 0, "first_cmd_acc": 0, "avg_line_f1": 0.0},
        "clarify": {"total": 0, "route_acc": 0, "exact_acc": 0, "nonempty_question_acc": 0},
        "cannot_execute": {"total": 0, "route_acc": 0, "exact_acc": 0, "reason_acc": 0},
        "unknown": {"total": 0}
    }
}

execute_f1s = []
execute_cmd_count_diffs = []

prompt_tokens_all = []
completion_tokens_all = []
total_tokens_all = []

results = []

for i, ex in enumerate(dataset, 1):
    query = extract_user_query(ex)
    gold_output = extract_assistant_target(ex)
    gold_lines = extract_actionable_lines(gold_output)
    gold_route = detect_route(gold_lines)

    pred_raw = None
    usage_obj = None
    last_err = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": skill_prompt},
                    {"role": "user", "content": query},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )

            pred_raw = resp.choices[0].message.content.strip() if resp.choices[0].message.content else ""
            usage_obj = resp.usage
            break
        except Exception as e:
            last_err = str(e)
            print(f"[WARN] sample {i} attempt {attempt} failed: {e}")
            time.sleep(2.0 * attempt)

    if pred_raw is None:
        pred_raw = f"CANNOT_EXECUTE \"api_error: {last_err}\""

    pred_lines = extract_actionable_lines(pred_raw)
    pred_route = detect_route(pred_lines)

    route_ok = (pred_route == gold_route)
    actionable_exact = (pred_lines == gold_lines)
    chatter_free = not has_leading_chatter(pred_raw, pred_lines)
    pathological_repetition = has_pathological_repetition(pred_lines, gold_lines)

    prompt_tokens = getattr(usage_obj, "prompt_tokens", None) if usage_obj is not None else None
    completion_tokens = getattr(usage_obj, "completion_tokens", None) if usage_obj is not None else None
    total_tokens = getattr(usage_obj, "total_tokens", None) if usage_obj is not None else None

    if prompt_tokens is not None:
        prompt_tokens_all.append(prompt_tokens)
    if completion_tokens is not None:
        completion_tokens_all.append(completion_tokens)
    if total_tokens is not None:
        total_tokens_all.append(total_tokens)

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
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
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

    time.sleep(SLEEP_BETWEEN_CALLS)

# =========================
# FINAL AGG
# =========================

summary["route_acc"] = safe_div(summary["route_acc"], summary["total"])
summary["actionable_exact_acc"] = safe_div(summary["actionable_exact_acc"], summary["total"])
summary["chatter_free_acc"] = safe_div(summary["chatter_free_acc"], summary["total"])
summary["no_pathological_repetition_acc"] = safe_div(summary["no_pathological_repetition_acc"], summary["total"])

if execute_f1s:
    summary["avg_execute_line_f1"] = sum(execute_f1s) / len(execute_f1s)
if execute_cmd_count_diffs:
    summary["avg_execute_cmd_count_diff"] = sum(execute_cmd_count_diffs) / len(execute_cmd_count_diffs)

if prompt_tokens_all:
    summary["avg_prompt_tokens"] = sum(prompt_tokens_all) / len(prompt_tokens_all)
if completion_tokens_all:
    summary["avg_completion_tokens"] = sum(completion_tokens_all) / len(completion_tokens_all)
if total_tokens_all:
    summary["avg_total_tokens"] = sum(total_tokens_all) / len(total_tokens_all)

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