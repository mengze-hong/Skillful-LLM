import os
import json
import time
import re
from pathlib import Path

from openai import OpenAI
from poc.tools.browser_executor import list_available_adapters, run_executor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POC_ROOT = PROJECT_ROOT / "poc"

DEMO_TASKS_FILE = PROJECT_ROOT / "agent_loop" / "demo_tasks.json"
OUT_FILE = POC_ROOT / "results" / "agent_poc_results.json"

PLANNER_MODEL = "qwen3.5-plus"
PLANNER_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MAX_ROUNDS = 2

def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()

def parse_planner_json(text: str):
    text = strip_code_fence(text)
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return json.loads(m.group(0))
        raise

api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    raise RuntimeError("DASHSCOPE_API_KEY not set")

planner_client = OpenAI(
    api_key=api_key,
    base_url=PLANNER_BASE_URL,
)

available_adapters = list_available_adapters()

PLANNER_SYSTEM_PROMPT = f"""
You are a controller model for a local browser executor.

Available adapters:
{json.dumps(available_adapters, ensure_ascii=False, indent=2)}

Given a user task, decide whether to call one local adapter or return a final answer.

Return JSON only:
{{
  "action": "call_executor" or "final",
  "adapter_name": "string or null",
  "subtask": "string or null",
  "final_output": "string or null"
}}

Rules:
- Do not write browser commands yourself.
- If browser action is needed, use action="call_executor".
- Choose adapter_name from the registry.
- Keep subtask short and executable.
- If executor output is already enough, return action="final".
""".strip()

def planner_round(original_task, history):
    history_text = json.dumps(history, ensure_ascii=False, indent=2)

    user_prompt = f"""
User task:
{original_task}

History:
{history_text}

Return the next action in JSON.
""".strip()

    t0 = time.time()
    resp = planner_client.chat.completions.create(
        model=PLANNER_MODEL,
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=512,
    )
    latency = time.time() - t0

    raw = resp.choices[0].message.content.strip() if resp.choices[0].message.content else ""
    parsed = parse_planner_json(raw)

    usage = {
        "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
        "completion_tokens": getattr(resp.usage, "completion_tokens", None),
        "total_tokens": getattr(resp.usage, "total_tokens", None),
    }

    return {
        "raw": raw,
        "parsed": parsed,
        "latency_sec": latency,
        "usage": usage,
    }

with open(DEMO_TASKS_FILE, "r", encoding="utf-8") as f:
    demo_tasks = json.load(f)

all_results = []

for item in demo_tasks:
    task_id = item["id"]
    original_task = item["task"]

    print("=" * 80)
    print(f"Task: {task_id}")
    print(original_task)

    history = []
    planner_logs = []
    executor_logs = []
    final_output = None

    total_planner_time = 0.0
    total_executor_time = 0.0

    for round_idx in range(1, MAX_ROUNDS + 1):
        planner_res = planner_round(original_task, history)
        planner_logs.append(planner_res)
        total_planner_time += planner_res["latency_sec"]

        print(
            f"\n[Planner {round_idx}] "
            f"action={planner_res['parsed'].get('action')} "
            f"adapter={planner_res['parsed'].get('adapter_name')}"
        )
        
        action = planner_res["parsed"].get("action", "").strip()

        if action == "final":
            final_output = planner_res["parsed"].get("final_output", "").strip()
            break

        if action != "call_executor":
            final_output = f'CANNOT_EXECUTE "planner_invalid_action: {action}"'
            break

        adapter_name = planner_res["parsed"].get("adapter_name", "").strip()
        subtask = planner_res["parsed"].get("subtask", "").strip()

        if adapter_name not in available_adapters:
            final_output = f'CANNOT_EXECUTE "unknown_adapter: {adapter_name}"'
            break

        if not subtask:
            final_output = 'CANNOT_EXECUTE "planner_empty_subtask"'
            break

        t0 = time.time()
        exec_res = run_executor(adapter_name, subtask)
        total_executor_time += time.time() - t0
        executor_logs.append(exec_res)

        print(f"[Executor {round_idx}]")
        print(exec_res["output"])

        history.append({
            "round": round_idx,
            "adapter_name": adapter_name,
            "planner_subtask": subtask,
            "executor_output": exec_res["output"]
        })

    if final_output is None:
        if executor_logs:
            final_output = executor_logs[-1]["output"]
        else:
            final_output = 'CANNOT_EXECUTE "no_result"'

    result = {
        "task_id": task_id,
        "original_task": original_task,
        "planner_logs": planner_logs,
        "executor_logs": executor_logs,
        "final_output": final_output,
        "planner_total_latency_sec": total_planner_time,
        "executor_total_latency_sec": total_executor_time,
        "end_to_end_latency_sec": total_planner_time + total_executor_time
    }

    all_results.append(result)

    print("\n[FINAL OUTPUT]")
    print(final_output)
    print("[LATENCY] planner={:.2f}s executor={:.2f}s total={:.2f}s".format(
        total_planner_time, total_executor_time, total_planner_time + total_executor_time
    ))

with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print("\nSaved to:", OUT_FILE)
