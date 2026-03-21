import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POC_ROOT = PROJECT_ROOT / "poc"

INPUT_FILE = POC_ROOT / "results" / "agent_poc_results.json"
OUTPUT_FILE = POC_ROOT / "results" / "agent_demo_results.json"


def round_if_number(x, ndigits=2):
    if isinstance(x, (int, float)):
        return round(x, ndigits)
    return x


def simplify_planner_logs(planner_logs):
    out = []
    for item in planner_logs:
        parsed = item.get("parsed", {}) or {}
        usage = item.get("usage", {}) or {}

        out.append({
            "action": parsed.get("action"),
            "adapter_name": parsed.get("adapter_name"),
            "subtask": parsed.get("subtask"),
            "final_output": parsed.get("final_output"),
            "latency_sec": round_if_number(item.get("latency_sec")),
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            }
        })
    return out


def simplify_executor_logs(executor_logs):
    out = []
    for item in executor_logs:
        out.append({
            "adapter_name": item.get("adapter_name"),
            "subtask": item.get("subtask"),
            "output": item.get("output"),
            "input_tokens": item.get("input_tokens"),
            "output_tokens": item.get("output_tokens"),
            "total_tokens": item.get("total_tokens"),
        })
    return out


def simplify_results(data):
    cleaned = []
    for item in data:
        cleaned.append({
            "task_id": item.get("task_id"),
            "original_task": item.get("original_task"),
            "planner_steps": simplify_planner_logs(item.get("planner_logs", [])),
            "executor_steps": simplify_executor_logs(item.get("executor_logs", [])),
            "final_output": item.get("final_output"),
            "planner_total_latency_sec": round_if_number(item.get("planner_total_latency_sec")),
            "executor_total_latency_sec": round_if_number(item.get("executor_total_latency_sec")),
            "end_to_end_latency_sec": round_if_number(item.get("end_to_end_latency_sec")),
        })
    return cleaned


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned = simplify_results(data)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    print(f"Saved cleaned result to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
