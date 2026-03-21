# SkillfulLLM with AdapterHub
### Distilling Prompt Skills into Efficient Local AI Models

🚀 Lightweight Skillful Models • ⚡ Local Deployment • 🔌 Plug-and-Play Adapters • 💸 Massive Cost Reduction

![License](https://img.shields.io/badge/license-MIT-blue)
![GPU](https://img.shields.io/badge/GPU-RTX%204090-success)
![Models](https://img.shields.io/badge/models-0.5B--7B-orange)

---

**SkillfulLLM** is a framework for converting **prompt-based AI skills into efficient local models**.

Instead of repeatedly injecting large skill prompts into expensive LLM APIs, Skillful-LLM **distills skills into lightweight fine-tuned models** that run locally on a single GPU.

The result is a **plug-and-play skill architecture** where small models can perform specialized tasks with **near-zero prompt overhead**.

If you want **fast, private, and cost-efficient AI skills**, this is it.

---

## Why Skillful Model over Skills?

### Skill prompts are expensive

Each skill often requires **hundreds or thousands of tokens** of instructions and demonstrations to be inserted into the model's context window. In large-scale applications, this results in:

- high inference latency  
- extremely high token cost  
- long context windows  
- limited scalability for real-world systems

### Many tasks do not need a large model

At the same time, many skills simply encode **task-specific capabilities** that do not require a full-scale frontier model at inference time. These tasks typically involve well-defined procedural operations such as structured formatting, information extraction, or rule-guided generation, rather than complex reasoning or open-ended problem solving. As a result, repeatedly invoking a large model with extensive skill prompts introduces significant computational overhead without providing proportional benefits.

## Current POC

The current prototype implements a simple planner-executor loop for browser-task routing.

The workflow is:

1. a controller model receives the original user task
2. it reads the adapter registry
3. it selects a local LoRA executor
4. the executor generates browser-style commands
5. the controller returns the final output

This is a minimal proof-of-concept for letting a larger model call a smaller finetuned local executor.

## Repository Structure

- `poc/`  
  Minimal planner-executor prototype, including the adapter registry, executor tool, main entry script, and one demo result file.

- `data/`  
  Final dataset split used in the current experiments:
  - `browser_command_train.jsonl`
  - `browser_command_val.jsonl`
  - `browser_command_test.jsonl`

- `prompts/`  
  Prompt used in the skill-based baseline:
  - `browser_skill_prompt.txt`

- `exp_command_eval/`  
  Evaluation scripts and saved results for:
  - small model + skill prompt
  - finetuned small model
  - qwen3.5-plus + skill prompt
  - token cost comparison

## Current Experimental Takeaways

From the current runs:

- adding a skill prompt to the small model helps only to a limited extent
- the finetuned small model is noticeably more stable
- the finetuned small model can be compared directly with a larger model plus skill prompting
- the finetuned route is also cheaper from the token-cost side

## Minimal Run

The current POC entry point is:

```bash
python -m poc.scripts.run_agent
