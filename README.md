# Skillful-LLM with AdapterHub
### Distilling Prompt Skills into Efficient Local AI Models

🚀 Lightweight Skillful Models • ⚡ Local Deployment • 🔌 Plug-and-Play Adapters • 💸 Massive Cost Reduction

![License](https://img.shields.io/badge/license-MIT-blue)
![GPU](https://img.shields.io/badge/GPU-RTX%204090-success)
![Models](https://img.shields.io/badge/models-0.5B--7B-orange)

---

**Skillful-LLM** is a framework for converting **prompt-based AI skills into efficient local models**.

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

