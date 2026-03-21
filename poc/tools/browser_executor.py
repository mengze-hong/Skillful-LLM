import json
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POC_ROOT = PROJECT_ROOT / "poc"
REGISTRY_FILE = POC_ROOT / "config" / "adapter_registry.json"

_MODEL_CACHE = {}

def _get_local_base_model_path():
    snapshot_root = os.getenv(
        "QWEN25_SNAPSHOT_DIR",
        "/root/autodl-tmp/cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots"
    )
    snapshots_dir = Path(snapshot_root)
    subdirs = [p for p in snapshots_dir.iterdir() if p.is_dir()]
    if not subdirs:
        raise RuntimeError("No local Qwen2.5-7B-Instruct snapshot found.")
    return str(subdirs[0])

def _load_registry():
    with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def list_available_adapters():
    registry = _load_registry()
    return registry

def _load_executor(adapter_name: str):
    if adapter_name in _MODEL_CACHE:
        return _MODEL_CACHE[adapter_name]

    registry = _load_registry()
    if adapter_name not in registry:
        raise ValueError(f"Unknown adapter: {adapter_name}")

    adapter_path = Path(registry[adapter_name]["adapter_path"])
    if not adapter_path.is_absolute():
        adapter_path = PROJECT_ROOT / adapter_path

    base_model_path = _get_local_base_model_path()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    _MODEL_CACHE[adapter_name] = (tokenizer, model)
    return tokenizer, model

def run_executor(adapter_name: str, subtask: str, max_new_tokens: int = 160):
    tokenizer, model = _load_executor(adapter_name)

    messages = [{"role": "user", "content": subtask}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen = outputs[0][inputs["input_ids"].shape[1]:]
    output_text = tokenizer.decode(gen, skip_special_tokens=True).strip()

    return {
        "adapter_name": adapter_name,
        "subtask": subtask,
        "output": output_text,
        "input_tokens": int(inputs["input_ids"].shape[1]),
        "output_tokens": int(gen.shape[0]),
        "total_tokens": int(inputs["input_ids"].shape[1] + gen.shape[0]),
    }
