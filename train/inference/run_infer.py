"""Agent Parakeet–Qwen: một phiên ToolEnv/mẫu và file nộp lấy từ trạng thái thực."""
from __future__ import annotations

import json
from pathlib import Path
import re
import tempfile
import time
from typing import Any

from data.preprocess import SplitRoot, catalog_tools, filter_installed_tools, find_split, load_catalog, load_samples
from train.asr.train_asr import load_asr_model, transcribe_paths
from train.llm.train_llm import assistant_call, build_agent_messages, chat_tools, load_llm_model


# 1. ToolEnv giữ trạng thái trong bộ nhớ; mọi call thực sự đã chạy đều được ghi lại.
class ToolRuntime:
    def __init__(self, split_dir: str | Path, sample_id: str, root: SplitRoot | None = None) -> None:
        self.split_dir = Path(split_dir).resolve()
        split, _ = find_split(self.split_dir)
        self.root = root or SplitRoot(self.split_dir, split)
        self.env = self.root.env(sample_id)

    def execute(self, tool_name: str, arguments: dict) -> dict:
        return self.env.call(tool_name, arguments)


def parse_tool_calls(text: str) -> list[dict]:
    """Đọc JSON trong thẻ tool_call do chat template của Qwen sử dụng."""
    blocks = re.findall(r"<tool_call>(.*?)</tool_call>", text, flags=re.DOTALL)
    if not blocks:
        if "<tool_call>" in text:
            raise ValueError("Truncated tool_call block")
        return []
    calls = []
    for block in blocks:
        value = json.loads(block)
        if not isinstance(value, dict):
            raise ValueError("Tool call JSON must be an object")
        name, arguments = value.get("name"), value.get("arguments", {})
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not isinstance(name, str) or not isinstance(arguments, dict):
            raise ValueError("Tool call must contain a string name and object arguments")
        calls.append({"name": name, "arguments": arguments})
    return calls


# 2. Agent dùng cùng prompt với SFT; dừng theo deadline và số lượt.
class LifeMateAgent:
    def __init__(self, asr_model: Any, llm_model: Any, time_budget_seconds: float = 27.0, *,
                 tokenizer: Any = None, tools: dict | None = None, options: dict | None = None,
                 max_tool_turns: int = 8, transcribe_batch_size: int = 1) -> None:
        self.asr_model, self.llm_model = asr_model, llm_model
        self.tokenizer, self.tools, self.options = tokenizer, tools or {}, options or {}
        self.time_budget_seconds, self.max_tool_turns = time_budget_seconds, max_tool_turns
        self.transcribe_batch_size = transcribe_batch_size
        self.last_stats = {}

    def predict_sample(self, sample: dict, runtime: ToolRuntime) -> dict:
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        started = time.perf_counter()
        deadline = started + self.time_budget_seconds

        class Deadline(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return time.perf_counter() >= deadline

        audio_turns = [(f"{sample['id']}_h{i}", turn["audio"])
                       for i, turn in enumerate(sample.get("history", [])) if turn["role"] == "user"]
        audio_turns.append((sample["id"], sample["audio"]))
        paths = [str(runtime.split_dir / relative) for _, relative in audio_turns]
        texts = transcribe_paths(self.asr_model, paths, self.transcribe_batch_size)
        transcripts = {key: text for (key, _), text in zip(audio_turns, texts)}
        schemas = filter_installed_tools(sample, self.tools)
        messages = build_agent_messages(sample, schemas, transcripts)
        tools = chat_tools(schemas)
        context_length = self.options.get("context_length", 12288)
        max_tokens = self.options.get("max_new_tokens", 512)
        self.llm_model.eval()
        for _ in range(self.max_tool_turns):
            if time.perf_counter() >= deadline:
                break
            inputs = self.tokenizer.apply_chat_template(
                messages, tools=tools, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            ).to(self.llm_model.device)
            available = context_length - inputs["input_ids"].shape[1]
            if available < 1:
                raise ValueError(f"Inference context overflow: {sample['id']}")
            with torch.inference_mode():
                generated = self.llm_model.generate(
                    **inputs, max_new_tokens=min(max_tokens, available), do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    stopping_criteria=StoppingCriteriaList([Deadline()]),
                )
            text = self.tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=False)
            if time.perf_counter() >= deadline:
                break
            try:
                calls = parse_tool_calls(text)
            except (ValueError, TypeError, json.JSONDecodeError):
                messages.extend([{"role": "assistant", "content": text},
                                 {"role": "user", "content": "JSON tool call không hợp lệ. Hãy sửa lại."}])
                continue
            if not calls:
                break
            if len(calls) != 1:
                messages.extend([{"role": "assistant", "content": text},
                                 {"role": "user", "content": "Mỗi lượt chỉ gọi một công cụ."}])
                continue
            call = calls[0]
            # validate không thực thi; chỉ env.call mới tạo call trong file nộp.
            checked = runtime.env.validate(call["name"], call["arguments"])
            result = runtime.execute(call["name"], call["arguments"]) if checked["ok"] else checked
            messages.extend([assistant_call(call), {"role": "tool", "name": call["name"],
                                                   "content": json.dumps(result, ensure_ascii=False)}])
        elapsed = time.perf_counter() - started
        self.last_stats = {"id": sample["id"], "elapsed_seconds": elapsed,
                           "timed_out": elapsed >= self.time_budget_seconds}
        return {"id": sample["id"], "tool_calls": runtime.env.get_tool_calls(),
                "state_hash": runtime.env.get_state_hash()}


# 3. Nạp model một lần; ghi file nguyên tử và giữ thống kê thời gian riêng.
def run(cfg: dict) -> None:
    from train.run_train import precision_name

    split = cfg["inference"]["split"]
    directory = Path(cfg["paths"]["processed_dir"])
    environment_split = "train" if split == "val" else split
    split_dir = directory / "raw_extracted" / environment_split
    source = directory / "val_split.jsonl" if split == "val" else split_dir / f"{split}.jsonl"
    samples = load_samples(source)
    root = SplitRoot(split_dir, environment_split)
    tools = catalog_tools(load_catalog(split_dir, environment_split))
    asr = load_asr_model(cfg["asr"]["backbone"], checkpoint=cfg["asr"]["checkpoint"], device="cuda",
                         decoding_strategy=cfg["asr"]["decoding_strategy"])
    llm, tokenizer = load_llm_model(cfg["llm"]["backbone"], options=cfg["llm"],
                                    precision=precision_name(cfg), checkpoint=cfg["llm"]["checkpoint"])
    agent = LifeMateAgent(asr, llm, cfg["inference"]["time_budget_seconds"],
                          tokenizer=tokenizer, tools=tools, options=cfg["llm"],
                          max_tool_turns=cfg["inference"]["max_tool_turns"],
                          transcribe_batch_size=cfg["asr"]["transcribe_batch_size"])
    destination = Path(cfg["inference"]["submission_path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    stats = []
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent,
                                         prefix=".submission-", suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            for sample in samples:
                result = agent.predict_sample(sample, ToolRuntime(split_dir, sample["id"], root))
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                stats.append(agent.last_stats)
        temporary.replace(destination)
        destination.with_suffix(".latency.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"[Inference] {len(samples)} predictions -> {destination}")
