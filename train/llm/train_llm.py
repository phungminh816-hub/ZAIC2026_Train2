"""Qwen3: prompt dùng chung, SFT từng bước gọi tool và huấn luyện QLoRA."""
from __future__ import annotations

from functools import partial
import json
from pathlib import Path
from typing import Any

from data.preprocess import catalog_tools, filter_installed_tools, load_catalog, load_samples, write_jsonl

SYSTEM_PROMPT = (
    "Bạn là LifeMate, trợ lý tiếng Việt. Dùng đúng công cụ được cung cấp; không bịa ID, "
    "kết quả hay thông tin người dùng. Dùng kết quả tool trước khi quyết định bước tiếp theo. "
    "Các thao tác trong lịch sử đã có trong trạng thái hiện tại: không thực hiện lại chúng. "
    "Thời gian tương đối phải dựa vào current_time và timezone trong context. "
    "Nếu thiếu tham số không thể tra cứu, gọi sys.ask_missing_info. "
    "Mỗi lượt chỉ gọi một công cụ. Khi xong hoặc không cần gọi công cụ, trả {\"done\": true}."
)


# 1. Chuyển schema và history sang định dạng chat/tool của Qwen.
def chat_tools(schemas: list[dict]) -> list[dict]:
    return [{"type": "function", "function": {
        "name": schema["name"], "description": schema.get("description", ""),
        "parameters": schema.get("parameters", {"type": "object", "properties": {}}),
    }} for schema in schemas]


def assistant_call(call: dict) -> dict:
    return {"role": "assistant", "content": "", "tool_calls": [
        {"type": "function", "function": {"name": call["name"], "arguments": call["arguments"]}}
    ]}


def build_agent_messages(sample: dict, tools_schema: list[dict], transcripts: dict[str, str]) -> list[dict]:
    """Context chỉ chứa thông tin đầu vào; không đưa label hoặc state ẩn vào prompt."""
    context = json.dumps(sample["context"], ensure_ascii=False, separators=(",", ":"))
    messages = [{"role": "system", "content": SYSTEM_PROMPT + "\nContext: " + context}]
    for index, turn in enumerate(sample.get("history", [])):
        if turn["role"] == "user":
            key = f"{sample['id']}_h{index}"
            if key not in transcripts:
                raise KeyError(f"Missing transcript: {key}")
            messages.append({"role": "user", "content": transcripts[key]})
        elif turn["role"] == "assistant":
            calls = turn.get("tool_calls", [])
            results = turn.get("tool_results", [])
            if calls and len(calls) != len(results):
                raise ValueError(f"History tool/result mismatch: {sample['id']}")
            for call, result in zip(calls, results):
                messages.append(assistant_call(call))
                messages.append({"role": "tool", "name": call["name"], "content": json.dumps(result, ensure_ascii=False)})
            if turn.get("text"):
                messages.append({"role": "assistant", "content": turn["text"]})
    if sample["id"] not in transcripts:
        raise KeyError(f"Missing final transcript: {sample['id']}")
    messages.append({"role": "user", "content": transcripts[sample["id"]]})
    return messages


def reference_transcripts(samples: list[dict]) -> dict[str, str]:
    """Chỉ dùng khi transcript_source=reference được cấu hình rõ ràng."""
    result = {}
    for sample in samples:
        result[sample["id"]] = sample["audio_text"]
        for index, turn in enumerate(sample.get("history", [])):
            if turn["role"] == "user":
                result[f"{sample['id']}_h{index}"] = turn["audio_text"]
    return result


def supervised_examples(sample: dict, schemas: list[dict], transcripts: dict[str, str]) -> list[dict]:
    """Một ví dụ cho mỗi quyết định; loss không học lại system/user/tool hoặc history."""
    prefix = build_agent_messages(sample, schemas, transcripts)
    calls, results = sample["label"]["tool_calls"], sample["label"]["tool_results"]
    if len(calls) != len(results):
        raise ValueError(f"Label tool/result mismatch: {sample['id']}")
    examples = []
    tools = chat_tools(schemas)
    for index, (call, result) in enumerate(zip(calls, results)):
        target = assistant_call(call)
        examples.append({"sample_id": sample["id"], "step": index, "messages": prefix + [target], "tools": tools})
        prefix = prefix + [target, {"role": "tool", "name": call["name"], "content": json.dumps(result, ensure_ascii=False)}]
    examples.append({"sample_id": sample["id"], "step": len(calls), "messages":
                     prefix + [{"role": "assistant", "content": '{"done": true}'}], "tools": tools})
    return examples


# 2. Token hóa cùng chat template ở train/infer; không cắt mất tool call đích.
def encode_example(example: dict, tokenizer: Any, context_length: int) -> dict:
    messages, tools = example["messages"], example["tools"]
    prefix = tokenizer.apply_chat_template(messages[:-1], tools=tools, tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tools=tools, tokenize=True, add_generation_prompt=False)
    if full[:len(prefix)] != prefix:
        raise ValueError(f"Chat template prefix mismatch: {example['sample_id']}")
    if len(full) > context_length:
        raise ValueError(f"SFT context overflow: {example['sample_id']} step={example['step']} "
                         f"needs {len(full)} tokens, limit={context_length}; increase context or explicitly shorten inputs")
    if len(full) <= len(prefix):
        raise ValueError(f"No assistant tokens to supervise: {example['sample_id']}")
    return {"input_ids": full, "attention_mask": [1] * len(full),
            "labels": [-100] * len(prefix) + full[len(prefix):]}


def build_sft(cfg: dict) -> None:
    from transformers import AutoTokenizer
    from train.asr.train_asr import read_cache

    directory = Path(cfg["paths"]["processed_dir"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["llm"]["backbone"], revision=cfg["llm"]["revision"])
    all_tools = catalog_tools(load_catalog(directory / "raw_extracted/train", "train"))
    generated = {}
    for split in ("train", "val"):
        samples = load_samples(directory / f"{split}_split.jsonl")
        transcripts = (read_cache(cfg, split) if cfg["llm"]["transcript_source"] == "asr"
                       else reference_transcripts(samples))
        examples = []
        longest = 0
        for sample in samples:
            schemas = filter_installed_tools(sample, all_tools)
            for example in supervised_examples(sample, schemas, transcripts):
                encoded = encode_example(example, tokenizer, cfg["llm"]["context_length"])
                longest = max(longest, len(encoded["input_ids"]))
                examples.append(example)
        if not examples:
            raise ValueError(f"Empty SFT split: {split}")
        generated[split] = examples
        print(f"[SFT] {split}: {len(samples)} samples -> {len(examples)} decisions; max tokens={longest}")
    if {r["sample_id"] for r in generated["train"]} & {r["sample_id"] for r in generated["val"]}:
        raise ValueError("Train/validation sample leakage")
    for split, examples in generated.items():
        write_jsonl(directory / f"sft_{split}.jsonl", examples)


# 3. Nạp backbone 4-bit; train adapter hoặc nạp adapter đã lưu khi suy luận.
def load_llm_model(
    backbone: str, use_lora: bool = True, lora_rank: int = 16, *,
    options: dict | None = None, precision: str = "bf16",
    training: bool = False, checkpoint: str | None = None,
) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

    options = options or {}
    if checkpoint and not Path(checkpoint, "adapter_config.json").is_file():
        raise FileNotFoundError(f"Missing Qwen adapter: {checkpoint}")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
    kwargs = {
        "torch_dtype": dtype, "device_map": {"": torch.cuda.current_device()},
        "attn_implementation": options.get("attention_implementation", "sdpa"),
        "revision": options.get("revision", "main"),
    }
    quantized = options.get("load_in_4bit", True)
    if quantized:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
        )
    tokenizer_source = checkpoint if checkpoint else backbone
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **({"revision": kwargs["revision"]} if not checkpoint else {}))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" if training else "left"
    model = AutoModelForCausalLM.from_pretrained(backbone, **kwargs)
    if checkpoint:
        model = PeftModel.from_pretrained(model, checkpoint, is_trainable=training)
    elif training and use_lora:
        if quantized:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=options.get("gradient_checkpointing", True))
        model = get_peft_model(model, LoraConfig(
            task_type="CAUSAL_LM", r=lora_rank, lora_alpha=options.get("lora_alpha", 32),
            lora_dropout=options.get("lora_dropout", 0.05), bias="none",
            target_modules=options.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        ))
    elif training and quantized:
        raise ValueError("Cannot train frozen 4-bit weights without LoRA")
    model.config.use_cache = not training
    if training and options.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model, tokenizer


class SFTDataset:
    """Token hóa theo yêu cầu để không giữ toàn bộ token của tập SFT trong RAM."""
    def __init__(self, path: Path, tokenizer: Any, context_length: int) -> None:
        self.rows = load_samples(path)
        if not self.rows:
            raise ValueError(f"Empty SFT file: {path}")
        self.tokenizer, self.context_length = tokenizer, context_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return encode_example(self.rows[index], self.tokenizer, self.context_length)


def collate_examples(batch: list[dict], pad_token_id: int) -> dict:
    """Pad bên phải; chỉ nhãn assistant có loss, token padding luôn là -100."""
    import torch
    length = max(len(row["input_ids"]) for row in batch)
    output = {name: torch.full((len(batch), length), fill, dtype=torch.long)
              for name, fill in (("input_ids", pad_token_id), ("attention_mask", 0), ("labels", -100))}
    for index, row in enumerate(batch):
        for name in output:
            output[name][index, :len(row[name])] = torch.tensor(row[name], dtype=torch.long)
    return output


# 4. Trainer lưu adapter/tokenizer và chọn checkpoint theo validation loss.
def run(cfg: dict) -> None:
    from transformers import Trainer, TrainingArguments, set_seed
    from train.run_train import precision_name

    options = cfg["llm"]
    directory = Path(cfg["paths"]["processed_dir"])
    for split in ("train", "val"):
        if not (directory / f"sft_{split}.jsonl").is_file():
            raise FileNotFoundError(f"Missing sft_{split}.jsonl; run --stage sft")
    set_seed(cfg["project"]["seed"])
    precision = precision_name(cfg)
    model, tokenizer = load_llm_model(options["backbone"], options["use_lora"], options["lora_rank"],
                                     options=options, precision=precision, training=True)
    total = model.get_nb_trainable_parameters()[1] if hasattr(model, "get_nb_trainable_parameters") else model.num_parameters()
    if total + cfg["asr"]["estimated_parameters"] > cfg["project"]["total_parameter_limit"]:
        raise ValueError("Actual LLM + estimated ASR parameters exceed project limit")
    training = SFTDataset(directory / "sft_train.jsonl", tokenizer, options["context_length"])
    validation = SFTDataset(directory / "sft_val.jsonl", tokenizer, options["context_length"])
    # Kiểm tra độ dài toàn bộ dữ liệu trước bước optimizer đầu tiên.
    for dataset in (training, validation):
        for index in range(len(dataset)):
            dataset[index]
    destination = Path(options["checkpoint"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    arguments = TrainingArguments(
        output_dir=str(destination.parent / "trainer"),
        num_train_epochs=options["epochs"], max_steps=options["max_steps"],
        per_device_train_batch_size=options["batch_size"], per_device_eval_batch_size=options["eval_batch_size"],
        gradient_accumulation_steps=options["gradient_accumulation"],
        learning_rate=options["learning_rate"], weight_decay=options["weight_decay"],
        warmup_ratio=options["warmup_ratio"], lr_scheduler_type="cosine",
        bf16=precision == "bf16", fp16=precision == "fp16",
        gradient_checkpointing=options["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False,
        save_total_limit=options["save_total_limit"], logging_steps=options["logging_steps"],
        prediction_loss_only=True, report_to=["tensorboard"],
        dataloader_num_workers=0, remove_unused_columns=False,
        optim="adamw_torch", seed=cfg["project"]["seed"], data_seed=cfg["project"]["seed"],
        label_names=["labels"],
    )
    trainer = Trainer(
        model=model, args=arguments, train_dataset=training, eval_dataset=validation,
        processing_class=tokenizer, data_collator=partial(collate_examples, pad_token_id=tokenizer.pad_token_id),
    )
    trainer.train(resume_from_checkpoint=options["resume_from_checkpoint"])
    trainer.save_model(str(destination))
    tokenizer.save_pretrained(str(destination))
    (destination / "training_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[LLM] Saved {destination} ({total:,} parameters)")
