"""Parakeet CTC tiếng Việt: fine-tune bằng NeMo và cache transcript theo từng split."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from data.preprocess import load_samples, write_jsonl


# 1. Nạp đúng file .nemo của repository, giữ tokenizer và decoder tiếng Việt.
def load_asr_model(
    backbone: str, use_lora: bool = False, lora_rank: int = 16, *,
    checkpoint: str | None = None, revision: str = "main",
    nemo_filename: str = "parakeet-ctc-0.6b-vi.nemo", device: str = "cpu",
    decoding_strategy: str = "greedy_batch",
) -> Any:
    if use_lora:
        raise ValueError("Parakeet training uses native NeMo, not Whisper/PEFT LoRA")
    from huggingface_hub import hf_hub_download
    from nemo.collections.asr.models import ASRModel
    from omegaconf import OmegaConf, open_dict

    if checkpoint is not None:
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"Missing ASR checkpoint: {path}; run stage asr first")
    else:
        path = Path(hf_hub_download(repo_id=backbone, filename=nemo_filename, revision=revision))
    model = ASRModel.restore_from(restore_path=str(path), map_location=device)
    model.change_decoding_strategy(OmegaConf.create({"strategy": decoding_strategy}), verbose=False)
    return model.to(device)


def _manifest(cfg: dict, split: str) -> Path:
    return Path(cfg["paths"]["processed_dir"]) / f"asr_{'public_test' if split == 'public_test' else split}.jsonl"


def _check_manifest(path: Path, require_text: bool = True) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing manifest: {path}; run --stage prepare")
    records = load_samples(path)
    if not records:
        raise ValueError(f"Empty manifest: {path}")
    for row in records:
        if "duration" not in row or "audio_filepath" not in row:
            raise ValueError(f"Old manifest format in {path}; rerun --stage prepare on this machine")
        if not Path(row["audio_filepath"]).is_file() or row["duration"] <= 0:
            raise ValueError(f"Invalid audio record: {row['turn_id']}")
        if require_text and not row.get("text", "").strip():
            raise ValueError(f"Missing text: {row['turn_id']}")
    return records


# 2. Fine-tune CTC, giữ nguyên vocabulary; checkpoint tốt nhất theo val_wer.
def run(cfg: dict) -> None:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import TensorBoardLogger
    from omegaconf import OmegaConf, open_dict
    from train.run_train import precision_name

    options = cfg["asr"]
    train_path, val_path = _manifest(cfg, "train"), _manifest(cfg, "val")
    training = _check_manifest(train_path)
    validation = _check_manifest(val_path)
    for split, rows in (("train", training), ("val", validation)):
        usable = sum(row["duration"] <= options["max_duration"] for row in rows)
        if usable == 0:
            raise ValueError(f"No {split} audio below max_duration")
        print(f"[ASR] {split}: {usable}/{len(rows)} clips <= {options['max_duration']} s")
    pl.seed_everything(cfg["project"]["seed"], workers=True)
    destination = Path(options["checkpoint"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(
        dirpath=str(destination.parent), filename="best-{epoch:02d}",
        monitor="val_wer", mode="min", save_top_k=1, save_last=True,
    )
    precision = {"bf16": "bf16-mixed", "fp16": "16-mixed", "fp32": "32-true"}[precision_name(cfg)]
    trainer = pl.Trainer(
        accelerator="gpu", devices=1, precision=precision,
        max_epochs=options["epochs"], max_steps=options["max_steps"],
        accumulate_grad_batches=options["gradient_accumulation"],
        gradient_clip_val=options["gradient_clip"],
        callbacks=[checkpoint], logger=TensorBoardLogger(str(destination.parent), name="logs"),
        num_sanity_val_steps=0, log_every_n_steps=10, deterministic=False,
    )
    model = load_asr_model(options["backbone"], revision=options["revision"], nemo_filename=options["nemo_filename"],
                           decoding_strategy=options["decoding_strategy"])
    if model.cfg.preprocessor.sample_rate != cfg["data"]["sample_rate"]:
        raise ValueError("ASR preprocessor sample rate differs from dataset")
    model.set_trainer(trainer)
    count = sum(p.numel() for p in model.parameters())
    if count + cfg["llm"]["estimated_parameters"] > cfg["project"]["total_parameter_limit"]:
        raise ValueError("Actual ASR + estimated LLM parameter count exceeds project limit")
    if options["freeze_encoder"]:
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
    base = {
        "sample_rate": cfg["data"]["sample_rate"], "batch_size": options["batch_size"],
        "num_workers": cfg["runtime"]["num_workers"], "pin_memory": True,
        "use_lhotse": False, "is_tarred": False,
        "max_duration": options["max_duration"],
    }
    with open_dict(model.cfg):
        train_ds = OmegaConf.merge(model.cfg.get("train_ds") or {}, base,
                                   {"manifest_filepath": str(train_path), "shuffle": True})
        val_ds = OmegaConf.merge(model.cfg.get("validation_ds") or {}, base,
                                 {"manifest_filepath": str(val_path), "shuffle": False, "drop_last": False})
    model.setup_training_data(train_ds)
    model.setup_multiple_validation_data(val_ds)
    # Không thay vocabulary: tránh khởi tạo lại decoder/tokenizer đã học tiếng Việt.
    model.setup_optimization(OmegaConf.create({
        "name": "adamw", "lr": options["learning_rate"],
        "weight_decay": options["weight_decay"],
    }))
    trainer.fit(model, ckpt_path=options["resume_from_checkpoint"])
    if checkpoint.best_model_path:
        model = type(model).load_from_checkpoint(checkpoint.best_model_path, map_location="cpu")
    model.save_to(str(destination))
    (destination.parent / "training_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ASR] Saved {destination} ({count:,} parameters)")


# 3. Cache có fingerprint checkpoint và manifest để từ chối transcript lỗi thời.
def cache_fingerprint(checkpoint: Path, manifest: Path) -> str:
    digest = hashlib.sha256()
    for path in (checkpoint, manifest):
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def read_cache(cfg: dict, split: str) -> dict[str, str]:
    directory = Path(cfg["paths"]["processed_dir"])
    path = directory / f"transcripts_{split}.jsonl"
    metadata_path = path.with_suffix(".meta.json")
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Missing ASR cache for {split}; run --stage asr_cache")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = cache_fingerprint(Path(cfg["asr"]["checkpoint"]), _manifest(cfg, split))
    if metadata.get("fingerprint") != expected:
        raise ValueError(f"Stale ASR cache: {path}; rerun --stage asr_cache")
    rows = load_samples(path)
    if len({r["turn_id"] for r in rows}) != len(rows):
        raise ValueError(f"Duplicate turn IDs in {path}")
    expected_ids = {r["turn_id"] for r in load_samples(_manifest(cfg, split))}
    if {r["turn_id"] for r in rows} != expected_ids or metadata.get("records") != len(rows):
        raise ValueError(f"Incomplete ASR cache: {path}")
    return {r["turn_id"]: r["text"] for r in rows}


def transcribe_paths(model: Any, paths: list[str], batch_size: int = 1) -> list[str]:
    """NeMo có thể trả string hoặc Hypothesis; chuẩn hóa thành danh sách string."""
    outputs = model.transcribe(paths, batch_size=batch_size, return_hypotheses=False, num_workers=0, verbose=False)
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    texts = [value if isinstance(value, str) else value.text for value in outputs]
    if len(texts) != len(paths):
        raise ValueError("ASR returned a different number of transcripts")
    return texts


def cache_transcripts(cfg: dict) -> None:
    options = cfg["asr"]
    checkpoint = Path(options["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing ASR checkpoint: {checkpoint}")
    records_by_split = {s: _check_manifest(_manifest(cfg, s), s != "public_test") for s in options["cache_splits"]}
    model = load_asr_model(options["backbone"], checkpoint=str(checkpoint), device="cuda",
                           decoding_strategy=options["decoding_strategy"])
    model.eval()
    for split, records in records_by_split.items():
        path = Path(cfg["paths"]["processed_dir"]) / f"transcripts_{split}.jsonl"
        rows = []
        batch = options["transcribe_batch_size"]
        for start in range(0, len(records), batch):
            portion = records[start:start + batch]
            texts = transcribe_paths(model, [r["audio_filepath"] for r in portion], batch)
            rows.extend({"turn_id": r["turn_id"], "sample_id": r["sample_id"], "text": text}
                        for r, text in zip(portion, texts))
        write_jsonl(path, rows)
        path.with_suffix(".meta.json").write_text(json.dumps({
            "fingerprint": cache_fingerprint(checkpoint, _manifest(cfg, split)), "records": len(rows),
        }), encoding="utf-8")
        print(f"[ASR cache] {split}: {len(rows)} transcripts -> {path}")
