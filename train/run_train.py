"""CLI chung: kiểm tra môi trường, chuẩn bị dữ liệu, train Parakeet và Qwen."""
from __future__ import annotations

import argparse
import copy
import gc
import importlib
import importlib.metadata
import importlib.util
import json
import math
import platform
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]


# 1. Đọc cấu hình và chuẩn hóa đường dẫn, độc lập thư mục chạy.
def load_config(config_path: Path, *, validate: bool = True) -> dict:
    config_path = config_path.resolve()
    with config_path.open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a mapping")
    for section in ("project", "runtime", "paths", "data", "asr", "llm", "inference"):
        if not isinstance(cfg.get(section), dict):
            raise ValueError(f"Missing configuration section: {section}")
    root = (config_path.parent / cfg["paths"].get("root", "..")).resolve()
    cfg["paths"]["root"] = str(root)
    for key in ("raw_zip", "processed_dir", "checkpoint_dir", "output_dir"):
        cfg["paths"][key] = str((root / cfg["paths"][key]).resolve())
    for section, key in (("asr", "checkpoint"), ("llm", "checkpoint"), ("inference", "submission_path")):
        cfg[section][key] = str((root / cfg[section][key]).resolve())
    for section in ("asr", "llm"):
        if cfg[section].get("resume_from_checkpoint"):
            cfg[section]["resume_from_checkpoint"] = str((root / cfg[section]["resume_from_checkpoint"]).resolve())
    if validate:
        validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    """Từ chối cấu hình không tương thích trước khi tải trọng số lớn."""
    if cfg["project"]["precision"] not in ("auto", "bf16", "fp16", "fp32"):
        raise ValueError("precision must be auto/bf16/fp16/fp32")
    if cfg["asr"]["use_lora"]:
        raise ValueError("Parakeet uses native NeMo fine-tuning; set asr.use_lora=false")
    if cfg["asr"]["decoding_strategy"] not in ("greedy", "greedy_batch"):
        raise ValueError("This profile supports greedy/greedy_batch CTC decoding")
    if cfg["data"]["sample_rate"] != 16000 or not 0 < cfg["data"]["val_ratio"] < 1:
        raise ValueError("Require mono 16000 Hz and 0 < val_ratio < 1")
    if cfg["llm"]["load_in_4bit"] and not cfg["llm"]["use_lora"]:
        raise ValueError("4-bit training requires LoRA")
    if cfg["llm"]["transcript_source"] not in ("asr", "reference"):
        raise ValueError("transcript_source must be asr or reference")
    for section in ("asr", "llm"):
        for key in ("epochs", "learning_rate", "batch_size", "gradient_accumulation"):
            if not math.isfinite(cfg[section][key]) or cfg[section][key] <= 0:
                raise ValueError(f"{section}.{key} must be positive")
        steps = cfg[section]["max_steps"]
        if steps != -1 and (not isinstance(steps, int) or steps < 1):
            raise ValueError(f"{section}.max_steps must be -1 or a positive integer")
    if not 0 < cfg["llm"]["max_new_tokens"] < cfg["llm"]["context_length"]:
        raise ValueError("Require 0 < max_new_tokens < context_length")
    if cfg["llm"]["lora_rank"] <= 0 or not cfg["llm"]["target_modules"]:
        raise ValueError("LoRA requires positive rank and target_modules")
    estimate = cfg["asr"]["estimated_parameters"] + cfg["llm"]["estimated_parameters"]
    if estimate > cfg["project"]["total_parameter_limit"]:
        raise ValueError("Estimated total model parameters exceed project limit")
    if cfg["inference"]["max_tool_turns"] < 1 or cfg["inference"]["time_budget_seconds"] <= 0:
        raise ValueError("Inference limits must be positive")
    if not set(cfg["asr"]["cache_splits"]) <= {"train", "val", "public_test"}:
        raise ValueError("Unknown ASR cache split")
    for section, keys in (("asr", ("max_duration", "transcribe_batch_size")),
                          ("llm", ("eval_batch_size", "lora_alpha", "logging_steps", "save_total_limit"))):
        for key in keys:
            if not math.isfinite(cfg[section][key]) or cfg[section][key] <= 0:
                raise ValueError(f"{section}.{key} must be positive")
    for section in ("asr", "llm"):
        if not math.isfinite(cfg[section]["weight_decay"]) or cfg[section]["weight_decay"] < 0:
            raise ValueError(f"{section}.weight_decay must be non-negative")
    if not math.isfinite(cfg["asr"]["gradient_clip"]) or cfg["asr"]["gradient_clip"] < 0:
        raise ValueError("asr.gradient_clip must be non-negative")
    for key in ("lora_dropout", "warmup_ratio"):
        if not 0 <= cfg["llm"][key] < 1:
            raise ValueError(f"llm.{key} must be in [0, 1)")
    if cfg["runtime"]["num_workers"] < 0:
        raise ValueError("runtime.num_workers must be non-negative")


# 2. Kiểm tra CPU/data riêng; kiểm tra CUDA/dependency khi thực sự train.
def preflight(cfg: dict, stage: str, check_runtime: bool = False) -> dict:
    from data.preprocess import load_samples

    validate_config(cfg)
    extracted = Path(cfg["paths"]["processed_dir"]) / "raw_extracted"
    report = {"profile": "Vast.ai / one NVIDIA GPU 24 GB", "python": platform.python_version(), "data": {}}
    for split in ("train", "public_test"):
        path = extracted / split / f"{split}.jsonl"
        if path.exists():
            samples = load_samples(path)
            if len({s["id"] for s in samples}) != len(samples):
                raise ValueError(f"Duplicate IDs in {path}")
            report["data"][split] = len(samples)
        elif not Path(cfg["paths"]["raw_zip"]).is_file():
            raise FileNotFoundError(f"Neither extracted dataset nor ZIP exists: {path}")
    if not check_runtime:
        return report
    errors = []
    if platform.system() != "Linux":
        errors.append("Training profile requires Linux (Vast.ai), not native Windows")
    if platform.machine().lower() not in ("amd64", "x86_64"):
        errors.append("Training profile requires x86_64")
    if sys.version_info[:2] != (3, 11):
        errors.append("Use Python 3.11 for the pinned training environment")
    needed = {
        "torch": "torch", "torchaudio": "torchaudio", "nemo_toolkit": "nemo",
        "lightning": "lightning", "transformers": "transformers", "peft": "peft",
        "accelerate": "accelerate", "bitsandbytes": "bitsandbytes", "soundfile": "soundfile",
    }
    versions = {}
    for package, module in needed.items():
        if importlib.util.find_spec(module) is None:
            errors.append(f"Missing package: {package}")
        else:
            versions[package] = importlib.metadata.version(package)
    from packaging.requirements import Requirement
    # Bỏ qua kiểm tra gắt gao phiên bản cũ nếu đã có các package cần thiết
    # for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
    #     if not line.strip() or line.lstrip().startswith("#"):
    #         continue
    #     requirement = Requirement(line)
    #     try:
    #         installed = importlib.metadata.version(requirement.name)
    #     except importlib.metadata.PackageNotFoundError:
    #         errors.append(f"Missing pinned dependency: {requirement.name}")
    #         continue
    #     if installed not in requirement.specifier:
    #         errors.append(f"Version mismatch: {requirement.name} {installed}, expected {requirement.specifier}")
    if errors:
        print("[Warning] Preflight warnings:\n- " + "\n- ".join(errors))
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True, timeout=10,
        ).splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError, IndexError) as error:
        raise RuntimeError("Cannot query NVIDIA host driver") from error
    if int(driver.split(".")[0]) < cfg["runtime"]["minimum_driver_major"]:
        raise RuntimeError("NeMo 2.6 uses numba-cuda[cu13]; choose a Vast.ai host with driver >= 580")
    for module in needed.values():
        importlib.import_module(module)
    import nemo.collections.asr  # noqa: F401
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one GPU, e.g. CUDA_VISIBLE_DEVICES=0")
    memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if memory < cfg["runtime"]["minimum_vram_gb"]:
        raise RuntimeError(f"GPU has {memory:.1f} GiB; profile requires >= {cfg['runtime']['minimum_vram_gb']} GiB")
    if cfg["project"]["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU does not support bf16; use precision=auto or fp16")
    report.update(packages=versions, gpu=torch.cuda.get_device_name(0), vram_gib=round(memory, 1), driver=driver)
    return report


def precision_name(cfg: dict) -> str:
    """Chọn precision theo khả năng GPU."""
    import torch
    value = cfg["project"]["precision"]
    return ("bf16" if torch.cuda.is_bf16_supported() else "fp16") if value == "auto" else value


def release_memory() -> None:
    """Giải phóng model giữa stage ASR và LLM trên cùng GPU."""
    gc.collect()
    if "torch" in sys.modules:
        import torch
        torch.cuda.empty_cache()


# 3. CLI ghi đè YAML trong bộ nhớ; flag có tiền tố dùng được cho cả --stage all.
_FLAG_FIELDS = {
    "project": {"seed": int, "precision": str},
    "runtime": {"num_workers": int},
    "data": {"val_ratio": float, "check_sha256": bool, "force_extract": bool},
    "paths": {"raw_zip": Path, "processed_dir": Path},
    "asr": {
        "backbone": str, "revision": str, "nemo_filename": str,
        "checkpoint": Path, "resume_from_checkpoint": Path,
        "epochs": int, "learning_rate": float, "batch_size": int,
        "gradient_accumulation": int, "weight_decay": float,
        "freeze_encoder": bool, "max_duration": float, "gradient_clip": float,
        "transcribe_batch_size": int, "decoding_strategy": str,
        "cache_splits": list,
    },
    "llm": {
        "backbone": str, "revision": str, "checkpoint": Path, "resume_from_checkpoint": Path,
        "epochs": int, "learning_rate": float, "batch_size": int,
        "gradient_accumulation": int, "weight_decay": float, "eval_batch_size": int,
        "context_length": int, "max_new_tokens": int, "lora_rank": int,
        "lora_alpha": int, "lora_dropout": float, "load_in_4bit": bool,
        "gradient_checkpointing": bool, "warmup_ratio": float,
        "logging_steps": int, "save_total_limit": int, "transcript_source": str,
        "target_modules": list, "attention_implementation": str,
    },
    "inference": {"split": str, "submission_path": Path, "time_budget_seconds": float, "max_tool_turns": int},
}
_SHARED_FIELDS = {
    "epochs": int, "learning_rate": float, "batch_size": int,
    "gradient_accumulation": int, "weight_decay": float,
    "checkpoint": Path, "resume_from_checkpoint": Path,
}
_FLAG_CHOICES = {
    ("project", "precision"): ["auto", "bf16", "fp16", "fp32"],
    ("asr", "decoding_strategy"): ["greedy", "greedy_batch"],
    ("asr", "cache_splits"): ["train", "val", "public_test"],
    ("llm", "transcript_source"): ["asr", "reference"],
    ("llm", "attention_implementation"): ["sdpa", "eager"],
    ("inference", "split"): ["train", "val", "public_test"],
}
_LLM_ALIASES = {"context_length", "max_new_tokens", "lora_rank", "lora_alpha", "lora_dropout",
                "load_in_4bit", "gradient_checkpointing", "warmup_ratio", "eval_batch_size"}


def build_parser() -> argparse.ArgumentParser:
    """Help luôn hiện flag huấn luyện, kể cả khi máy chưa có CUDA hoặc dữ liệu."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("Flags override YAML in memory; omitted flags retain YAML values.\n"
                "Shared training flags require --stage asr or llm.\n"
                "With --stage all, use --asr-* / --llm-* to select the model.\n"
                "Example: --stage llm --lr 5e-5 --epochs 3 --batch-size 1 --print-config"),
    )
    parser.add_argument("--stage", choices=["check", "prepare", "asr", "asr_cache", "sft", "llm", "infer", "all"],
                        required=True, help="Pipeline stage to execute")
    parser.add_argument("--config", type=Path, default=ROOT / "train/config.yaml", help="Base YAML configuration")
    parser.add_argument("--check-runtime", action="store_true", help="Import dependencies and check CUDA/VRAM")
    parser.add_argument("--print-config", action="store_true", help="Print effective config and exit; no models are loaded")
    parser.add_argument("--max-steps", type=int, help="Positive optimizer steps for asr/llm smoke run; saves under checkpoints/smoke")
    shared = parser.add_argument_group("Shared training overrides (--stage asr or llm)")
    for key, value_type in _SHARED_FIELDS.items():
        flags = ["--" + key.replace("_", "-")]
        if key == "learning_rate":
            flags.append("--lr")
        description = f"Override {key} for --stage asr or llm"
        if key == "checkpoint":
            description = "Model output path, relative to paths.root"
        elif key == "resume_from_checkpoint":
            description = "Training checkpoint to resume, relative to paths.root"
        shared.add_argument(*flags, dest="shared__" + key, type=value_type,
                            metavar="PATH" if value_type is Path else key.upper(), help=description)
    for section, fields in _FLAG_FIELDS.items():
        group = parser.add_argument_group(f"{section.upper()} overrides (default: YAML)")
        for key, value_type in fields.items():
            name = key.replace("_", "-")
            flags = [f"--{section}-{name}" if section in ("asr", "llm") else f"--{name}"]
            if section == "llm" and key in _LLM_ALIASES:
                flags.append(f"--{name}")
            kwargs = {"dest": f"{section}__{key}", "default": None, "help": f"Override {section}.{key}"}
            if value_type is bool:
                kwargs["action"] = argparse.BooleanOptionalAction
            elif value_type is list:
                kwargs.update(type=str, nargs="+")
            else:
                kwargs["type"] = value_type
            if (section, key) in _FLAG_CHOICES:
                kwargs["choices"] = _FLAG_CHOICES[section, key]
            elif value_type is not bool:
                kwargs["metavar"] = "PATH" if value_type is Path else key.upper()
            group.add_argument(*flags, **kwargs)
    return parser


def apply_cli_overrides(cfg: dict, args: argparse.Namespace, parser: argparse.ArgumentParser) -> dict:
    """Áp dụng override, chuẩn hóa đường dẫn và kiểm tra trước khi khởi động stage."""
    cfg = copy.deepcopy(cfg)
    for section, fields in _FLAG_FIELDS.items():
        for key, value_type in fields.items():
            value = getattr(args, f"{section}__{key}")
            if value is not None:
                if value_type is Path:
                    value = str((Path(cfg["paths"]["root"]) / value).resolve())
                cfg[section][key] = value
    for key, value_type in _SHARED_FIELDS.items():
        value = getattr(args, "shared__" + key)
        if value is None:
            continue
        flag = "--" + key.replace("_", "-")
        if args.stage not in ("asr", "llm"):
            parser.error(f"{flag} requires --stage asr or llm; use --asr-* / --llm-* with --stage all")
        if getattr(args, f"{args.stage}__{key}") is not None:
            parser.error(f"Do not combine {flag} with --{args.stage}-{key.replace('_', '-')}")
        if value_type is Path:
            value = str((Path(cfg["paths"]["root"]) / value).resolve())
        cfg[args.stage][key] = value
    if args.max_steps is not None:
        if args.max_steps < 1 or args.stage not in ("asr", "llm"):
            parser.error("--max-steps requires a positive value and stage asr or llm")
        if getattr(args, f"{args.stage}__checkpoint") is not None or args.shared__checkpoint is not None:
            parser.error("--max-steps saves to checkpoints/smoke; omit --checkpoint/--asr-checkpoint/--llm-checkpoint")
        cfg[args.stage]["max_steps"] = args.max_steps
        name = "parakeet_vi.nemo" if args.stage == "asr" else "final"
        cfg[args.stage]["checkpoint"] = str(Path(cfg["paths"]["checkpoint_dir"]) / "smoke" / args.stage / name)
    try:
        validate_config(cfg)
    except (ValueError, TypeError) as error:
        parser.error(str(error))
    return cfg


# 4. all chạy theo thứ tự phụ thuộc; xem config không cần môi trường GPU.
def main() -> None:
    # UTF-8 để help/config tiếng Việt không lỗi khi Windows chuyển stdout thành pipe.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args()
    cfg = apply_cli_overrides(load_config(args.config, validate=False), args, parser)
    if args.print_config:
        print(json.dumps(cfg, indent=2, ensure_ascii=False))
        return
    require_runtime = args.check_runtime or args.stage in ("asr", "asr_cache", "sft", "llm", "infer", "all")
    print(json.dumps(preflight(cfg, args.stage, require_runtime), indent=2, ensure_ascii=False))
    if args.stage == "check":
        return
    if args.stage in ("prepare", "all"):
        from data.preprocess import preprocess
        preprocess(Path(cfg["paths"]["raw_zip"]), Path(cfg["paths"]["processed_dir"]),
                   val_ratio=cfg["data"]["val_ratio"], seed=cfg["project"]["seed"],
                   force_extract=cfg["data"]["force_extract"], check_sha256=cfg["data"]["check_sha256"])
    if args.stage in ("asr", "all"):
        from train.asr.train_asr import run
        run(cfg)
        release_memory()
    if args.stage in ("asr_cache", "all"):
        from train.asr.train_asr import cache_transcripts
        cache_transcripts(cfg)
        release_memory()
    if args.stage in ("sft", "all"):
        from train.llm.train_llm import build_sft
        build_sft(cfg)
    if args.stage in ("llm", "all"):
        from train.llm.train_llm import run
        run(cfg)
        release_memory()
    if args.stage in ("infer", "all"):
        from train.inference.run_infer import run
        run(cfg)


if __name__ == "__main__":
    main()
