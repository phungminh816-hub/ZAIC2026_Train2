"""Chuẩn bị dữ liệu: đọc mẫu, schema tool, môi trường mô phỏng và manifest ASR.

Chạy CLI: python -m data.preprocess. Trạng thái app chỉ đọc qua ToolEnv."""
from __future__ import annotations

import argparse
import glob
import hashlib
import importlib
import json
import os
import random
import sys
import tempfile
import zipfile
import wave
from pathlib import Path
from typing import Any, Dict, List, Tuple


# 1. Đọc mẫu JSONL và xác định split.

def find_split(data_dir: str | Path) -> Tuple[str, Path]:
    """Return (split_name, jsonl_path) for a split directory."""
    data_path = Path(data_dir).resolve()
    candidates = [
        Path(p)
        for p in glob.glob(str(data_path / "*.jsonl"))
        if not Path(p).name.startswith("sample_submission")
    ]
    if not candidates:
        raise FileNotFoundError(f"No <split>.jsonl found in {data_path}")

    for p in candidates:
        if p.stem == data_path.name:
            return data_path.name, p

    chosen = sorted(candidates)[0]
    return chosen.stem, chosen


def load_samples(jsonl_path: str | Path) -> List[Dict[str, Any]]:
    """Load all JSONL records from a split file."""
    with Path(jsonl_path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# 2. Đọc catalog và lọc schema tool được phép cho từng mẫu.

def load_catalog(data_dir: str | Path, split: str) -> Dict[str, Any]:
    """Load <split>_tools_catalog.json."""
    data_path = Path(data_dir)
    catalog_path = data_path / f"{split}_tools_catalog.json"
    if not catalog_path.exists():
        matches = list(data_path.glob("*_tools_catalog.json"))
        if not matches:
            raise FileNotFoundError(f"No *_tools_catalog.json found in {data_path}")
        catalog_path = matches[0]

    with catalog_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def catalog_tools(catalog: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Map tool_name -> tool schema (with 'app' field populated)."""
    tools: Dict[str, Dict[str, Any]] = {}
    for app in catalog.get("apps", []):
        app_id = app.get("id")
        for tool in app.get("tools", []):
            entry = dict(tool)
            entry.setdefault("app", app_id)
            tools[entry["name"]] = entry
    return tools


def filter_installed_tools(
    sample: Dict[str, Any],
    all_tools: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return only the tool schemas allowed for this sample's installed_apps (+ sys)."""
    installed_apps = sample.get("context", {}).get("installed_apps", {})
    allowed_names = set()
    for app_cfg in installed_apps.values():
        for name in app_cfg.get("tools", []):
            allowed_names.add(name)

    selected: List[Dict[str, Any]] = []
    for name, schema in all_tools.items():
        if name in allowed_names or schema.get("app") == "sys":
            selected.append(schema)
    return selected


# 3. Nạp đúng zaic_tools của split và mở phiên mô phỏng cho mẫu.

class SplitRoot:
    """Provides isolated zaic_tools.ToolEnv(split, sample_id, root=...) access."""

    def __init__(self, data_dir: str | Path, split: str) -> None:
        data_path = Path(data_dir).resolve()
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None

        if data_path.name == split:
            self.root = str(data_path.parent)
        else:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="zaic_root_")
            self.root = self._temp_dir.name
            os.symlink(str(data_path), os.path.join(self.root, split))

        self.split = split
        data_str = str(data_path)
        if data_str in sys.path:
            sys.path.remove(data_str)
        sys.path.insert(0, data_str)

        for mod in [m for m in list(sys.modules) if m == "zaic_tools" or m.startswith("zaic_tools.")]:
            del sys.modules[mod]
        self.zaic = importlib.import_module("zaic_tools")

    def env(self, sample_id: str) -> Any:
        return self.zaic.ToolEnv(self.split, sample_id, root=self.root)


# 4. Dataset PyTorch trả về đường dẫn audio và transcript từ manifest.

class ASRManifestDataset:
    """Map-style dataset; DataLoader sử dụng __len__/__getitem__ mà không cần import torch."""

    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path)
        self.records = load_samples(self.manifest_path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.records[idx]


# 5. Giải nén, kiểm tra file audio, chia tập và ghi manifest.

def extract_raw_zip(raw_zip: Path, extract_dir: Path, force: bool = False) -> Path:
    """Extract train+public_test.zip into data/processed/raw_extracted."""
    train_marker = extract_dir / "train" / "train.jsonl"
    pub_marker = extract_dir / "public_test" / "public_test.jsonl"
    if train_marker.exists() and pub_marker.exists() and not force:
        print(f"[Preprocess] Already extracted at {extract_dir}")
        return extract_dir

    if not raw_zip.exists():
        raise FileNotFoundError(f"Raw zip not found: {raw_zip}")

    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Preprocess] Extracting {raw_zip} -> {extract_dir} ...")
    with zipfile.ZipFile(raw_zip, "r") as zf:
        zf.extractall(extract_dir)
    return extract_dir


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_audio_records(
    split_dir: Path,
    samples: List[Dict[str, Any]],
    has_text: bool,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Collect all user audio turns (history + final request) and verify files exist."""
    records: List[Dict[str, Any]] = []
    missing: List[str] = []

    for sample in samples:
        sample_id = sample["id"]
        # History user turns
        for turn_idx, turn in enumerate(sample.get("history", [])):
            if turn.get("role") != "user":
                continue
            rel_audio = turn.get("audio")
            if not rel_audio:
                continue
            abs_audio = (split_dir / rel_audio).resolve()
            if not abs_audio.exists():
                missing.append(str(abs_audio))
                continue
            rec: Dict[str, Any] = {
                "sample_id": sample_id,
                "turn_id": f"{sample_id}_h{turn_idx}",
                "is_final_turn": False,
                "audio_rel": rel_audio,
                "audio_path": abs_audio.as_posix(),
            }
            if has_text:
                rec["text"] = turn.get("audio_text", "")
            records.append(rec)

        # Final user request
        rel_audio = sample["audio"]
        abs_audio = (split_dir / rel_audio).resolve()
        if not abs_audio.exists():
            missing.append(str(abs_audio))
            continue
        rec = {
            "sample_id": sample_id,
            "turn_id": sample_id,
            "is_final_turn": True,
            "audio_rel": rel_audio,
            "audio_path": abs_audio.as_posix(),
        }
        if has_text:
            rec["text"] = sample.get("audio_text", "")
        records.append(rec)

    # NeMo yêu cầu audio_filepath, duration (giây), text; giữ trường cũ cho cache.
    for record in records:
        record["audio_filepath"] = record["audio_path"]
        with wave.open(record["audio_path"], "rb") as audio:
            if audio.getnchannels() != 1 or audio.getframerate() != 16000:
                raise ValueError(f"Expected mono 16000 Hz: {record['audio_path']}")
            if audio.getnframes() <= 0:
                raise ValueError(f"Empty audio: {record['audio_path']}")
            record["duration"] = audio.getnframes() / audio.getframerate()
        if has_text and not record.get("text", "").strip():
            raise ValueError(f"Missing transcript: {record['turn_id']}")
    return records, missing


def split_train_val(
    samples: List[Dict[str, Any]],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Deterministic sample-level train/val split."""
    if not 0 < val_ratio < 1 or len(samples) < 2:
        raise ValueError("Require 0 < val_ratio < 1 and at least two samples")
    ids = [s["id"] for s in samples]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate sample IDs")
    rng = random.Random(seed)
    rng.shuffle(ids)
    val_count = min(len(ids) - 1, max(1, int(len(ids) * val_ratio)))
    val_ids = set(ids[:val_count])

    train_samples = [s for s in samples if s["id"] not in val_ids]
    val_samples = [s for s in samples if s["id"] in val_ids]
    return train_samples, val_samples


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Ghi nguyên tử: giữ file cũ nếu bị lỗi giữa lúc xuất dữ liệu."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".jsonl-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def preprocess(
    raw_zip: Path,
    processed_dir: Path,
    val_ratio: float = 0.1,
    seed: int = 42,
    force_extract: bool = False,
    check_sha256: bool = False,
) -> Dict[str, Any]:
    """Run full data preparation: extract, verify audio, split train/val, write ASR manifests."""
    extract_dir = extract_raw_zip(raw_zip, processed_dir / "raw_extracted", force=force_extract)

    train_dir = extract_dir / "train"
    public_dir = extract_dir / "public_test"

    train_all = load_samples(train_dir / "train.jsonl")
    public_all = load_samples(public_dir / "public_test.jsonl")

    train_split, val_split = split_train_val(train_all, val_ratio=val_ratio, seed=seed)

    asr_train, missing_train = collect_audio_records(train_dir, train_split, has_text=True)
    asr_val, missing_val = collect_audio_records(train_dir, val_split, has_text=True)
    asr_public, missing_public = collect_audio_records(public_dir, public_all, has_text=False)

    missing = missing_train + missing_val + missing_public
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} audio files; first: {missing[0]}")

    if check_sha256:
        val_hashes = {sha256_file(Path(r["audio_path"])) for r in asr_val}
        before = len(asr_train)
        asr_train = [r for r in asr_train if sha256_file(Path(r["audio_path"])) not in val_hashes]
        dedup_removed = before - len(asr_train)
    else:
        dedup_removed = 0

    write_jsonl(processed_dir / "train_split.jsonl", train_split)
    write_jsonl(processed_dir / "val_split.jsonl", val_split)
    write_jsonl(processed_dir / "asr_train.jsonl", asr_train)
    write_jsonl(processed_dir / "asr_val.jsonl", asr_val)
    write_jsonl(processed_dir / "asr_public_test.jsonl", asr_public)

    summary = {
        "train_total_samples": len(train_all),
        "train_split_samples": len(train_split),
        "val_split_samples": len(val_split),
        "public_test_samples": len(public_all),
        "asr_train_clips": len(asr_train),
        "asr_val_clips": len(asr_val),
        "asr_public_clips": len(asr_public),
        "missing_audio_train": len(missing_train),
        "missing_audio_val": len(missing_val),
        "missing_audio_public": len(missing_public),
        "sha256_dedup_removed": dedup_removed,
    }
    with (processed_dir / "dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[Preprocess] Summary:", json.dumps(summary, indent=2))
    return summary


# 6. Tham số dòng lệnh cho bước tiền xử lý.

def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess LifeMate dataset")
    parser.add_argument("--raw-zip", type=Path, default=Path("data/raw/train+public_test.zip"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--check-sha256", action="store_true")
    args = parser.parse_args()

    preprocess(
        raw_zip=args.raw_zip,
        processed_dir=args.processed_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        force_extract=args.force_extract,
        check_sha256=args.check_sha256,
    )


if __name__ == "__main__":
    main()
