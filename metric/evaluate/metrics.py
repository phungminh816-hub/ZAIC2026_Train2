"""WER, exact-call F1 và replay trạng thái; đây là thước đo local, không thay scorer BTC."""
from __future__ import annotations

from collections import Counter
import copy
import json
from pathlib import Path
import unicodedata

from data.preprocess import SplitRoot, find_split, load_samples


# 1. WER chuẩn hóa Unicode, chữ thường, dấu câu và khoảng trắng.
def _words(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text).lower()
    return "".join(" " if unicodedata.category(c).startswith("P") else c for c in text).split()


def compute_asr_wer(references: list[str], hypotheses: list[str]) -> float:
    if len(references) != len(hypotheses):
        raise ValueError("WER requires the same number of references and hypotheses")
    total, errors = 0, 0
    for reference, hypothesis in zip(references, hypotheses):
        expected, actual = _words(reference), _words(hypothesis)
        previous = list(range(len(actual) + 1))
        for i, token in enumerate(expected, 1):
            current = [i]
            for j, predicted in enumerate(actual, 1):
                current.append(min(current[-1] + 1, previous[j] + 1,
                                   previous[j - 1] + (token != predicted)))
            previous = current
        total += len(expected)
        errors += previous[-1]
    return errors / total if total else (0.0 if errors == 0 else float("inf"))


# 2. So sánh exact call; không tự nhận đây là F1 fuzzy của BTC.
def _index(rows: list[dict]) -> dict[str, dict]:
    indexed = {}
    for row in rows:
        if row["id"] in indexed:
            raise ValueError(f"Duplicate prediction ID: {row['id']}")
        indexed[row["id"]] = row
    return indexed


def compute_tool_metrics(predictions: list[dict], targets: list[dict]) -> dict[str, float]:
    predicted, expected = _index(predictions), _index(targets)
    if predicted.keys() != expected.keys():
        raise ValueError("Prediction and target IDs differ")
    hits = predicted_count = expected_count = exact = 0
    for identifier, target in expected.items():
        left, right = predicted[identifier]["tool_calls"], target.get("label", target)["tool_calls"]
        exact += left == right
        key = lambda call: json.dumps(call, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        a, b = Counter(map(key, left)), Counter(map(key, right))
        hits += sum((a & b).values())
        predicted_count += len(left)
        expected_count += len(right)
    precision = hits / predicted_count if predicted_count else float(expected_count == 0)
    recall = hits / expected_count if expected_count else float(predicted_count == 0)
    return {"tool_sequence_exact_match": exact / len(expected) if expected else 0.0,
            "exact_call_precision": precision, "exact_call_recall": recall,
            "exact_call_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


# 3. Replay qua ToolEnv; kiểm tra đầy đủ ID, schema và state_hash.
def validate_submission(data_dir: Path, submission_path: Path, sample_ids: set[str] | None = None) -> dict:
    data_dir = Path(data_dir)
    split, sample_path = find_split(data_dir)
    expected = sample_ids if sample_ids is not None else {s["id"] for s in load_samples(sample_path)}
    rows = _index(load_samples(submission_path))
    if set(rows) != expected:
        raise ValueError(f"Submission IDs differ: missing={len(expected-set(rows))}, extra={len(set(rows)-expected)}")
    root = SplitRoot(data_dir, split)
    for identifier, row in rows.items():
        if not {"id", "tool_calls", "state_hash"} <= row.keys() or not isinstance(row["tool_calls"], list):
            raise ValueError(f"Invalid submission record: {identifier}")
        env = root.env(identifier)
        for call in row["tool_calls"]:
            if not isinstance(call, dict) or not isinstance(call.get("name"), str) or not isinstance(call.get("arguments"), dict):
                raise ValueError(f"Invalid tool call: {identifier}")
            result = env.call(call["name"], call["arguments"])
            if not result.get("ok"):
                raise ValueError(f"Rejected tool call in {identifier}: {result}")
        if row["state_hash"] != env.get_state_hash():
            raise ValueError(f"Replay hash mismatch: {identifier}")
    return {"samples": len(rows), "valid": True}


# 4. Holdout sử dụng danh sách val_split, environment vẫn là train.
def evaluate_holdout(data_dir: Path, pred_path: Path, val_path: Path | None = None) -> dict:
    data_dir = Path(data_dir)
    if val_path is None:
        val_path = data_dir.parent.parent / "val_split.jsonl"
    targets = load_samples(val_path)
    predictions = load_samples(pred_path)
    validate_submission(data_dir, pred_path, {r["id"] for r in targets})
    metrics = compute_tool_metrics(predictions, targets)
    split, _ = find_split(data_dir)
    root = SplitRoot(data_dir, split)
    predicted = _index(predictions)
    matched = 0
    for target in targets:
        env = root.env(target["id"])
        expected = copy.deepcopy(env.get_state())
        expected.update(target["label"]["final_state"])
        for call in predicted[target["id"]]["tool_calls"]:
            env.call(call["name"], call["arguments"])
        matched += root.zaic.canon(env.get_state()) == root.zaic.canon(expected)
    metrics["canonical_final_state_match"] = matched / len(targets) if targets else 0.0
    print(json.dumps(metrics, indent=2))
    return metrics
