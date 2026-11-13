"""Kiểm thử CPU cho dữ liệu, loss mask, tool protocol và lỗi cấu hình trước khi train."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import wave

import pytest

from data.preprocess import collect_audio_records, load_samples, split_train_val, write_jsonl
from metric.evaluate.metrics import compute_asr_wer, compute_tool_metrics, validate_submission
from train.asr.train_asr import cache_fingerprint, read_cache, transcribe_paths
from train.inference.run_infer import ToolRuntime, parse_tool_calls
from train.llm.train_llm import (
    build_agent_messages, chat_tools, encode_example, reference_transcripts, supervised_examples,
)
from train.run_train import ROOT, apply_cli_overrides, build_parser, load_config, preflight, validate_config


@pytest.fixture
def config():
    return load_config(ROOT / "train/config.yaml")


@pytest.fixture
def sample():
    return {
        "id": "test_1", "audio": "test.wav", "audio_text": "Gọi mẹ",
        "context": {"current_time": "2026-11-13T10:00:00+07:00", "timezone": "Asia/Bangkok",
                    "installed_apps": {"phone": {"tools": ["phone.call"]}}},
        "history": [],
        "label": {"tool_calls": [{"name": "phone.call", "arguments": {"contact_id": "c1"}}],
                  "tool_results": [{"ok": True}], "final_state": {}},
    }


# 1. Kiểm tra cấu hình, đường dẫn và chia tập.
def test_config_uses_parakeet_and_qwen(config):
    assert config["asr"]["backbone"] == "nvidia/parakeet-ctc-0.6b-Vietnamese"
    assert config["llm"]["backbone"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert config["llm"]["load_in_4bit"] and not config["asr"]["use_lora"]
    assert Path(config["paths"]["processed_dir"]) == ROOT / "data/processed"


def test_config_is_independent_of_cwd(config, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert load_config(ROOT / "train/config.yaml") == config


@pytest.mark.parametrize("section,key,value", [
    ("project", "precision", "int8"), ("asr", "use_lora", True),
    ("asr", "decoding_strategy", "flashlight"),
    ("asr", "batch_size", 0), ("llm", "max_steps", 0),
    ("llm", "context_length", 1), ("llm", "transcript_source", "silent-fallback"),
    ("data", "sample_rate", 8000), ("project", "total_parameter_limit", 1),
    ("inference", "max_tool_turns", 0),
])
def test_config_rejects_invalid_values(config, section, key, value):
    config[section][key] = value
    with pytest.raises(ValueError):
        validate_config(config)


@pytest.mark.parametrize("ratio", [0, 1, -0.1, 2])
def test_split_rejects_bad_ratio(ratio):
    with pytest.raises(ValueError):
        split_train_val([{"id": "a"}, {"id": "b"}], val_ratio=ratio)


def test_split_is_reproducible_and_disjoint():
    samples = [{"id": str(i)} for i in range(30)]
    training, validation = split_train_val(samples)
    assert (training, validation) == split_train_val(samples)
    assert {s["id"] for s in training}.isdisjoint(s["id"] for s in validation)
    with pytest.raises(ValueError):
        split_train_val([{"id": "same"}, {"id": "same"}])


# 2. Manifest NeMo phải có duration và đường dẫn tuyệt đối trên máy hiện tại.
def _wav(path, rate=16000, channels=1):
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(b"\0\0" * rate * channels)


def test_audio_manifest_has_nemo_fields(tmp_path):
    _wav(tmp_path / "test.wav")
    rows, missing = collect_audio_records(tmp_path, [{"id": "a", "audio": "test.wav", "audio_text": "Xin chào"}], True)
    assert not missing
    assert rows[0]["duration"] == 1.0
    assert Path(rows[0]["audio_filepath"]).is_absolute()
    assert rows[0]["text"] == "Xin chào"


@pytest.mark.parametrize("rate,channels,text", [(8000, 1, "a"), (16000, 2, "a"), (16000, 1, "")])
def test_bad_audio_or_transcript_is_rejected(tmp_path, rate, channels, text):
    _wav(tmp_path / "test.wav", rate, channels)
    with pytest.raises(ValueError):
        collect_audio_records(tmp_path, [{"id": "a", "audio": "test.wav", "audio_text": text}], True)


def test_missing_audio_is_reported(tmp_path):
    rows, missing = collect_audio_records(tmp_path, [{"id": "a", "audio": "absent.wav"}], False)
    assert not rows and len(missing) == 1


def test_failed_jsonl_write_preserves_original_and_cleans_temporary(tmp_path):
    path = tmp_path / "records.jsonl"
    write_jsonl(path, [{"id": "original"}])
    with pytest.raises(TypeError):
        write_jsonl(path, [{"not_serializable": object()}])
    assert load_samples(path) == [{"id": "original"}]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["records.jsonl"]


def test_cache_fingerprint_changes_with_model_or_manifest(tmp_path):
    checkpoint, manifest = tmp_path / "model.nemo", tmp_path / "manifest.jsonl"
    checkpoint.write_bytes(b"model")
    manifest.write_bytes(b"manifest")
    initial = cache_fingerprint(checkpoint, manifest)
    checkpoint.write_bytes(b"new model")
    assert cache_fingerprint(checkpoint, manifest) != initial


def test_cache_rejects_incomplete_transcripts(config, tmp_path):
    checkpoint = tmp_path / "model.nemo"
    checkpoint.write_bytes(b"weights")
    manifest = tmp_path / "asr_train.jsonl"
    write_jsonl(manifest, [{"turn_id": "a"}, {"turn_id": "b"}])
    path = tmp_path / "transcripts_train.jsonl"
    write_jsonl(path, [{"turn_id": "a", "text": "only one transcript"}])
    path.with_suffix(".meta.json").write_text(json.dumps({
        "fingerprint": cache_fingerprint(checkpoint, manifest), "records": 1,
    }), encoding="utf-8")
    config["paths"]["processed_dir"] = str(tmp_path)
    config["asr"]["checkpoint"] = str(checkpoint)
    with pytest.raises(ValueError, match="Incomplete"):
        read_cache(config, "train")


def test_transcribe_accepts_strings_and_hypotheses():
    class Model:
        def transcribe(self, paths, **kwargs):
            return ["xin chao", SimpleNamespace(text="tam biet")]
    assert transcribe_paths(Model(), ["a", "b"]) == ["xin chao", "tam biet"]
    with pytest.raises(ValueError):
        transcribe_paths(Model(), ["a"])


# 3. Prompt không chứa label; supervision học từng bước, không replay history.
def test_prompt_does_not_expose_label(sample):
    sample["label"]["secret_marker"] = "NEVER_EXPOSE_LABEL"
    messages = build_agent_messages(sample, [], reference_transcripts([sample]))
    assert "NEVER_EXPOSE_LABEL" not in json.dumps(messages)
    assert messages[-1] == {"role": "user", "content": "Gọi mẹ"}


def test_prompt_requires_transcripts_instead_of_label_fallback(sample):
    with pytest.raises(KeyError):
        build_agent_messages(sample, [], {})


def test_history_user_turn_uses_turn_index(sample):
    sample["history"] = [{"role": "user", "audio": "past.wav", "audio_text": "Lời trước"}]
    messages = build_agent_messages(sample, [], reference_transcripts([sample]))
    assert messages[1]["content"] == "Lời trước"
    assert messages[-1]["content"] == "Gọi mẹ"


def test_sft_targets_one_call_then_stop(sample):
    rows = supervised_examples(sample, [], reference_transcripts([sample]))
    assert len(rows) == 2
    assert rows[0]["messages"][-1]["tool_calls"][0]["function"]["name"] == "phone.call"
    assert rows[1]["messages"][-2]["role"] == "tool"
    assert json.loads(rows[1]["messages"][-1]["content"]) == {"done": True}


class TinyTokenizer:
    """Chỉ kiểm tra masking độc lập; không thay kiểm tra tokenizer Qwen thật."""
    def apply_chat_template(self, messages, *, tools, tokenize, add_generation_prompt):
        text = ""
        for message in messages:
            text += "<" + message["role"] + ">" + json.dumps({k: v for k, v in message.items() if k != "role"})
        if add_generation_prompt:
            text += "<assistant>"
        return list(text.encode())


def test_supervision_masks_context_and_rejects_truncation(sample):
    row = supervised_examples(sample, [], reference_transcripts([sample]))[0]
    encoded = encode_example(row, TinyTokenizer(), 20000)
    first_label = next(i for i, value in enumerate(encoded["labels"]) if value != -100)
    assert first_label > 0
    assert encoded["labels"][first_label:] == encoded["input_ids"][first_label:]
    with pytest.raises(ValueError, match="overflow"):
        encode_example(row, TinyTokenizer(), 1)


def test_tools_keep_parameters():
    schema = {"name": "phone.call", "parameters": {"type": "object", "required": ["contact_id"]}}
    assert chat_tools([schema])[0]["function"]["parameters"] == schema["parameters"]


# 4. Tool protocol và metrics phải phát hiện sai ID, hash, JSON.
def test_parse_nested_tool_json():
    text = '<tool_call>{"name":"app.run","arguments":{"nested":{"value":1}}}</tool_call>'
    assert parse_tool_calls(text)[0]["arguments"] == {"nested": {"value": 1}}
    assert parse_tool_calls('{"done":true}') == []
    with pytest.raises(ValueError):
        parse_tool_calls('<tool_call>{"name":1}</tool_call>')
    with pytest.raises(ValueError):
        parse_tool_calls('<tool_call>[]</tool_call>')
    with pytest.raises(ValueError):
        parse_tool_calls('<tool_call>{"name":"unfinished"')


def test_wer_normalizes_text_and_counts_edits():
    assert compute_asr_wer(["Xin chào!"], ["xin chào"]) == 0.0
    assert compute_asr_wer(["a b"], ["a c"]) == 0.5
    with pytest.raises(ValueError):
        compute_asr_wer(["a"], [])


def test_tool_metrics_align_by_id_and_detect_duplicates():
    calls = [{"name": "test.read", "arguments": {}}]
    expected = [{"id": "a", "tool_calls": calls}, {"id": "b", "tool_calls": []}]
    assert compute_tool_metrics(list(reversed(expected)), expected)["exact_call_f1"] == 1.0
    with pytest.raises(ValueError):
        compute_tool_metrics([expected[0], expected[0]], expected)


def test_simulator_submission_replay_and_hash(tmp_path):
    directory = ROOT / "data/processed/raw_extracted/public_test"
    if not directory.is_dir():
        pytest.skip("Official dataset not present")
    first = load_samples(directory / "public_test.jsonl")[0]
    runtime = ToolRuntime(directory, first["id"])
    row = {"id": first["id"], "tool_calls": [], "state_hash": runtime.env.get_state_hash()}
    destination = tmp_path / "submission.jsonl"
    write_jsonl(destination, [row])
    assert validate_submission(directory, destination, {first["id"]})["valid"]
    row["state_hash"] = "invalid"
    write_jsonl(destination, [row])
    with pytest.raises(ValueError, match="hash"):
        validate_submission(directory, destination, {first["id"]})


def test_preflight_checks_dataset_without_importing_torch(config):
    assert preflight(config, "check")["data"]["train"] == 1545


# 5. CLI phải hiện flag và truyền đúng giá trị tới stage, không sửa YAML.
def _cli_config(config, *arguments):
    parser = build_parser()
    args = parser.parse_args(arguments)
    return apply_cli_overrides(config, args, parser)


@pytest.mark.parametrize("entry", [["-m", "train.run_train"], ["train/run_train.py"]])
def test_cli_help_shows_training_flags_even_with_legacy_output_encoding(entry):
    result = subprocess.run(
        [sys.executable, "-B", *entry, "-h"], cwd=ROOT,
        env=dict(os.environ, PYTHONIOENCODING="cp1252", PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True, text=True, encoding="utf-8", timeout=30,
    )
    assert result.returncode == 0, result.stderr
    for option in ("--learning-rate", "--lr", "--epochs", "--batch-size", "--lora-rank",
                   "--context-length", "--asr-learning-rate", "--llm-learning-rate", "--print-config"):
        assert option in result.stdout


def test_cli_defaults_preserve_yaml(config):
    assert _cli_config(config, "--stage", "llm") == config


@pytest.mark.parametrize("stage", ["asr", "llm"])
def test_shared_training_overrides_only_selected_stage(config, stage):
    result = _cli_config(config, "--stage", stage, "--lr", "5e-5", "--epochs", "4",
                         "--batch-size", "2", "--gradient-accumulation", "8")
    assert result[stage]["learning_rate"] == 5e-5
    assert result[stage]["epochs"] == 4
    assert result[stage]["batch_size"] == 2
    assert result[stage]["gradient_accumulation"] == 8
    other = "asr" if stage == "llm" else "llm"
    assert result[other] == config[other]
    assert config[stage]["batch_size"] == 1


def test_stage_all_uses_explicit_model_prefixes(config):
    result = _cli_config(config, "--stage", "all", "--asr-learning-rate", "1e-5",
                         "--llm-learning-rate", "5e-5", "--llm-epochs", "4")
    assert result["asr"]["learning_rate"] == 1e-5
    assert result["llm"]["learning_rate"] == 5e-5
    assert result["llm"]["epochs"] == 4


@pytest.mark.parametrize("arguments", [
    ["--stage", "all", "--lr", "1e-5"],
    ["--stage", "llm", "--epochs", "3", "--llm-epochs", "4"],
    ["--stage", "llm", "--lr", "nan"],
    ["--stage", "asr", "--batch-size", "0"],
    ["--stage", "llm", "--lora-dropout", "1.5"],
    ["--stage", "llm", "--max-steps", "1", "--checkpoint", "custom"],
])
def test_cli_rejects_ambiguous_or_invalid_overrides(config, arguments):
    with pytest.raises(SystemExit) as error:
        _cli_config(config, *arguments)
    assert error.value.code == 2


def test_cli_false_flags_and_list_values_are_applied(config):
    result = _cli_config(config, "--stage", "llm", "--no-llm-gradient-checkpointing",
                         "--no-load-in-4bit", "--llm-target-modules", "q_proj", "v_proj",
                         "--asr-cache-splits", "train", "val")
    assert result["llm"]["gradient_checkpointing"] is False
    assert result["llm"]["load_in_4bit"] is False
    assert result["llm"]["target_modules"] == ["q_proj", "v_proj"]
    assert result["asr"]["cache_splits"] == ["train", "val"]


def test_cli_paths_resolve_from_project_root(config):
    result = _cli_config(config, "--stage", "llm", "--checkpoint", "checkpoints/custom",
                         "--resume-from-checkpoint", "checkpoints/previous",
                         "--submission-path", "outputs/custom.jsonl")
    assert result["llm"]["checkpoint"] == str(ROOT / "checkpoints/custom")
    assert result["llm"]["resume_from_checkpoint"] == str(ROOT / "checkpoints/previous")
    assert result["inference"]["submission_path"] == str(ROOT / "outputs/custom.jsonl")


def test_cli_smoke_keeps_separate_checkpoint(config):
    result = _cli_config(config, "--stage", "llm", "--max-steps", "1")
    assert result["llm"]["max_steps"] == 1
    assert result["llm"]["checkpoint"] == str(ROOT / "checkpoints/smoke/llm/final")


def test_cli_print_config_needs_no_gpu_and_does_not_modify_yaml():
    path = ROOT / "train/config.yaml"
    original = path.read_bytes()
    result = subprocess.run(
        [sys.executable, "-B", "-m", "train.run_train", "--stage", "llm", "--lr", "5e-5",
         "--epochs", "3", "--context-length", "12288", "--print-config"], cwd=ROOT,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True, text=True, encoding="utf-8", timeout=30,
    )
    assert result.returncode == 0, result.stderr
    config = json.loads(result.stdout)
    assert config["llm"]["learning_rate"] == 5e-5
    assert config["llm"]["epochs"] == 3
    assert path.read_bytes() == original


def test_main_passes_cli_overrides_to_training_stage(monkeypatch, capsys):
    import train.run_train as cli
    import train.llm.train_llm as llm
    captured = {}
    monkeypatch.setattr(sys, "argv", ["run_train", "--stage", "llm", "--lr", "5e-5", "--batch-size", "2"])
    monkeypatch.setattr(cli, "preflight", lambda *args: {"runtime": "test"})
    monkeypatch.setattr(cli, "release_memory", lambda: None)
    monkeypatch.setattr(llm, "run", lambda cfg: captured.update(cfg))
    cli.main()
    assert captured["llm"]["learning_rate"] == 5e-5
    assert captured["llm"]["batch_size"] == 2
