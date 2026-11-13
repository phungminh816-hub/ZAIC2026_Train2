# LifeMate — Parakeet CTC và Qwen3

Pipeline tiếng Việt nhận yêu cầu bằng giọng nói, chọn và thực thi công cụ qua bộ mô phỏng `zaic_tools`. Giữ cách chia mã nguồn `data/`, `train/asr/`, `train/llm/`, `train/inference/`, `metric/`.

## Model và cơ chế huấn luyện

- ASR: [`nvidia/parakeet-ctc-0.6b-Vietnamese`](https://huggingface.co/nvidia/parakeet-ctc-0.6b-Vietnamese), FastConformer–CTC khoảng 0,6 tỷ tham số. Tải đúng `parakeet-ctc-0.6b-vi.nemo`, giữ tokenizer/decoder tiếng Việt, fine-tune bằng NeMo. Profile dùng greedy CTC, không cần tải KenLM/Flashlight.
- LLM: [`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), QLoRA 4-bit NF4, rank 16. Prompt dùng chat template của chính Qwen và schema tool được phép trong từng mẫu.
- SFT tạo một ví dụ cho mỗi bước gọi tool và một bước kết thúc. Loss chỉ tính trên lượt assistant đích; lịch sử, user và kết quả tool được mask bằng `-100`.
- Train/validation chia theo ID mẫu. Transcript ASR có fingerprint checkpoint/manifest; cache thiếu hoặc cũ sẽ báo lỗi. Không tự lấy transcript nhãn để thay cache thiếu.

## Môi trường Vast.ai

Profile mặc định trong `train/config.yaml`: Linux x86_64, **Python 3.11**, **một GPU NVIDIA 24 GB**, **driver host từ 580**, RAM hệ thống nên từ 32 GB. Chương trình kiểm tra tối thiểu 20 GiB VRAM cho profile này; mức sử dụng thực tế còn phụ thuộc độ dài audio/prompt.

Chạy trong thư mục gốc project trên máy Vast.ai. Không dùng môi trường Python 3.14/Windows của máy cá nhân để cài bộ dependency này.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
sudo apt-get update
sudo apt-get install -y build-essential sox libsndfile1 ffmpeg

python -m pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m pip check
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Nếu container đã chạy với quyền root, bỏ `sudo`. Kiểm tra driver bằng `nvidia-smi`: NeMo 2.6 kéo thêm `numba-cuda[cu13]`, nên chọn host có driver **từ 580** dù wheel PyTorch dùng CUDA 12.6. Driver mới hỗ trợ chạy wheel CUDA cũ. Bộ requirements cố định Transformers 4.53.3 theo ràng buộc `~=4.53.0` của NeMo 2.6.

## Dữ liệu và cấu hình

Đặt gói dữ liệu gốc tại `data/raw/train+public_test.zip`, hoặc chuyển nguyên thư mục `data/processed/raw_extracted/` chứa `train/` và `public_test/` lên máy train. Không phát tán bộ dữ liệu.

Luôn chạy lại `prepare` trên Vast.ai để tạo manifest theo đường dẫn Linux, có `audio_filepath`, `duration` và `text` đúng định dạng NeMo. Audio phải là WAV mono 16 kHz; chương trình báo lỗi khi thiếu file, transcript, audio rỗng hoặc sai định dạng.

Các đường dẫn trong YAML được tính từ `paths.root`, tương đối với thư mục chứa config. Checkpoint mặc định:

- ASR: `checkpoints/asr/parakeet_vi.nemo`; Lightning lưu thêm `best-*.ckpt` và `last.ckpt`.
- Qwen adapter/tokenizer: `checkpoints/llm/final/`; Trainer checkpoint nằm trong `checkpoints/llm/trainer/`.
- Cache và dữ liệu SFT: `data/processed/transcripts_*.jsonl`, `sft_*.jsonl`.

## Kiểm tra trước khi thuê máy chạy dài

```bash
# Chỉ kiểm tra cấu hình/dữ liệu; không tải model, không yêu cầu CUDA.
python -m train.run_train --stage check

# Kiểm tra phiên bản dependency, import NeMo, CUDA, GPU và VRAM.
python -m train.run_train --stage check --check-runtime

python -m train.run_train --stage prepare
python -m pytest metric/test -q -p no:cacheprovider

# Huấn luyện thử một optimizer step, lưu riêng trong checkpoints/smoke/.
python -m train.run_train --stage asr --max-steps 1
```

Smoke checkpoint không tự thay checkpoint chính. Sau smoke run, chạy ASR đầy đủ trước khi tạo cache cho Qwen. Smoke run LLM cần dữ liệu SFT đã được xây dựng.

## Huấn luyện

Các flag huấn luyện được đăng ký tại CLI chung. Chạy `python -m train.run_train -h` để xem các nhóm ASR, LLM, dữ liệu và suy luận.

Flag CLI ghi đè YAML cho lần chạy hiện tại, không sửa file YAML. `--print-config` in cấu hình cuối cùng rồi thoát, không tải model hoặc khởi động GPU:

```bash
python -m train.run_train --stage llm --lr 0.00005 --epochs 3 --batch-size 1 --lora-rank 16 --context-length 12288 --print-config
```

Bỏ `--print-config` để chạy stage. Các flag chung `--lr`/`--learning-rate`, `--epochs`, `--batch-size`, `--gradient-accumulation`, `--weight-decay`, `--checkpoint`, `--resume-from-checkpoint` áp dụng cho model được chọn bằng `--stage asr` hoặc `--stage llm`.

Khi chạy `all`, dùng tiền tố để chỉ rõ model:

```bash
python -m train.run_train --stage all --asr-epochs 3 --asr-learning-rate 0.00001 --llm-epochs 2 --llm-learning-rate 0.00005 --print-config
```

Các flag như `--context-length` và `--lora-rank` là tên ngắn của `--llm-context-length` và `--llm-lora-rank`. Flag boolean dùng cặp bật/tắt, ví dụ `--asr-freeze-encoder` / `--no-asr-freeze-encoder`, `--llm-load-in-4bit` / `--no-llm-load-in-4bit`. Đường dẫn CLI tương đối được tính từ `paths.root`.

`--max-steps` giữ chế độ smoke: checkpoint luôn ghi vào `checkpoints/smoke/`, không kết hợp với flag `--checkpoint`. Dùng `--resume-from-checkpoint` để chọn checkpoint đầu vào cần tiếp tục train.

```bash
python -m train.run_train --stage asr
python -m train.run_train --stage asr_cache
python -m train.run_train --stage sft
python -m train.run_train --stage llm --max-steps 1
python -m train.run_train --stage llm
```

Hoặc chạy toàn pipeline, bao gồm chuẩn bị dữ liệu và sinh submission:

```bash
python -m train.run_train --stage all
```

Context mặc định là **12288**. Kiểm tra bằng tokenizer Qwen thật trên transcript tham chiếu cho thấy 4593 ví dụ train và 493 ví dụ validation, dài nhất **9681 token**; mức 8192 không đủ. Transcript ASR và kết quả tool thực tế có thể làm độ dài thay đổi, nên bước SFT vẫn kiểm tra toàn bộ ví dụ trước khi train.

Nếu bị thiếu VRAM, chỉ giảm `llm.context_length` khi dữ liệu vẫn vừa; mã sẽ từ chối ví dụ vượt context thay vì âm thầm cắt mất tool call. Có thể đặt `asr.freeze_encoder: true` để chỉ tinh chỉnh phần decoder. Chạy smoke test trên GPU mục tiêu trước khi train dài.

Tiếp tục lần train bị ngắt bằng `asr.resume_from_checkpoint` trỏ tới Lightning `.ckpt`, hoặc `llm.resume_from_checkpoint` trỏ tới thư mục `checkpoint-*` của Trainer. Đổi checkpoint ASR thì phải tạo lại transcript cache và dữ liệu SFT.

## Suy luận và đánh giá

```bash
python -m train.run_train --stage infer --split val
python -c "from pathlib import Path; from metric.evaluate import evaluate_holdout; evaluate_holdout(Path('data/processed/raw_extracted/train'), Path('outputs/submission.jsonl'))"

python -m train.run_train --stage infer --split public_test
python -c "from pathlib import Path; from metric.evaluate import validate_submission; print(validate_submission(Path('data/processed/raw_extracted/public_test'), Path('outputs/submission.jsonl')))"
```

Mỗi mẫu chỉ tạo một phiên `ToolEnv`; không chạy lại thao tác trong history. File nộp lấy `tool_calls` và `state_hash` từ phiên thực. Thời gian ASR và vòng lặp LLM được ghi riêng tại `outputs/submission.latency.json`. Deadline được kiểm tra trước mỗi bước và sau mỗi token sinh; nó không thể ngắt giữa một CUDA kernel/prefill, nên cần đo thời gian trên GPU mục tiêu.

Metrics local gồm WER chuẩn hóa, exact-call F1 và trạng thái cuối đã canonicalize. Exact-call F1 không thay cho scorer có fuzzy matching của ban tổ chức.

## Nguồn kỹ thuật

- [Parakeet model card và định dạng checkpoint](https://huggingface.co/nvidia/parakeet-ctc-0.6b-Vietnamese/tree/main).
- [NeMo 2.6 fine-tuning API](https://github.com/NVIDIA-NeMo/NeMo/blob/v2.6.0/examples/asr/speech_to_text_finetune.py).
- [Qwen3-4B-Instruct-2507 model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).
- [Transformers 4.53.3 TrainingArguments](https://github.com/huggingface/transformers/blob/v4.53.3/src/transformers/training_args.py).

Kiểm thử logic CPU không thay thế được smoke train/backward trên Vast.ai. Chỉ xác nhận môi trường và khả năng train sau khi các lệnh kiểm tra cùng smoke run trên GPU mục tiêu hoàn tất.
