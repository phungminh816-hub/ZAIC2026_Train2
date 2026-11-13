"""Kiểm tra dependency và điều kiện chạy LifeMate trước khi train trên Vast.ai.

Chạy từ thư mục gốc project hoặc truyền đường dẫn tuyệt đối tới script:
    python check_constraints.py                         # Kiểm tra môi trường hiện tại.
    python check_constraints.py --resolve               # Thêm pip dry-run trên môi trường hiện tại.
    python check_constraints.py --resolve-only --fresh  # Chỉ giải dependency như khi cài mới.

--resolve/--resolve-only cần mạng hoặc nguồn package nội bộ. Pip có thể tải wheel
lớn hoặc chạy build metadata, nhưng không cài package; file tạm được dọn sau khi chạy.
"""
from __future__ import annotations

import argparse
from collections import deque
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True


def _output_utf8() -> None:
    """Cho phép hiển thị tiếng Việt trên terminal/pipe Windows."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="backslashreplace")
            except (OSError, ValueError):
                pass


class Report:
    """Ghi kết quả và phân biệt lỗi với phần chưa được kiểm tra."""

    def __init__(self) -> None:
        self.errors = 0
        self.warnings = 0

    def ok(self, message: str) -> None:
        print(f"[OK] {message}", flush=True)

    def fail(self, message: str) -> None:
        self.errors += 1
        print(f"[LOI] {message}", flush=True)

    def warn(self, message: str) -> None:
        self.warnings += 1
        print(f"[LUU Y] {message}", flush=True)


def _packaging():
    """Dùng packaging đã cài, hoặc bản đi kèm pip trong môi trường mới."""
    try:
        from packaging.requirements import Requirement
        from packaging.specifiers import SpecifierSet
        from packaging.utils import canonicalize_name
        from packaging.version import Version
    except ImportError:
        try:
            from pip._vendor.packaging.requirements import Requirement
            from pip._vendor.packaging.specifiers import SpecifierSet
            from pip._vendor.packaging.utils import canonicalize_name
            from pip._vendor.packaging.version import Version
        except ImportError as error:
            raise RuntimeError("Cần pip hoặc package 'packaging' trong Python đang chạy script") from error
    return Requirement, SpecifierSet, canonicalize_name, Version


def read_requirements(path: Path, requirement_type) -> list:
    """Đọc các dependency trực tiếp dạng PEP 508 trong requirements.txt."""
    if not path.is_file():
        raise FileNotFoundError(f"Không thấy requirements: {path}")
    result = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = re.split(r"\s+#", raw, maxsplit=1)[0].strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-") or " --hash=" in line:
            raise ValueError(f"Dòng {number} dùng cú pháp pip chưa được checker hỗ trợ: {line}")
        try:
            result.append(requirement_type(line))
        except Exception as error:
            raise ValueError(f"Không đọc được dependency ở dòng {number}: {line}") from error
    if not result:
        raise ValueError(f"Không có dependency trong {path}")
    return result


def check_host(report: Report) -> None:
    """Đối chiếu hệ điều hành và Python với preflight của project."""
    print("\n--- Môi trường Python ---", flush=True)
    print(f"Python: {sys.version.split()[0]} | interpreter: {sys.executable}", flush=True)
    print(f"Hệ điều hành: {platform.system()} | kiến trúc: {platform.machine()}", flush=True)
    if sys.version_info[:2] == (3, 11):
        report.ok("Python 3.11 đúng profile train.")
    else:
        report.fail("Project hiện yêu cầu Python 3.11; hãy kiểm tra venv đang kích hoạt.")
    if platform.system() == "Linux" and platform.machine().lower() in ("x86_64", "amd64"):
        report.ok("Linux x86_64 đúng profile train.")
    else:
        report.fail("Profile train hiện yêu cầu Linux x86_64.")
    try:
        report.ok(f"pip {metadata.version('pip')} trong môi trường Python hiện tại.")
    except metadata.PackageNotFoundError:
        report.fail("Không tìm thấy pip trong môi trường Python hiện tại.")


def check_dependencies(report: Report, requirements: list, packaging_api) -> None:
    """Kiểm tra package đã cài và dependency con, kể cả extra như NeMo[asr]."""
    Requirement, SpecifierSet, canonicalize_name, Version = packaging_api
    print("\n--- Dependency trực tiếp ---", flush=True)
    queue = deque()
    python_version = Version(platform.python_version())
    for requirement in requirements:
        if requirement.marker and not requirement.marker.evaluate():
            print(f"[BO QUA] {requirement}: marker không áp dụng trên máy này", flush=True)
            continue
        queue.append((requirement, "requirements.txt", True))
        try:
            version = metadata.version(requirement.name)
        except metadata.PackageNotFoundError:
            report.fail(f"Thiếu {requirement}; pip chưa cài vào {sys.executable}")
            continue
        if requirement.url:
            report.fail(f"Không thể xác nhận nguồn URL của {requirement} chỉ bằng metadata phiên bản.")
        elif requirement.specifier and not requirement.specifier.contains(version, prereleases=True):
            report.fail(f"{requirement.name}: đang có {version}, cần {requirement.specifier}")
        else:
            report.ok(f"{requirement.name}: {version} thỏa {requirement.specifier or 'yêu cầu'}")

    print("\n--- Dependency con và extras đã cài ---", flush=True)
    visited = set()
    seen_errors = set()
    inspected = 0
    missing_metadata = set()
    while queue:
        requirement, parent, direct = queue.popleft()
        key = (canonicalize_name(requirement.name), tuple(sorted(requirement.extras)))
        try:
            dist = metadata.distribution(requirement.name)
        except metadata.PackageNotFoundError:
            if not direct:
                error_key = (parent, str(requirement))
                if error_key not in seen_errors:
                    seen_errors.add(error_key)
                    report.fail(f"{parent} cần {requirement}, nhưng package chưa được cài.")
            missing_metadata.add(requirement.name)
            continue

        version = dist.version
        if not direct and requirement.specifier and not requirement.specifier.contains(version, prereleases=True):
            error_key = (parent, str(requirement), version)
            if error_key not in seen_errors:
                seen_errors.add(error_key)
                report.fail(f"{parent} cần {requirement}, nhưng đang có {requirement.name} {version}.")
        python_spec = dist.metadata.get("Requires-Python")
        if python_spec:
            try:
                if not SpecifierSet(python_spec).contains(python_version, prereleases=True):
                    error_key = (canonicalize_name(requirement.name), "python", python_spec)
                    if error_key not in seen_errors:
                        seen_errors.add(error_key)
                        report.fail(f"{requirement.name} {version} cần Python {python_spec}.")
            except Exception as error:
                report.fail(f"Không đọc được Requires-Python của {requirement.name}: {error}")
        if key in visited:
            continue
        visited.add(key)
        inspected += 1
        for raw in dist.requires or []:
            try:
                child = Requirement(raw)
                # Marker 'extra' phải được xét cho từng extra của package cha.
                if child.marker and not any(child.marker.evaluate({"extra": extra})
                                            for extra in (set(requirement.extras) | {""})):
                    continue
            except Exception as error:
                report.fail(f"Không đọc được metadata dependency của {requirement.name}: {error}")
                continue
            queue.append((child, f"{requirement.name} {version}", False))
    print(f"Đã kiểm tra metadata của {inspected} tổ hợp package/extras.", flush=True)
    if missing_metadata:
        report.warn("Package chưa cài không có metadata cục bộ để phân tích tiếp; dùng --resolve-only để kiểm tra bộ giải pip.")


def check_pip_environment(report: Report) -> None:
    """pip check bao phủ toàn môi trường, có thể nêu cả package ngoài project."""
    print("\n--- pip check: toàn môi trường hiện tại ---", flush=True)
    try:
        result = subprocess.run([sys.executable, "-B", "-m", "pip", "check"], text=True,
                                encoding="utf-8", errors="replace", capture_output=True, timeout=90)
    except (OSError, subprocess.TimeoutExpired) as error:
        report.fail(f"Không chạy được pip check: {error}")
        return
    output = (result.stdout + result.stderr).strip()
    if output:
        print(output, flush=True)
    if result.returncode == 0:
        report.ok("pip check không phát hiện xung đột package đã cài.")
    else:
        report.fail("pip check phát hiện lỗi trong môi trường; kiểm tra các dòng phía trên.")


def show_cuda_packages() -> None:
    """Hiện phiên bản CUDA/Numba ngay cả khi NeMo chưa được cài thành công."""
    print("\n--- Phiên bản CUDA / Numba trong môi trường ---", flush=True)
    for name in ("numba-cuda", "numba", "cuda-python", "cuda-bindings", "cuda-core",
                 "nvidia-cuda-runtime", "nvidia-cuda-runtime-cu12", "nvidia-cuda-nvrtc",
                 "nvidia-cuda-nvrtc-cu12", "nvidia-cuda-cccl", "nvidia-cublas-cu12"):
        try:
            version = metadata.version(name)
        except metadata.PackageNotFoundError:
            version = "chưa cài"
        print(f"{name}: {version}", flush=True)


def read_runtime_limits(path: Path) -> tuple[float, int, str]:
    """Đọc scalar trong YAML profile, kể cả khi PyYAML chưa được cài."""
    if not path.is_file():
        raise FileNotFoundError(f"Không thấy cấu hình project: {path}")
    try:
        import yaml
    except ImportError:
        # Fallback chỉ nhận dạng key/value đơn giản như config.yaml hiện tại.
        sections: dict[str, dict[str, str]] = {}
        current = None
        for raw in path.read_text(encoding="utf-8").splitlines():
            if re.match(r"^[A-Za-z_][\w-]*:\s*(?:#.*)?$", raw):
                current = raw.split(":", 1)[0]
                sections[current] = {}
            elif current and re.match(r"^\s+[A-Za-z_][\w-]*:\s*", raw):
                key, value = raw.strip().split(":", 1)
                sections[current][key] = value.split("#", 1)[0].strip().strip("\"'")
        try:
            return (float(sections["runtime"]["minimum_vram_gb"]),
                    int(sections["runtime"]["minimum_driver_major"]), sections["project"]["precision"])
        except (KeyError, ValueError) as error:
            raise ValueError("Không đọc được profile YAML; cài PyYAML hoặc dùng config YAML đơn giản") from error
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"Config YAML không hợp lệ: {error}") from error
    return (float(config["runtime"]["minimum_vram_gb"]),
            int(config["runtime"]["minimum_driver_major"]), str(config["project"]["precision"]))


def check_gpu(report: Report, config_path: Path) -> None:
    """Kiểm tra driver/VRAM bằng nvidia-smi và CUDA mà PyTorch nhìn thấy."""
    print("\n--- NVIDIA / CUDA ---", flush=True)
    try:
        min_vram, min_driver, precision = read_runtime_limits(config_path)
    except (OSError, ValueError, TypeError, KeyError) as error:
        report.fail(f"Không đọc được giới hạn runtime: {error}")
        return
    try:
        query = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
                                "--format=csv,noheader,nounits"], capture_output=True,
                               text=True, encoding="utf-8", errors="replace", timeout=20)
    except (OSError, subprocess.TimeoutExpired) as error:
        report.fail(f"Không chạy được nvidia-smi: {error}")
        return
    if query.returncode != 0 or not query.stdout.strip():
        report.fail(f"nvidia-smi không phát hiện GPU: {(query.stderr or query.stdout).strip()}")
        return
    print(query.stdout.strip(), flush=True)
    try:
        gpu_rows = [line.rsplit(",", 2) for line in query.stdout.splitlines() if line.strip()]
        drivers = [int(row[-1].strip().split(".")[0]) for row in gpu_rows]
        memories = [float(row[-2].strip()) / 1024 for row in gpu_rows]
    except (ValueError, IndexError):
        report.fail("Không phân tích được driver/VRAM từ nvidia-smi.")
        return
    if min(drivers) < min_driver:
        report.fail(f"Driver phải từ {min_driver} theo {config_path}.")
    else:
        report.ok(f"Driver đạt ngưỡng >= {min_driver}.")
    if len(memories) == 1:
        if memories[0] < min_vram:
            report.fail(f"GPU có {memories[0]:.1f} GiB; profile yêu cầu >= {min_vram:g} GiB.")
        else:
            report.ok(f"GPU có {memories[0]:.1f} GiB; đạt ngưỡng >= {min_vram:g} GiB.")
    else:
        report.warn("nvidia-smi thấy nhiều GPU; kiểm tra GPU được chọn qua CUDA_VISIBLE_DEVICES/PyTorch.")
    try:
        metadata.version("torch")
    except metadata.PackageNotFoundError:
        report.warn("Chưa có torch: chưa thể kiểm tra CUDA và số GPU PyTorch nhìn thấy.")
        return
    code = (
        "import json, torch\n"
        "available = torch.cuda.is_available()\n"
        "count = torch.cuda.device_count() if available else 0\n"
        "devices = [{'name': torch.cuda.get_device_name(i), "
        "'memory_gib': torch.cuda.get_device_properties(i).total_memory / 1024**3} for i in range(count)]\n"
        "print('LIFEMATE_CUDA=' + json.dumps({'available': available, 'count': count, "
        "'devices': devices, 'torch_cuda': torch.version.cuda, "
        "'bf16': bool(torch.cuda.is_bf16_supported()) if available else False}))\n"
    )
    try:
        result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True,
                                text=True, encoding="utf-8", errors="replace", timeout=60)
        line = next((line for line in result.stdout.splitlines() if line.startswith("LIFEMATE_CUDA=")), None)
        if result.returncode != 0 or line is None:
            raise RuntimeError((result.stderr or result.stdout).strip()[-1500:] or "torch không trả kết quả")
        info = json.loads(line.split("=", 1)[1])
    except (OSError, subprocess.TimeoutExpired, ValueError, RuntimeError) as error:
        report.fail(f"Không kiểm tra được CUDA qua PyTorch: {error}")
        return
    if not info["available"] or not info["torch_cuda"]:
        report.fail("PyTorch không nhìn thấy CUDA; kiểm tra wheel torch/driver.")
    elif info["count"] != 1:
        report.fail(f"PyTorch nhìn thấy {info['count']} GPU; project yêu cầu đúng 1 GPU.")
    elif info["devices"][0]["memory_gib"] < min_vram:
        report.fail(f"GPU PyTorch đang chọn có dưới {min_vram:g} GiB VRAM.")
    elif precision == "bf16" and not info["bf16"]:
        report.fail("Config yêu cầu bf16 nhưng GPU không hỗ trợ.")
    else:
        report.ok(f"PyTorch CUDA {info['torch_cuda']}, một GPU hợp lệ: {info['devices'][0]['name']}.")


def check_imports(report: Report) -> None:
    """Import thật các thư viện chính trong tiến trình con, không nạp model."""
    print("\n--- Import runtime ---", flush=True)
    packages = {"torch": "torch", "torchaudio": "torchaudio", "nemo-toolkit": "nemo.collections.asr",
                "lightning": "lightning", "transformers": "transformers", "peft": "peft",
                "accelerate": "accelerate", "bitsandbytes": "bitsandbytes", "soundfile": "soundfile"}
    modules = []
    for dist, module in packages.items():
        try:
            metadata.version(dist)
        except metadata.PackageNotFoundError:
            report.warn(f"Bỏ qua import {module}: package chưa cài.")
        else:
            modules.append(module)
    if not modules:
        return
    code = (
        "import importlib,json,sys\n"
        "out={}\n"
        "for name in json.loads(sys.argv[1]):\n"
        " try:\n  importlib.import_module(name); out[name]=None\n"
        " except Exception as exc:\n  out[name]=type(exc).__name__+': '+str(exc)\n"
        "print('LIFEMATE_IMPORTS='+json.dumps(out))\n"
    )
    try:
        result = subprocess.run([sys.executable, "-B", "-c", code, json.dumps(modules)], capture_output=True,
                                text=True, encoding="utf-8", errors="replace", timeout=180)
        line = next((line for line in result.stdout.splitlines() if line.startswith("LIFEMATE_IMPORTS=")), None)
        if result.returncode != 0 or line is None:
            raise RuntimeError((result.stderr or result.stdout).strip()[-2000:] or "không có kết quả import")
        results = json.loads(line.split("=", 1)[1])
    except (OSError, subprocess.TimeoutExpired, ValueError, RuntimeError) as error:
        report.fail(f"Tiến trình kiểm tra import thất bại: {error}")
        return
    for module, error in results.items():
        if error is None:
            report.ok(f"Import {module} thành công.")
        else:
            report.fail(f"Import {module} thất bại: {error}")


def _remove_owned_temp(path: Path, expected_parent: Path) -> None:
    """Chỉ xóa thư mục tạm do script tạo sau khi kiểm tra symlink/junction."""
    if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
        raise RuntimeError(f"Từ chối xóa đường dẫn tạm là symlink/junction: {path}")
    if not path.name.startswith("lifemate-constraints-") or path.resolve().parent != expected_parent.resolve():
        raise RuntimeError(f"Thư mục tạm nằm ngoài phạm vi dự kiến: {path}")
    pending = [path]
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                info = entry.stat(follow_symlinks=False)
                if entry.is_symlink() or getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
                    raise RuntimeError(f"Từ chối dọn đệ quy qua symlink/junction: {entry.path}")
                if entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))
    shutil.rmtree(path)


def run_resolver(report: Report, requirements: Path, fresh: bool, timeout: int) -> None:
    """Dùng pip dry-run để thử toàn bộ dependency; không thay package đã cài."""
    print("\n--- pip dependency resolver: dry-run ---", flush=True)
    print("Pip có thể tải wheel/build metadata lớn để giải dependency; không cài package.", flush=True)
    print("Dùng cấu hình/index pip hiện tại" + (
        "; bỏ qua package đã cài (--ignore-installed)." if fresh
        else "; không truyền --ignore-installed, cấu hình pip có thể ảnh hưởng kết quả."
    ), flush=True)
    temp_parent = Path(tempfile.gettempdir()).resolve()
    temporary = Path(tempfile.mkdtemp(prefix="lifemate-constraints-", dir=str(temp_parent)))
    env = os.environ.copy()
    env.update({"TMPDIR": str(temporary), "TMP": str(temporary), "TEMP": str(temporary),
                "PYTHONDONTWRITEBYTECODE": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1"})
    command = [sys.executable, "-B", "-m", "pip", "install", "--dry-run", "--no-cache-dir", "--no-input",
               "--retries", "1", "--timeout", "20", "-r", str(requirements)]
    if fresh:
        command.append("--ignore-installed")
    try:
        try:
            result = subprocess.run(command, env=env, timeout=timeout)
        except subprocess.TimeoutExpired:
            report.fail(f"Pip resolver vượt quá {timeout} giây; tăng --resolver-timeout nếu cần.")
        except OSError as error:
            report.fail(f"Không chạy được pip resolver: {error}")
        else:
            if result.returncode == 0:
                report.ok("Pip đã tìm được phương án dependency cho Python/platform hiện tại.")
                report.warn("Dry-run không xác nhận import runtime, CUDA, model hoặc smoke train.")
            else:
                report.fail(f"Pip resolver thất bại (exit {result.returncode}); xem toàn bộ log phía trên.")
    finally:
        try:
            _remove_owned_temp(temporary, temp_parent)
            print("Đã dọn file tạm của pip resolver.", flush=True)
        except (OSError, RuntimeError) as error:
            report.warn(f"Không thể dọn thư mục tạm {temporary}: {error}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--requirements", type=Path, default=ROOT / "requirements.txt", help="Đường dẫn requirements.txt")
    parser.add_argument("--config", type=Path, default=ROOT / "train/config.yaml", help="Config chứa ngưỡng GPU/driver")
    parser.add_argument("--skip-gpu", action="store_true", help="Bỏ kiểm tra GPU/CUDA; không xác nhận khả năng train")
    parser.add_argument("--skip-imports", action="store_true", help="Bỏ import runtime; chỉ kiểm tra metadata")
    parser.add_argument("--resolve", action="store_true", help="Chạy thêm pip install --dry-run (cần index, có thể tải lớn)")
    parser.add_argument("--resolve-only", action="store_true", help="Chỉ chạy pip resolver; không kiểm tra môi trường/GPU")
    parser.add_argument("--fresh", action="store_true", help="Kết hợp resolver: bỏ qua package đã cài (--ignore-installed)")
    parser.add_argument("--resolver-timeout", type=int, default=600, help="Thời gian tối đa cho pip resolver, giây (mặc định 600)")
    return parser


def main() -> int:
    _output_utf8()
    parser = build_parser()
    args = parser.parse_args()
    if args.fresh and not (args.resolve or args.resolve_only):
        parser.error("--fresh cần --resolve hoặc --resolve-only")
    if args.resolver_timeout <= 0:
        parser.error("--resolver-timeout phải lớn hơn 0")
    requirements_path = args.requirements.resolve()
    report = Report()
    try:
        api = _packaging()
        requirements = read_requirements(requirements_path, api[0])
    except (OSError, ValueError, RuntimeError) as error:
        report.fail(str(error))
        return 1
    print(f"Requirements: {requirements_path}", flush=True)
    print(f"Python đang dùng: {sys.executable} ({platform.python_version()})", flush=True)
    if args.resolve_only:
        report.warn("Chế độ chỉ giải dependency: không kiểm tra runtime, GPU hay khả năng train.")
        if sys.version_info[:2] != (3, 11):
            report.warn("Project yêu cầu Python 3.11; resolver đang dùng phiên bản Python khác.")
    else:
        check_host(report)
        check_dependencies(report, requirements, api)
        show_cuda_packages()
        check_pip_environment(report)
        if args.skip_gpu:
            report.warn("Đã bỏ kiểm tra GPU/CUDA; chưa thể kết luận sẵn sàng train.")
        else:
            check_gpu(report, args.config.resolve())
        if args.skip_imports:
            report.warn("Đã bỏ import runtime; metadata khớp chưa bảo đảm thư viện import được.")
        else:
            check_imports(report)
    if args.resolve or args.resolve_only:
        run_resolver(report, requirements_path, args.fresh, args.resolver_timeout)
    else:
        report.warn("Chưa chạy pip resolver: dùng --resolve hoặc --resolve-only để kiểm tra phương án cài đặt.")
    print(f"\nKết quả: {report.errors} lỗi; {report.warnings} lưu ý.", flush=True)
    if report.errors:
        print("Chưa đạt các kiểm tra đã chọn.", flush=True)
        return 1
    if args.resolve_only:
        print("Pip giải được ràng buộc đã chọn; vẫn cần kiểm tra runtime và smoke train.", flush=True)
    else:
        print("Các kiểm tra đã chọn đạt; vẫn cần smoke train trên GPU mục tiêu.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
