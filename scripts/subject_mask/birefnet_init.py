#!/usr/bin/env python3
"""Official BiRefNet first-frame mask for the SAM 2.1 tracker.

This process never downloads weights and never sends frames anywhere.
It loads only a local ZhengPeng7/BiRefNet snapshot (MIT) through the
official transformers class, with the hub forced offline. The rembg
package is not imported, and this script will not fall back to Bria RMBG.
White (255) is the subject. Black (0) is background.
A missing dependency or weight is a non-zero exit, not an empty success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path


def _force_utf8_stdio() -> None:
    """Windows consoles use a legacy code page. The desktop app reads UTF-8 JSON."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            continue


_force_utf8_stdio()

# Refuse any hub fetch before a later import can touch it.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

# Git LFS oid and size of ZhengPeng7/BiRefNet model.safetensors at
# revision e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4. Not a local estimate.
OFFICIAL_BYTES = 444_473_596
OFFICIAL_SHA256 = "9ab37426bf4de0567af6b5d21b16151357149139362e6e8992021b8ce356a154"
OFFICIAL_URL = "https://huggingface.co/ZhengPeng7/BiRefNet/resolve/main/model.safetensors"
OFFICIAL_REVISION = "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"
# Official inference size from the BiRefNet model card, not a new network.
OFFICIAL_IMAGE_SIZE = (1024, 1024)
CODE_FILES = ("config.json", "birefnet.py", "BiRefNet_config.py")
FORBIDDEN = ("bria", "rmbg", "u2net", "isnet", "silueta", "rembg", "background-removal")
# Same bounds as the Rust subject-mask coverage check.
MIN_COVERAGE = 0.005
MAX_COVERAGE = 0.95


def emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def fail(message: str, code: str = "birefnet_failed") -> None:
    emit({"ok": False, "code": code, "message": message})
    raise SystemExit(1)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _forbidden(path: Path) -> bool:
    lowered = str(path).lower()
    return any(token in lowered for token in FORBIDDEN)


def resolve_snapshot(checkpoint: str) -> tuple[Path, Path]:
    path = Path(checkpoint)
    if _forbidden(path):
        fail(
            "这个路径像是 Bria RMBG、rembg 或其他未授权权重。自动首帧只接受官方 MIT BiRefNet，不会改用 Bria。",
            "forbidden_weights",
        )
    if not path.exists():
        fail("找不到 BiRefNet 本地快照。本程序不会自动下载。", "missing_checkpoint")
    snapshot = path if path.is_dir() else path.parent
    weight = snapshot / "model.safetensors"
    if path.is_file() and path.name != "model.safetensors":
        fail(
            "只接受官方文件名 model.safetensors。请指向 ZhengPeng7/BiRefNet 的本地快照。",
            "unknown_checkpoint",
        )
    if not weight.is_file():
        fail("快照里没有 model.safetensors。本程序不会自动下载。", "missing_checkpoint")
    for name in CODE_FILES:
        if not (snapshot / name).is_file():
            fail(
                f"本地快照缺少官方代码 {name}。请下载完整的 ZhengPeng7/BiRefNet 目录，不要只放权重。本程序不会自动下载。",
                "missing_code",
            )
    try:
        config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    except Exception as error:  # noqa: BLE001
        fail(f"无法读取官方 BiRefNet 配置：{error}", "bad_config")
    if config.get("bb_pretrained") is not False:
        fail(
            "官方配置必须是 bb_pretrained=false，避免下载骨干网络。",
            "bad_config",
        )
    auto_map = config.get("auto_map") or {}
    if auto_map.get("AutoModelForImageSegmentation") != "birefnet.BiRefNet":
        fail("本地代码不是官方 BiRefNet。", "bad_config")
    if config.get("architectures") != ["BiRefNet"]:
        fail("本地配置的模型结构不是 BiRefNet。", "bad_config")
    actual = weight.stat().st_size
    if actual != OFFICIAL_BYTES:
        fail(
            f"model.safetensors 字节数为 {actual}，官方文件是 {OFFICIAL_BYTES}。请重新下载，不要继续。",
            "bad_checkpoint",
        )
    digest = sha256_file(weight)
    if digest != OFFICIAL_SHA256:
        fail(
            "model.safetensors 的 SHA256 与官方 Git LFS 校验不一致。请重新下载，不要继续。",
            "bad_checkpoint",
        )
    return snapshot, weight


def _runtime_failure(message: str) -> None:
    emit(
        {
            "ok": False,
            "code": "missing_runtime",
            "license": "MIT",
            "licenseUrl": "https://github.com/ZhengPeng7/BiRefNet/blob/main/LICENSE",
            "message": message,
        }
    )
    raise SystemExit(2)


def check_runtime() -> None:
    try:
        import torch
    except Exception as error:  # noqa: BLE001
        _runtime_failure(f"未安装 PyTorch：{error}")
    if not torch.cuda.is_available():
        _runtime_failure("当前 PyTorch 看不到 CUDA。自动首帧不会在 CPU 上假装完成。")
    try:
        import transformers  # noqa: F401
    except Exception as error:  # noqa: BLE001
        _runtime_failure(f"未安装 transformers，无法加载官方 BiRefNet 代码：{error}")


def preflight(checkpoint: str | None) -> None:
    info: dict = {
        "ok": False,
        "configured": bool(checkpoint),
        "torch": False,
        "cuda": False,
        "transformers": False,
        "checkpoint": None,
        "bytes": OFFICIAL_BYTES,
        "sha256": OFFICIAL_SHA256,
        "url": OFFICIAL_URL,
        "revision": OFFICIAL_REVISION,
        "license": "MIT",
        "licenseUrl": "https://github.com/ZhengPeng7/BiRefNet/blob/main/LICENSE",
        "message": "",
    }
    if checkpoint:
        try:
            snapshot, weight = resolve_snapshot(checkpoint)
        except SystemExit:
            raise
        info["checkpoint"] = {
            "path": str(weight.resolve()),
            "snapshot": str(snapshot.resolve()),
            "exists": True,
            "bytes": weight.stat().st_size,
            "sha256": OFFICIAL_SHA256,
        }
    try:
        import torch
    except Exception as error:  # noqa: BLE001
        info["message"] = f"未安装 PyTorch：{error}"
        emit(info)
        raise SystemExit(2)
    info["torch"] = True
    info["cuda"] = bool(torch.cuda.is_available())
    if not info["cuda"]:
        info["message"] = "当前 PyTorch 看不到 CUDA。自动首帧不会在 CPU 上假装完成。"
        emit(info)
        raise SystemExit(2)
    try:
        import transformers  # noqa: F401
    except Exception as error:  # noqa: BLE001
        info["message"] = f"未安装 transformers，无法加载官方 BiRefNet 代码：{error}"
        emit(info)
        raise SystemExit(2)
    info["transformers"] = True
    if not checkpoint:
        info["message"] = "未填写 BiRefNet 本地快照。自动首帧不会运行，也不会下载权重。"
        emit(info)
        raise SystemExit(2)
    info["ok"] = True
    info["message"] = "BiRefNet 官方快照与 CUDA 可用。权重仍由你本机提供，不会上传画面。"
    emit(info)


def _save_mask(image, destination: Path) -> float:
    from PIL import Image

    # Official card returns a soft matte. SAM 2 consumes a binary mask and
    # uses 0.5 when it resizes one, so the dichotomous cut is 128.
    binary = image.point(lambda value: 255 if value >= 128 else 0).convert("L")
    destination.parent.mkdir(parents=True, exist_ok=True)
    binary.save(destination, format="PNG")
    histogram = binary.histogram()
    foreground = float(sum(histogram[128:]))
    total = float(binary.size[0] * binary.size[1])
    coverage = 0.0 if total == 0 else foreground / total
    if not (MIN_COVERAGE <= coverage <= MAX_COVERAGE):
        destination.unlink(missing_ok=True)
        fail(
            "BiRefNet 首帧掩码几乎为空或几乎覆盖整帧。请改用点或框。没有把背景涂黑，也没有开始跟踪。",
            "bad_coverage",
        )
    return coverage


def segment(job_path: Path) -> None:
    try:
        job = json.loads(job_path.read_text(encoding="utf-8"))
    except Exception as error:  # noqa: BLE001
        fail(f"无法读取 BiRefNet 任务：{error}", "bad_job")
    session = Path(job["sessionDir"])
    frame = Path(job["frame"])
    output = Path(job["output"])
    if not _inside(frame, session / "frames"):
        fail("BiRefNet 只能读取本次会话的抽帧。", "unsafe_path")
    if frame.suffix.lower() not in {".jpg", ".jpeg"}:
        fail("BiRefNet 只接受本次会话里的 JPEG 抽帧。", "unsafe_path")
    if not _inside(output, session) or output.resolve() == frame.resolve():
        fail("首帧掩码必须写在会话目录内，不能写回抽帧。", "unsafe_path")
    snapshot, weight = resolve_snapshot(job["checkpoint"])
    check_runtime()
    import torch
    from PIL import Image
    from torchvision import transforms
    from transformers import AutoModelForImageSegmentation

    image = Image.open(frame).convert("RGB")
    # Same preprocessing as the official BiRefNet model card.
    transform = transforms.Compose(
        [
            transforms.Resize(OFFICIAL_IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    model = AutoModelForImageSegmentation.from_pretrained(
        str(snapshot.resolve()),
        trust_remote_code=True,
        local_files_only=True,
    )
    coverage = 0.0
    try:
        torch.set_float32_matmul_precision("high")
        model.to("cuda")
        model.eval()
        model.half()
        batch = transform(image).unsqueeze(0).to("cuda").half()
        with torch.no_grad():
            prediction = model(batch)[-1].sigmoid().cpu()
        soft = transforms.ToPILImage()(prediction[0].squeeze())
        mask = soft.resize(image.size)
        coverage = _save_mask(mask, output)
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    emit(
        {
            "ok": True,
            "code": "initialized",
            "width": image.size[0],
            "height": image.size[1],
            "coverage": coverage,
            "sha256": sha256_file(weight),
            "bytes": weight.stat().st_size,
            "message": "BiRefNet 首帧掩码已写好。尚未跟踪后续帧，也没有开始训练。",
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Local BiRefNet first-frame mask")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--job", default=None)
    args = parser.parse_args()
    try:
        if args.preflight:
            preflight(args.checkpoint)
            return
        if not args.job:
            fail("缺少 --job。", "bad_job")
        segment(Path(args.job))
    except SystemExit:
        raise
    except Exception as error:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        fail(str(error), "birefnet_failed")


if __name__ == "__main__":
    main()
