#!/usr/bin/env python3
"""Local SAM 2.1 video masks for OOOSplat.

This process never downloads weights and never sends frames anywhere.
White (255) is the subject to keep. Black (0) is background to ignore.
A missing dependency or weight is a non-zero exit, not an empty success.
"""

from __future__ import annotations

import argparse
import json
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

MAX_FRAMES = 400
MAX_EDGE = 4096

# Byte sizes are HTTP Content-Length values read from the official files on
# 2026-11-19. They are not estimates. Parameter counts in the SAM 2 README
# ("Size (M)") are not file sizes.
CHECKPOINTS = {
    "sam2.1_hiera_tiny.pt": {
        "bytes": 156_008_466,
        "config": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
    },
    "sam2.1_hiera_small.pt": {
        "bytes": 184_416_285,
        "config": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
    },
    "sam2.1_hiera_base_plus.pt": {
        "bytes": 323_606_802,
        "config": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
    },
    "sam2.1_hiera_large.pt": {
        "bytes": 898_083_611,
        "config": "configs/sam2.1/sam2.1_hiera_l.yaml",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
    },
}


def emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def fail(message: str, code: str = "subject_mask_failed") -> None:
    emit({"ok": False, "code": code, "message": message})
    raise SystemExit(1)


def preflight(checkpoint: str | None) -> None:
    info: dict = {
        "ok": False,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "torch": False,
        "cuda": False,
        "sam2": False,
        "checkpoint": None,
        "checkpoints": CHECKPOINTS,
        "license": "Apache-2.0",
        "licenseUrl": "https://github.com/facebookresearch/sam2/blob/main/LICENSE",
        "message": "",
    }
    try:
        import torch
    except Exception as error:  # noqa: BLE001 - reported to the desktop app
        info["message"] = f"未安装 PyTorch：{error}"
        emit(info)
        raise SystemExit(2)
    info["torch"] = True
    info["cuda"] = bool(torch.cuda.is_available())
    try:
        import sam2  # noqa: F401
    except Exception as error:  # noqa: BLE001
        info["message"] = f"未安装 SAM 2：{error}"
        emit(info)
        raise SystemExit(2)
    info["sam2"] = True
    if not info["cuda"]:
        info["message"] = "当前 PyTorch 看不到 CUDA。本模式不在 CPU 上假装完成视频分割。"
        emit(info)
        raise SystemExit(2)
    if checkpoint:
        path = Path(checkpoint)
        if not path.is_file():
            info["message"] = "权重文件不存在。本程序不会自动下载。"
            info["checkpoint"] = {"path": str(path), "exists": False}
            emit(info)
            raise SystemExit(2)
        known = CHECKPOINTS.get(path.name)
        info["checkpoint"] = {
            "path": str(path.resolve()),
            "exists": True,
            "bytes": path.stat().st_size,
            "expectedBytes": None if known is None else known["bytes"],
            "knownName": known is not None,
        }
        if known and path.stat().st_size != known["bytes"]:
            info["message"] = "权重文件名匹配官方检查点，但字节数不一致。请重新下载，不要继续。"
            emit(info)
            raise SystemExit(2)
    info["ok"] = True
    info["message"] = "SAM 2 与 CUDA 可用。权重仍由你本机提供，不会上传画面。"
    emit(info)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _frames(directory: Path) -> list[Path]:
    frames = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
    )
    if not frames:
        fail("抽帧目录里没有 JPEG。透明 PNG 和全景不走这条分割。", "no_frames")
    if len(frames) > MAX_FRAMES:
        fail(f"画面超过 {MAX_FRAMES} 张，请降低保留帧率后再分割。", "too_many_frames")
    return frames


def _link_indexed(frames: list[Path], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(frames):
        alias = directory / f"{index:05d}.jpg"
        if alias.exists() or alias.is_symlink():
            alias.unlink()
        try:
            alias.symlink_to(source.resolve())
        except OSError:
            alias.write_bytes(source.read_bytes())


def _save_mask(mask, destination: Path, width: int, height: int) -> None:
    import numpy as np
    from PIL import Image

    array = np.asarray(mask).squeeze()
    if array.ndim != 2:
        fail(f"SAM 2 返回了无法保存的掩码形状 {array.shape}。", "bad_mask")
    image = Image.fromarray((array > 0).astype(np.uint8) * 255, mode="L")
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.NEAREST)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG")


def segment(job_path: Path) -> None:
    try:
        job = json.loads(job_path.read_text(encoding="utf-8"))
    except Exception as error:  # noqa: BLE001
        fail(f"无法读取分割任务：{error}", "bad_job")
    session = Path(job["sessionDir"])
    frames_dir = Path(job["framesDir"])
    masks_dir = Path(job["masksDir"])
    indexed = Path(job["indexedDir"])
    for folder in (frames_dir, masks_dir, indexed):
        if not _inside(folder, session):
            fail("分割输出必须留在本次会话目录内。", "unsafe_path")
    points = job.get("points") or []
    labels = job.get("labels") or []
    box = job.get("box")
    init_mask = job.get("initMask")
    init_path = None
    if points and len(points) != len(labels):
        fail("前景/背景点与标签数量不一致。", "bad_prompt")
    if len(points) > 32:
        fail("提示点超过 32 个。", "bad_prompt")
    if init_mask:
        if points or box is not None:
            fail(
                "自动首帧掩码不能和点或框一起使用。请改用手动 SAM 提示。",
                "bad_prompt",
            )
        init_path = Path(init_mask)
        if not _inside(init_path, session):
            fail("首帧掩码必须留在本次会话目录内。", "unsafe_path")
        if not init_path.is_file():
            fail("找不到 BiRefNet 首帧掩码。没有开始跟踪。", "missing_init_mask")
    elif not points and box is None:
        fail("至少需要一个前景点或一个主体框。", "bad_prompt")
    checkpoint = Path(job["checkpoint"])
    if not checkpoint.is_file():
        fail("找不到 SAM 2.1 权重。本程序不会自动下载。", "missing_checkpoint")
    frames = _frames(frames_dir)
    from PIL import Image

    size = Image.open(frames[0]).size
    if max(size) > MAX_EDGE:
        fail(
            f"抽帧长边 {max(size)} 超过 {MAX_EDGE}。请改用快速或均衡档，避免分割和训练同时占用过多显存。",
            "frame_too_large",
        )
    for frame in frames:
        if Image.open(frame).size != size:
            fail("抽帧尺寸不一致，已停止，没有生成掩码。", "mixed_frame_size")
    _link_indexed(frames, indexed)
    prompt_frame = int(job["frameIndex"])
    if prompt_frame < 0 or prompt_frame >= len(frames):
        fail("提示帧超出抽帧范围。", "bad_prompt")

    import numpy as np
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    dtype = "bf16" if torch.cuda.is_bf16_supported() else "fp32"
    predictor = build_sam2_video_predictor(job["modelCfg"], str(checkpoint.resolve()))
    masks: dict[int, object] = {}
    try:
        context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if dtype == "bf16"
            else torch.autocast("cuda", enabled=False)
        )
        with torch.inference_mode(), context:
            state = predictor.init_state(str(indexed), offload_video_to_cpu=True)
            if init_path is not None:
                loaded = Image.open(init_path).convert("L")
                if loaded.size != size:
                    fail("BiRefNet 首帧掩码与抽帧尺寸不一致。", "bad_mask")
                predictor.add_new_mask(
                    inference_state=state,
                    frame_idx=prompt_frame,
                    obj_id=1,
                    mask=torch.from_numpy(np.array(loaded) >= 128),
                )
            else:
                kwargs = {
                    "inference_state": state,
                    "frame_idx": prompt_frame,
                    "obj_id": 1,
                    "normalize_coords": True,
                }
                if points:
                    kwargs["points"] = np.asarray(points, dtype=np.float32)
                    kwargs["labels"] = np.asarray(labels, dtype=np.int32)
                if box is not None:
                    kwargs["box"] = np.asarray(box, dtype=np.float32)
                predictor.add_new_points_or_box(**kwargs)
            for frame_idx, _obj_ids, video_masks in predictor.propagate_in_video(state):
                masks[int(frame_idx)] = video_masks
            if prompt_frame > 0:
                for frame_idx, _obj_ids, video_masks in predictor.propagate_in_video(
                    state, start_frame_idx=prompt_frame, reverse=True
                ):
                    masks[int(frame_idx)] = video_masks
    finally:
        del predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if len(masks) != len(frames):
        fail(
            f"SAM 2 只返回了 {len(masks)} / {len(frames)} 帧，已丢弃不完整结果。",
            "incomplete_masks",
        )
    written = []
    width, height = size
    for index, source in enumerate(frames):
        mask_name = f"{source.stem}.png"
        _save_mask(masks[index].detach().float().cpu().numpy(), masks_dir / mask_name, width, height)
        written.append({"frame": source.name, "mask": mask_name, "index": index})
    emit(
        {
            "ok": True,
            "code": "segmented",
            "dtype": dtype,
            "width": width,
            "height": height,
            "frames": written,
            "message": "分割完成。掩码尚未送入训练。",
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Local SAM 2.1 video subject masks")
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
        fail(str(error), "subject_mask_failed")


if __name__ == "__main__":
    main()
