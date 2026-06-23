# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Equivalence + VRAM check for SAM3DBodyEstimator.process_images.

Runs the same set of frames through the per-frame `process_one_image` and the
batched `process_images`, then asserts the per-person outputs match. This needs
a GPU and the model checkpoint, so it is meant to be run on the target pod, e.g.

    uv run python tools/test_process_images_batching.py \
        --image_folder ./images \
        --checkpoint_path /tmp/models/sam-3d-body-dinov3/model.ckpt \
        --mhr_path /tmp/models/sam-3d-body-dinov3/assets/mhr_model.pt \
        --num 18 --max_batch 8

All frames are treated as coming from one static camera (a single shared
`cam_int`), matching the GPU-worker swing-analyzer use case.
"""
import argparse
import os
from glob import glob

import cv2
import numpy as np
import torch

from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator


def default_cam_int(height: int, width: int) -> torch.Tensor:
    """Replicate prepare_batch's default intrinsics as a (1, 3, 3) tensor."""
    f = (height**2 + width**2) ** 0.5
    return torch.tensor(
        [[[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1]]],
        dtype=torch.float32,
    )


def load_images(image_folder: str, num: int) -> list[np.ndarray]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    paths = sorted(p for ext in exts for p in glob(os.path.join(image_folder, ext)))
    if not paths:
        raise FileNotFoundError(f"No images found in {image_folder}")
    paths = paths[:num]
    # Repeat the available frames if the folder has fewer than requested so the
    # batching / chunking logic is still exercised across many instances.
    while len(paths) < num:
        paths.append(paths[len(paths) % len(paths)])
    imgs = []
    for p in paths:
        bgr = cv2.imread(p)
        imgs.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return imgs


def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model, model_cfg = load_sam_3d_body(
        args.checkpoint_path, device=device, mhr_path=args.mhr_path
    )
    estimator = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=model_cfg)

    imgs = load_images(args.image_folder, args.num)
    height, width = imgs[0].shape[:2]
    # Shared intrinsics for the (assumed) static camera.
    cam_int = default_cam_int(height, width).to(device)

    # One full-frame box per frame (no detector dependency for the test). The same
    # boxes are passed to both code paths so the comparison is apples-to-apples.
    full_frame = np.array([[0.0, 0.0, float(width), float(height)]], dtype=np.float32)
    bboxes_list = [full_frame.copy() for _ in imgs]

    # --- Reference: per-frame ---
    torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
    single = [
        estimator.process_one_image(imgs[i], bboxes=bboxes_list[i], cam_int=cam_int)
        for i in range(len(imgs))
    ]
    single_peak = (
        torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
    )

    # --- Batched ---
    torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
    batched = estimator.process_images(
        imgs, bboxes_list, cam_int=cam_int, max_batch=args.max_batch
    )
    batched_peak = (
        torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
    )

    # --- Compare ---
    assert len(single) == len(batched), (len(single), len(batched))
    fields = ["pred_keypoints_3d", "pred_vertices", "pred_keypoints_2d", "pred_cam_t"]
    max_diffs = {f: 0.0 for f in fields}
    n_people = 0
    for fi, (s, b) in enumerate(zip(single, batched)):
        assert len(s) == len(b), f"frame {fi}: {len(s)} vs {len(b)} people"
        for so, bo in zip(s, b):
            n_people += 1
            for f in fields:
                diff = float(np.abs(np.asarray(so[f]) - np.asarray(bo[f])).max())
                max_diffs[f] = max(max_diffs[f], diff)
                assert np.allclose(
                    so[f], bo[f], atol=args.atol
                ), f"frame {fi} field {f}: max abs diff {diff} > atol {args.atol}"

    print(f"OK: {len(imgs)} frames, {n_people} people compared (atol={args.atol})")
    print("max abs diff per field:")
    for f in fields:
        print(f"  {f:20s} {max_diffs[f]:.3e}")
    if device.type == "cuda":
        print(f"peak VRAM  per-frame: {single_peak:.2f} GB")
        print(f"peak VRAM  batched (max_batch={args.max_batch}): {batched_peak:.2f} GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image_folder", required=True, type=str)
    parser.add_argument("--checkpoint_path", required=True, type=str)
    parser.add_argument(
        "--mhr_path", default=os.environ.get("SAM3D_MHR_PATH", ""), type=str
    )
    parser.add_argument("--num", default=18, type=int, help="frames to test")
    parser.add_argument("--max_batch", default=8, type=int)
    parser.add_argument("--atol", default=1e-3, type=float)
    main(parser.parse_args())
