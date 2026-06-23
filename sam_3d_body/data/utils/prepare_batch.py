# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np
import torch
from torch.utils.data import default_collate


class NoCollate:
    def __init__(self, data):
        self.data = data


def prepare_batch(
    img,
    transform,
    boxes,
    masks=None,
    masks_score=None,
    cam_int=None,
):
    """A helper function to prepare data batch for SAM 3D Body model inference."""
    height, width = img.shape[:2]

    # construct batch data samples
    data_list = []
    for idx in range(boxes.shape[0]):
        data_info = dict(img=img)
        data_info["bbox"] = boxes[idx]  # shape (4,)
        data_info["bbox_format"] = "xyxy"

        if masks is not None:
            data_info["mask"] = masks[idx].copy()
            if masks_score is not None:
                data_info["mask_score"] = masks_score[idx]
            else:
                data_info["mask_score"] = np.array(1.0, dtype=np.float32)
        else:
            data_info["mask"] = np.zeros((height, width, 1), dtype=np.uint8)
            data_info["mask_score"] = np.array(0.0, dtype=np.float32)

        data_list.append(transform(data_info))

    batch = default_collate(data_list)

    max_num_person = batch["img"].shape[0]
    for key in [
        "img",
        "img_size",
        "ori_img_size",
        "bbox_center",
        "bbox_scale",
        "bbox",
        "affine_trans",
        "mask",
        "mask_score",
    ]:
        if key in batch:
            batch[key] = batch[key].unsqueeze(0).float()
    if "mask" in batch:
        batch["mask"] = batch["mask"].unsqueeze(2)
    batch["person_valid"] = torch.ones((1, max_num_person))

    if cam_int is not None:
        batch["cam_int"] = cam_int.to(batch["img"])
    else:
        # Default camera intrinsics according image size
        batch["cam_int"] = torch.tensor(
            [
                [
                    [(height**2 + width**2) ** 0.5, 0, width / 2.0],
                    [0, (height**2 + width**2) ** 0.5, height / 2.0],
                    [0, 0, 1],
                ]
            ],
        ).to(batch["img"])

    batch["img_ori"] = [NoCollate(img)]
    return batch


# Per-instance keys produced by ``prepare_batch`` whose person dimension is axis 1.
PERSON_BATCH_KEYS = [
    "img",
    "img_size",
    "ori_img_size",
    "bbox_center",
    "bbox_scale",
    "bbox",
    "affine_trans",
    "mask",
    "mask_score",
    "person_valid",
]


def concat_person_batches(batches):
    """Concatenate several ``prepare_batch`` outputs along the instance axis.

    Each input is a single-image batch whose per-instance tensors have the person
    dimension at axis 1 (shape ``(1, Ni, ...)``). The crops from different images
    are concatenated into one batch with ``sum(Ni)`` instances so they can be run
    through the model together (e.g. batched multi-frame inference).

    ``cam_int`` is treated as shared and the first batch's value is reused — callers
    that batch across frames pass the same intrinsics to every ``prepare_batch``
    call (static camera). ``img_ori`` lists are concatenated so the per-instance
    original images remain available and aligned with the instance dimension.
    """
    if len(batches) == 1:
        return batches[0]

    out = {}
    for key in PERSON_BATCH_KEYS:
        if key in batches[0]:
            out[key] = torch.cat([b[key] for b in batches], dim=1)

    out["cam_int"] = batches[0]["cam_int"]

    img_ori = []
    for b in batches:
        img_ori.extend(b.get("img_ori", []))
    out["img_ori"] = img_ori

    return out
