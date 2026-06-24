# Copyright (c) Meta Platforms, Inc. and affiliates.
from typing import Optional, Union

import cv2

import numpy as np
import torch

from sam_3d_body.data.transforms import (
    Compose,
    GetBBoxCenterScale,
    TopdownAffine,
    VisionTransformWrapper,
)

from sam_3d_body.data.utils.io import load_image
from sam_3d_body.data.utils.prepare_batch import (
    concat_person_batches,
    PERSON_BATCH_KEYS,
    prepare_batch,
)
from sam_3d_body.utils import recursive_to
from torchvision.transforms import ToTensor


class SAM3DBodyEstimator:
    def __init__(
        self,
        sam_3d_body_model,
        model_cfg,
        human_detector=None,
        human_segmentor=None,
        fov_estimator=None,
    ):
        self.device = sam_3d_body_model.device
        self.model, self.cfg = sam_3d_body_model, model_cfg
        self.detector = human_detector
        self.sam = human_segmentor
        self.fov_estimator = fov_estimator
        self.thresh_wrist_angle = 1.4

        # For mesh visualization
        self.faces = self.model.head_pose.faces.cpu().numpy()

        if self.detector is None:
            print("No human detector is used...")
        if self.sam is None:
            print("Mask-condition inference is not supported...")
        if self.fov_estimator is None:
            print("No FOV estimator... Using the default FOV!")

        self.transform = Compose(
            [
                GetBBoxCenterScale(),
                TopdownAffine(input_size=self.cfg.MODEL.IMAGE_SIZE, use_udp=False),
                VisionTransformWrapper(ToTensor()),
            ]
        )
        self.transform_hand = Compose(
            [
                GetBBoxCenterScale(padding=0.9),
                TopdownAffine(input_size=self.cfg.MODEL.IMAGE_SIZE, use_udp=False),
                VisionTransformWrapper(ToTensor()),
            ]
        )

    @torch.no_grad()
    def process_one_image(
        self,
        img: Union[str, np.ndarray],
        bboxes: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        cam_int: Optional[np.ndarray] = None,
        det_cat_id: int = 0,
        bbox_thr: float = 0.5,
        nms_thr: float = 0.3,
        use_mask: bool = False,
        inference_type: str = "full",
    ):
        """
        Perform model prediction in top-down format: assuming input is a full image.

        Args:
            img: Input image (path or numpy array)
            bboxes: Optional pre-computed bounding boxes
            masks: Optional pre-computed masks (numpy array). If provided, SAM2 will be skipped.
            det_cat_id: Detection category ID
            bbox_thr: Bounding box threshold
            nms_thr: NMS threshold
            inference_type:
                - full: full-body inference with both body and hand decoders
                - body: inference with body decoder only (still full-body output)
                - hand: inference with hand decoder only (only hand output)
        """

        # clear all cached results
        self.batch = None
        self.image_embeddings = None
        self.output = None
        self.prev_prompt = []
        torch.cuda.empty_cache()

        if type(img) == str:
            img = load_image(img, backend="cv2", image_format="bgr")
            image_format = "bgr"
        else:
            print("####### Please make sure the input image is in RGB format")
            image_format = "rgb"
        height, width = img.shape[:2]

        if bboxes is not None:
            boxes = bboxes.reshape(-1, 4)
            self.is_crop = True
        elif self.detector is not None:
            if image_format == "rgb":
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                image_format = "bgr"
            print("Running object detector...")
            boxes = self.detector.run_human_detection(
                img,
                det_cat_id=det_cat_id,
                bbox_thr=bbox_thr,
                nms_thr=nms_thr,
                default_to_full_image=False,
            )
            print("Found boxes:", boxes)
            self.is_crop = True
        else:
            boxes = np.array([0, 0, width, height]).reshape(1, 4)
            self.is_crop = False

        # If there are no detected humans, don't run prediction
        if len(boxes) == 0:
            return []

        # The following models expect RGB images instead of BGR
        if image_format == "bgr":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Handle masks - either provided externally or generated via SAM2
        masks_score = None
        if masks is not None:
            # Use provided masks - ensure they match the number of detected boxes
            print(f"Using provided masks: {masks.shape}")
            assert (
                bboxes is not None
            ), "Mask-conditioned inference requires bboxes input!"
            masks = masks.reshape(-1, height, width, 1).astype(np.uint8)
            masks_score = np.ones(
                len(masks), dtype=np.float32
            )  # Set high confidence for provided masks
            use_mask = True
        elif use_mask and self.sam is not None:
            print("Running SAM to get mask from bbox...")
            # Generate masks using SAM2
            masks, masks_score = self.sam.run_sam(img, boxes)
        else:
            masks, masks_score = None, None

        #################### Construct batch data samples ####################
        batch = prepare_batch(img, self.transform, boxes, masks, masks_score)

        #################### Run model inference on an image ####################
        batch = recursive_to(batch, "cuda")
        self.model._initialize_batch(batch)

        # Handle camera intrinsics
        # - either provided externally or generated via default FOV estimator
        if cam_int is not None:
            print("Using provided camera intrinsics...")
            cam_int = cam_int.to(batch["img"])
            batch["cam_int"] = cam_int.clone()
        elif self.fov_estimator is not None:
            print("Running FOV estimator ...")
            input_image = batch["img_ori"][0].data
            cam_int = self.fov_estimator.get_cam_intrinsics(input_image).to(
                batch["img"]
            )
            batch["cam_int"] = cam_int.clone()
        else:
            cam_int = batch["cam_int"].clone()

        outputs = self.model.run_inference(
            img,
            batch,
            inference_type=inference_type,
            transform_hand=self.transform_hand,
            thresh_wrist_angle=self.thresh_wrist_angle,
        )
        if inference_type == "full":
            pose_output, batch_lhand, batch_rhand, _, _ = outputs
        else:
            pose_output = outputs

        out = pose_output["mhr"]
        out = recursive_to(out, "cpu")
        out = recursive_to(out, "numpy")
        all_out = []
        for idx in range(batch["img"].shape[1]):
            all_out.append(
                {
                    "bbox": batch["bbox"][0, idx].cpu().numpy(),
                    "focal_length": out["focal_length"][idx],
                    "pred_keypoints_3d": out["pred_keypoints_3d"][idx],
                    "pred_keypoints_2d": out["pred_keypoints_2d"][idx],
                    "pred_vertices": out["pred_vertices"][idx],
                    "pred_cam_t": out["pred_cam_t"][idx],
                    "pred_pose_raw": out["pred_pose_raw"][idx],
                    "global_rot": out["global_rot"][idx],
                    "body_pose_params": out["body_pose"][idx],
                    "hand_pose_params": out["hand"][idx],
                    "scale_params": out["scale"][idx],
                    "shape_params": out["shape"][idx],
                    "expr_params": out["face"][idx],
                    "mask": masks[idx] if masks is not None else None,
                    "pred_joint_coords": out["pred_joint_coords"][idx],
                    "pred_global_rots": out["joint_global_rots"][idx],
                    "mhr_model_params": out["mhr_model_params"][idx],
                }
            )

            if inference_type == "full":
                all_out[-1]["lhand_bbox"] = np.array(
                    [
                        (
                            batch_lhand["bbox_center"].flatten(0, 1)[idx][0]
                            - batch_lhand["bbox_scale"].flatten(0, 1)[idx][0] / 2
                        ).item(),
                        (
                            batch_lhand["bbox_center"].flatten(0, 1)[idx][1]
                            - batch_lhand["bbox_scale"].flatten(0, 1)[idx][1] / 2
                        ).item(),
                        (
                            batch_lhand["bbox_center"].flatten(0, 1)[idx][0]
                            + batch_lhand["bbox_scale"].flatten(0, 1)[idx][0] / 2
                        ).item(),
                        (
                            batch_lhand["bbox_center"].flatten(0, 1)[idx][1]
                            + batch_lhand["bbox_scale"].flatten(0, 1)[idx][1] / 2
                        ).item(),
                    ]
                )
                all_out[-1]["rhand_bbox"] = np.array(
                    [
                        (
                            batch_rhand["bbox_center"].flatten(0, 1)[idx][0]
                            - batch_rhand["bbox_scale"].flatten(0, 1)[idx][0] / 2
                        ).item(),
                        (
                            batch_rhand["bbox_center"].flatten(0, 1)[idx][1]
                            - batch_rhand["bbox_scale"].flatten(0, 1)[idx][1] / 2
                        ).item(),
                        (
                            batch_rhand["bbox_center"].flatten(0, 1)[idx][0]
                            + batch_rhand["bbox_scale"].flatten(0, 1)[idx][0] / 2
                        ).item(),
                        (
                            batch_rhand["bbox_center"].flatten(0, 1)[idx][1]
                            + batch_rhand["bbox_scale"].flatten(0, 1)[idx][1] / 2
                        ).item(),
                    ]
                )

        return all_out

    @staticmethod
    def _frames_in_slice(imgs, counts, start, end):
        """Frames (and per-frame instance counts) covered by the global instance
        slice ``[start, end)``, in frame order.

        Used when a VRAM chunk spans multiple frames so each instance's hands can
        be re-cropped from the correct frame. Returns ``(chunk_imgs, chunk_counts)``
        with ``sum(chunk_counts) == end - start``.
        """
        chunk_imgs = []
        chunk_counts = []
        pos = 0
        for img_i, count in zip(imgs, counts):
            f0, f1 = pos, pos + count
            pos = f1
            lo = max(f0, start)
            hi = min(f1, end)
            if hi > lo:
                chunk_imgs.append(img_i)
                chunk_counts.append(hi - lo)
        return chunk_imgs, chunk_counts

    def _build_person_outputs(
        self, out, batch, inference_type, batch_lhand=None, batch_rhand=None
    ):
        """Assemble per-person output dicts for one (chunk) batch.

        This mirrors the per-instance assembly loop in ``process_one_image`` so the
        batched ``process_images`` returns dicts with an identical schema. Masks are
        not used in the batched path, so ``mask`` is always ``None``. Keep this in
        sync with ``process_one_image`` if its output schema changes.
        """
        all_out = []
        for idx in range(batch["img"].shape[1]):
            all_out.append(
                {
                    "bbox": batch["bbox"][0, idx].cpu().numpy(),
                    "focal_length": out["focal_length"][idx],
                    "pred_keypoints_3d": out["pred_keypoints_3d"][idx],
                    "pred_keypoints_2d": out["pred_keypoints_2d"][idx],
                    "pred_vertices": out["pred_vertices"][idx],
                    "pred_cam_t": out["pred_cam_t"][idx],
                    "pred_pose_raw": out["pred_pose_raw"][idx],
                    "global_rot": out["global_rot"][idx],
                    "body_pose_params": out["body_pose"][idx],
                    "hand_pose_params": out["hand"][idx],
                    "scale_params": out["scale"][idx],
                    "shape_params": out["shape"][idx],
                    "expr_params": out["face"][idx],
                    "mask": None,
                    "pred_joint_coords": out["pred_joint_coords"][idx],
                    "pred_global_rots": out["joint_global_rots"][idx],
                    "mhr_model_params": out["mhr_model_params"][idx],
                }
            )

            if inference_type == "full":
                all_out[-1]["lhand_bbox"] = np.array(
                    [
                        (
                            batch_lhand["bbox_center"].flatten(0, 1)[idx][0]
                            - batch_lhand["bbox_scale"].flatten(0, 1)[idx][0] / 2
                        ).item(),
                        (
                            batch_lhand["bbox_center"].flatten(0, 1)[idx][1]
                            - batch_lhand["bbox_scale"].flatten(0, 1)[idx][1] / 2
                        ).item(),
                        (
                            batch_lhand["bbox_center"].flatten(0, 1)[idx][0]
                            + batch_lhand["bbox_scale"].flatten(0, 1)[idx][0] / 2
                        ).item(),
                        (
                            batch_lhand["bbox_center"].flatten(0, 1)[idx][1]
                            + batch_lhand["bbox_scale"].flatten(0, 1)[idx][1] / 2
                        ).item(),
                    ]
                )
                all_out[-1]["rhand_bbox"] = np.array(
                    [
                        (
                            batch_rhand["bbox_center"].flatten(0, 1)[idx][0]
                            - batch_rhand["bbox_scale"].flatten(0, 1)[idx][0] / 2
                        ).item(),
                        (
                            batch_rhand["bbox_center"].flatten(0, 1)[idx][1]
                            - batch_rhand["bbox_scale"].flatten(0, 1)[idx][1] / 2
                        ).item(),
                        (
                            batch_rhand["bbox_center"].flatten(0, 1)[idx][0]
                            + batch_rhand["bbox_scale"].flatten(0, 1)[idx][0] / 2
                        ).item(),
                        (
                            batch_rhand["bbox_center"].flatten(0, 1)[idx][1]
                            + batch_rhand["bbox_scale"].flatten(0, 1)[idx][1] / 2
                        ).item(),
                    ]
                )

        return all_out

    @torch.no_grad()
    def process_images(
        self,
        imgs,
        bboxes_list,
        cam_int=None,
        inference_type: str = "full",
        max_batch: int = 8,
    ):
        """Batched multi-frame inference for a static camera.

        Runs the per-frame crops of many frames through the model in a single
        (chunked) forward pass instead of calling ``process_one_image`` once per
        frame. All frames are assumed to share one camera intrinsics (static
        camera), so a single ``cam_int`` is applied to every frame. The output
        schema for each person is identical to ``process_one_image``.

        Args:
            imgs: list of RGB ``np.ndarray`` (H, W, 3), one per frame.
            bboxes_list: list of ``np.ndarray`` (Ni, 4) xyxy boxes aligned with
                ``imgs``. The batched path never runs a detector, so boxes are
                required (the golf pipeline always provides them).
            cam_int: shared intrinsics for all frames — a ``torch.Tensor`` (or
                array) broadcastable to ``(1, 3, 3)``. If ``None``, falls back to
                the FOV estimator (run once on the first frame) or the default FOV.
            inference_type: "full" (body + hands), "body", or "hand".
            max_batch: chunk size over the instance dimension to bound VRAM.

        Returns:
            list aligned to ``imgs``; element ``i`` is the list of per-person dicts
            for frame ``i`` (same keys as ``process_one_image``).
        """
        assert len(imgs) == len(bboxes_list), "imgs and bboxes_list must be aligned"

        # Clear cached results, mirroring process_one_image.
        self.batch = None
        self.image_embeddings = None
        self.output = None
        self.prev_prompt = []
        torch.cuda.empty_cache()

        results = [[] for _ in imgs]
        if len(imgs) == 0:
            return results

        # 1. Build a per-frame batch; record how many instances each frame has.
        #    Masks are unused in the batched (golf) path.
        per_frame_batches = []
        counts = []
        for img, bboxes in zip(imgs, bboxes_list):
            if not isinstance(img, np.ndarray):
                raise TypeError("process_images expects pre-decoded RGB numpy images")
            boxes = np.asarray(bboxes).reshape(-1, 4)
            b = prepare_batch(img, self.transform, boxes, None, None)
            per_frame_batches.append(b)
            counts.append(b["img"].shape[1])

        total = sum(counts)
        if total == 0:
            return results

        # 2. Concatenate all frames' crops along the instance axis.
        big_batch = concat_person_batches(per_frame_batches)

        # Resolve the shared camera intrinsics once (mirrors process_one_image).
        if cam_int is not None:
            cam_int_t = (
                cam_int if torch.is_tensor(cam_int) else torch.as_tensor(cam_int)
            )
            cam_int_t = cam_int_t.reshape(1, 3, 3)
        elif self.fov_estimator is not None:
            cam_int_t = self.fov_estimator.get_cam_intrinsics(
                big_batch["img_ori"][0].data
            ).reshape(1, 3, 3)
        else:
            cam_int_t = big_batch["cam_int"].reshape(1, 3, 3)

        # 3. Chunk the instance dim to bound VRAM; run the model on each chunk.
        flat_out = []  # frame-major, length == total
        for start in range(0, total, max_batch):
            end = min(start + max_batch, total)
            chunk = {
                key: big_batch[key][:, start:end]
                for key in PERSON_BATCH_KEYS
                if key in big_batch
            }
            chunk_imgs, chunk_counts = self._frames_in_slice(imgs, counts, start, end)

            chunk = recursive_to(chunk, "cuda")
            self.model._initialize_batch(chunk)
            chunk["cam_int"] = cam_int_t.to(chunk["img"]).clone()

            outputs = self.model.run_inference(
                chunk_imgs[0],
                chunk,
                inference_type=inference_type,
                transform_hand=self.transform_hand,
                thresh_wrist_angle=self.thresh_wrist_angle,
                imgs=chunk_imgs,
                counts=chunk_counts,
            )
            if inference_type == "full":
                pose_output, batch_lhand, batch_rhand, _, _ = outputs
            else:
                pose_output = outputs
                batch_lhand = batch_rhand = None

            out = recursive_to(pose_output["mhr"], "cpu")
            out = recursive_to(out, "numpy")

            flat_out.extend(
                self._build_person_outputs(
                    out, chunk, inference_type, batch_lhand, batch_rhand
                )
            )

        # 4. Split the frame-major instance list back into per-frame lists.
        pos = 0
        for i, count in enumerate(counts):
            results[i] = flat_out[pos : pos + count]
            pos += count
        return results
