import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import torch
from mmpose.structures import PoseDataSample
from pycocotools.coco import COCO
from tqdm import tqdm

from deploy2serve.deployment.projects.sapiens.utils.palettes import (
    COCO_HALPE26_SKELETON_INFO,
    COCO_WHOLEBODY_SKELETON_INFO,
)

sys.path.insert(0, Path(__file__).parents[2].as_posix())
from project.label_studio.pipelines.pipeline import MMPipeline
from config import EvalConfig
from overlay import (
    MODEL_JOINT_COLORS, MODEL_LIMB_COLORS,
    draw_skeleton_pretty, render_hud,
)
from mmpose.utils.logger import MMLogger


logger = MMLogger.get_instance("Comparison")


def _skeleton_pairs(info: dict) -> list[tuple[int, int]]:
    return [(v["link"][0], v["link"][1]) for v in info.values()]


SKELETON_META = {
    26: _skeleton_pairs(COCO_HALPE26_SKELETON_INFO),
    133: _skeleton_pairs(COCO_WHOLEBODY_SKELETON_INFO),
}


@dataclass
class ModelVisState:
    pipeline: MMPipeline
    legend: str
    color: tuple[float, float, float] # joint color (r,g,b) 0-1
    limb_color: tuple[float, float, float] # limb color  (r,g,b) 0-1
    skeleton: list[tuple[int, int]]


def _build_model_states(cfg: EvalConfig) -> list[ModelVisState]:
    first_pipeline = MMPipeline(
        pose_checkpoint=cfg.models[0].model_path,
        pose_config=cfg.models[0].config_path,
    )
    num_joints = first_pipeline.model_cfg.num_keypoints

    if num_joints not in SKELETON_META:
        raise ValueError(
            f"No skeleton metadata for {num_joints} joints. Supported: {list(SKELETON_META)}"
        )

    skeleton = SKELETON_META[num_joints]
    states: list[ModelVisState] = []

    for idx, model_cfg in enumerate(cfg.models):
        pipeline = first_pipeline if idx == 0 else MMPipeline(
            pose_checkpoint=model_cfg.model_path,
            pose_config=model_cfg.config_path,
        )
        color = MODEL_JOINT_COLORS[idx % len(MODEL_JOINT_COLORS)]
        limb_color = MODEL_LIMB_COLORS[idx % len(MODEL_LIMB_COLORS)]
        states.append(ModelVisState(pipeline, model_cfg.legend, color, limb_color, skeleton))
        logger.info(f"[{idx}] {model_cfg.legend}")

    return states


def _infer(
    states: list[ModelVisState],
    image_path: str,
    bboxes: list,
) -> list[list[PoseDataSample]]:
    results: list[Optional[list[PoseDataSample]]] = [None] * len(states)

    if torch.cuda.is_available() and len(states) > 1:
        streams = [torch.cuda.Stream() for _ in states]
        default = torch.cuda.default_stream()
        for idx, (state, stream) in enumerate(zip(states, streams)):
            stream.wait_stream(default)
            with torch.cuda.stream(stream):
                results[idx] = state.pipeline(image_path, bboxes)
        for stream in streams:
            default.wait_stream(stream)
    else:
        for idx, state in enumerate(states):
            results[idx] = state.pipeline(image_path, bboxes)

    return results


def visual_difference(cfg: EvalConfig) -> None:
    logger.info("Loading models...")
    states = _build_model_states(cfg)

    # ann_file and dataset_path: prefer top-level [comparison] section,
    # fall back to first model's fields
    ann_file = cfg.models[0].ann_file
    dataset_path = cfg.models[0].dataset_folder

    coco = COCO(ann_file)
    cat_ids = coco.getCatIds(catNms=["person"])
    img_ids = coco.getImgIds(catIds=cat_ids)

    cfg.window_name = "Pose Comparison"
    cv2.namedWindow(cfg.window_name, cv2.WINDOW_GUI_EXPANDED)

    legends = [s.legend for s in states]
    joint_colors = [s.color for s in states]
    limb_colors = [s.limb_color for s in states]
    total = len(img_ids)

    for frame_idx, image_id in enumerate(tqdm(img_ids, desc="Images"), start=1):
        for instance in coco.loadImgs(image_id):
            bboxes = [ann["bbox"] for ann in coco.imgToAnns.get(image_id, [])]
            if not bboxes:
                continue

            image_path = f'{dataset_path}/{instance["file_name"]}'
            image = cv2.imread(image_path)
            if image is None:
                logger.warning(f"Cannot read: {image_path}")
                continue

            all_results = _infer(states, image_path, bboxes)

            for state, samples in zip(states, all_results):
                for sample in samples:
                    kpts = sample.pred_instances.keypoints[0]
                    draw_skeleton_pretty(
                        image, kpts, state.skeleton,
                        state.color, state.limb_color,
                    )

            image = render_hud(image, legends, joint_colors, limb_colors, frame_idx, total)

            cv2.imshow(cfg.window_name, image)
            if cv2.waitKey(0) == ord("q"):
                logger.info("Quit")
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visual pose model comparison")
    parser.add_argument("--config", "-c", type=str, required=True,
                        help="Path to eval-config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = EvalConfig.load(args.config)
    visual_difference(cfg)
