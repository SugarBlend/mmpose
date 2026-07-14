from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from mmpose.structures import PoseDataSample
from pycocotools.coco import COCO
from tqdm import tqdm

from project.label_studio.pipelines.pipeline import MMPipeline
from config import EvalConfig
from overlay import (
    MODEL_JOINT_COLORS, MODEL_LIMB_COLORS, MODEL_SIDE_COLORS,
    BodyPartFilter, body_filter,
)
from mmpose.utils.logger import MMLogger
from project.label_studio.ann_utils import structs

logger = MMLogger.get_instance("Comparison")


def _skeleton_pairs(info: dict) -> list[tuple[int, int]]:
    return [(v["link"][0], v["link"][1]) for v in info.values()]


SKELETON_META = {nk: [link.link for link in structs[nk].skeleton.values()] for nk in structs}


@dataclass
class ModelVisState:
    pipeline: MMPipeline
    legend: str
    color: tuple[float, float, float]
    limb_color: tuple[float, float, float]
    side_colors: dict[str, tuple[float, float, float]]
    skeleton: list[tuple[int, int]]


@dataclass
class RawFrame:
    base_bgr: np.ndarray # original image
    kpt_sets: list[np.ndarray] # one kpt array per model
    states: list[ModelVisState] # same order as kpt_sets
    legends: list[str]
    joint_colors: list[tuple[float, float, float]]
    limb_colors: list[tuple[float, float, float]]
    side_colors_list: list[dict[str, tuple[float, float, float]]]


class RenderSettings:
    DEFAULT_JOINT_R = 5.0
    DEFAULT_LIMB_W = 3.0
    DEFAULT_SCORE_THR = 0.7

    def __init__(self) -> None:
        self.joint_r: float = self.DEFAULT_JOINT_R
        self.limb_w: float = self.DEFAULT_LIMB_W
        self.score_thr: float = self.DEFAULT_SCORE_THR
        self.use_side_color: bool = False
        self.bpart_filter: BodyPartFilter = body_filter


def _build_model_states(cfg: EvalConfig) -> list[ModelVisState]:
    states: list[ModelVisState] = []

    for idx, model_cfg in enumerate(cfg.models):
        if "sapiens2" in model_cfg.config_path:
            from project.label_studio.pipelines.sapiens2 import Sapiens2
            pipeline = Sapiens2(
                pose_checkpoint=model_cfg.model_path,
                pose_config=model_cfg.config_path,
            )
        else:
            pipeline = MMPipeline(
                pose_checkpoint=model_cfg.model_path,
                pose_config=model_cfg.config_path,
            )
        num_joints = pipeline.model_cfg.num_keypoints

        if num_joints not in SKELETON_META:
            raise ValueError(
                f"No skeleton metadata for {num_joints} joints. "
                f"Supported: {list(SKELETON_META)}"
            )

        color = MODEL_JOINT_COLORS[idx % len(MODEL_JOINT_COLORS)]
        limb_color = MODEL_LIMB_COLORS[idx % len(MODEL_LIMB_COLORS)]
        side_colors = MODEL_SIDE_COLORS[idx % len(MODEL_SIDE_COLORS)]
        states.append(ModelVisState(pipeline, model_cfg.legend, color, limb_color,
                                    side_colors, SKELETON_META[num_joints]))
        logger.info(f"[{idx}] {model_cfg.legend}")

    return states


def _infer(
    states: list[ModelVisState],
    image_path: str,
    bboxes: list,
) -> list[list[PoseDataSample]]:
    results: list[Optional[list[PoseDataSample]]] = [None] * len(states)
    for idx, state in enumerate(states):
        results[idx] = state.pipeline(image_path, bboxes)
    return results


def preprocess_all_frames(cfg: EvalConfig) -> list[RawFrame]:
    logger.info("Loading models…")
    states = _build_model_states(cfg)

    ann_file = cfg.models[0].ann_file
    dataset_path = cfg.models[0].dataset_folder
    coco = COCO(ann_file)
    cat_ids = coco.getCatIds(catNms=["person"])
    img_ids = coco.getImgIds(catIds=cat_ids)

    legends = [s.legend for s in states]
    joint_colors = [s.color for s in states]
    limb_colors = [s.limb_color for s in states]
    side_colors_list = [s.side_colors for s in states]

    raw_frames: list[RawFrame] = []

    from project.visualizers.heatmaps.reader import S3Reader
    reader = S3Reader()

    logger.info(f"Running inference on {len(img_ids)} images…")
    for image_id in tqdm(img_ids[:], desc="Inference"):
        for instance in coco.loadImgs(image_id):
            bboxes = [ann["bbox"] for ann in coco.imgToAnns.get(image_id, [])]
            if not bboxes:
                continue

            if dataset_path.startswith("minio://") or dataset_path.startswith("s3://"):
                image_path = "/".join([*Path(dataset_path).parts[1:], instance["file_name"]])
                image = reader.fetch(image_path, "datasets")
                image_path = image
            else:
                image_path = f"{dataset_path}/{instance['file_name']}"
                image = cv2.imread(image_path)

            if image is None:
                logger.warning(f"Cannot read: {image_path}")
                continue

            # launch inference sequentially
            results = _infer(states, image_path, bboxes)

            kpt_sets: list[np.ndarray] = []
            for samples in results:
                for data_sample in samples:
                    union = np.concatenate([data_sample.pred_instances.keypoints[0],
                                            data_sample.pred_instances.keypoint_scores[0].reshape(-1, 1)], axis=1)
                    kpt_sets.append(union)

            raw_frames.append(RawFrame(
                base_bgr=image.copy(),
                kpt_sets=kpt_sets,
                states=states,
                legends=legends,
                joint_colors=joint_colors,
                limb_colors=limb_colors,
                side_colors_list=side_colors_list,
            ))

    logger.info(f"Inference done. {len(raw_frames)} frames stored.")
    return raw_frames
