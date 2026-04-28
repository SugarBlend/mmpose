import torch
from typing import Any
import numpy as np
import uuid
from enum import Enum


def _patched_torch_load(*args, **kwargs) -> Any:
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


_orig_torch_load = torch.load
torch.load = _patched_torch_load


class Labels(Enum):
    COCO17 = 17
    COCO133 = 133
    HALPE26 = 26
    HALPE136 = 136


_FEETS_KEYPOINTS: list[str] = [
    "left_big_toe", "left_small_toe", "left_heel", "right_big_toe", "right_small_toe", "right_heel",
]

_HANDS_KEYPOINTS: list[str] = [
    "hand_root", "thumb1", "thumb2", "thumb3", "thumb4", "forefinger1", "forefinger2", "forefinger3", "forefinger4",
    "middle_finger1", "middle_finger2", "middle_finger3", "middle_finger4", "ring_finger1", "ring_finger2",
    "ring_finger3", "ring_finger4", "pinky_finger1", "pinky_finger2", "pinky_finger3", "pinky_finger4",
]

COCO17_KEYPOINTS: list[str] = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

HALPE26_KEYPOINTS: list[str] = [
    # body 0-16
    *COCO17_KEYPOINTS,
    # additional points 17-19
    "head",
    "neck",
    "hip",
    # foot 20-25
    *np.array(_FEETS_KEYPOINTS).reshape(-1, 3).flatten(order="F").tolist()
]

COCO133_KEYPOINTS: list[str] = [
    # body 0-16
    *COCO17_KEYPOINTS,
    # foot 17-22
    *_FEETS_KEYPOINTS,
    # face 23-90
    *[f"face_{i:02d}" for i in range(68)],
    # left hand 91-111
    *[f"left_{name}" for name in _HANDS_KEYPOINTS],
    # right hand 112-132
    *[f"right_{name}" for name in _HANDS_KEYPOINTS]
]

HALPE136_KEYPOINTS: list[str] = [
    # body + foot 0-25
    *HALPE26_KEYPOINTS,
    # face 26-94
    *[f"face_{i}" for i in range(68)],
    # left hand 95-115
    *[f"left_{name}" for name in _HANDS_KEYPOINTS],
    # right hand 116-136
    *[f"right_{name}" for name in _HANDS_KEYPOINTS]
]


def get_ls_fields(num_joints: int) -> tuple[dict[str, str], list[str]]:
    if num_joints == Labels.COCO17.value:
        return {
            kpt: "label_body_keypoints" for idx, kpt in enumerate(COCO17_KEYPOINTS)
        }, COCO17_KEYPOINTS
    if num_joints == Labels.HALPE26.value:
        # correspondence to labeling configuration file - "config-halpe.xml"
        return {
            kpt: ("label_body_keypoints" if idx < 20 else "label_foot_keypoints")
            for idx, kpt in enumerate(HALPE26_KEYPOINTS)
        }, HALPE26_KEYPOINTS
    elif num_joints == Labels.COCO133.value:
        # correspondence to labeling configuration file - "config-wholebody.xml"
        return {
            kpt: (
                "label_body_keypoints" if idx < 17  else
                "label_foot_keypoints" if idx < 23  else
                "label_face_keypoints" if idx < 91  else
                "label_left_hand_keypoints" if idx < 112 else
                "label_right_hand_keypoints"
            )
            for idx, kpt in enumerate(COCO133_KEYPOINTS)
        }, COCO133_KEYPOINTS
    elif num_joints == Labels.HALPE136.value:
        return {
            kpt: (
                "label_body_keypoints" if idx < 17 else
                "label_foot_keypoints" if idx < 26 else
                "label_face_keypoints" if idx < 95 else
                "label_left_hand_keypoints" if idx < 116 else
                "label_right_hand_keypoints"
            )
            for idx, kpt in enumerate(HALPE136_KEYPOINTS)
        }, HALPE136_KEYPOINTS
    else:
        raise NotImplementedError(f"Passed unsupported label type. Supported num joints: {Labels._member_names_=}, "
                                  f"but received: {num_joints}.")


def make_rectanglelabels(
    x1: float, y1: float, x2: float, y2: float,
    img_w: int, img_h: int,
    rect_id: str,
    score: float,
) -> dict[str, Any]:
    return {
        "id": rect_id,
        "type": "rectanglelabels",
        "from_name": "label_rectangles",
        "to_name": "image",
        "original_width": img_w,
        "original_height": img_h,
        "image_rotation": 0,
        "value": {
            "x": x1 / img_w * 100,
            "y": y1 / img_h * 100,
            "width": (x2 - x1) / img_w * 100,
            "height": (y2 - y1) / img_h * 100,
            "rotation": 0,
            "rectanglelabels": ["person"],
        },
        "score": score,
    }


def make_keypointlabels(
    kx: float, ky: float,
    img_w: int, img_h: int,
    kpt_name: str,
    from_name: str,
    rect_id: str,
    score: float,
) -> dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "type": "keypointlabels",
        "from_name": from_name,
        "to_name": "image",
        "parentID": rect_id,
        "original_width": img_w,
        "original_height": img_h,
        "image_rotation": 0,
        "value": {
            "x": kx / img_w * 100,
            "y": ky / img_h * 100,
            "width": 0.5,
            "keypointlabels": [kpt_name],
        },
        "score": score,
    }
