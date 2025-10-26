import click
import json
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Optional

BODY_KEYPOINT_COCO_NAMES = {
    0: "nose", 1: "left_eye", 2: "right_eye", 3: "left_ear", 4: "right_ear",
    5: "left_shoulder", 6: "right_shoulder", 7: "left_elbow", 8: "right_elbow",
    9: "left_wrist", 10: "right_wrist", 11: "left_hip", 12: "right_hip",
    13: "left_knee", 14: "right_knee", 15: "left_ankle", 16: "right_ankle"
}
BODY_KEYPOINT_HALPE26_NAMES = BODY_KEYPOINT_COCO_NAMES.copy()
BODY_KEYPOINT_HALPE26_NAMES.update({17: "head", 18: "neck", 19: "hip"})

FOOT_KEYPOINT_WHOLEBODY_NAMES = {
    17: "left_big_toe", 18: "left_small_toe", 19: "left_heel",
    20: "right_big_toe", 21: "right_small_toe", 22: "right_heel"
}
FOOT_KEYPOINT_HALPE26_NAMES = {
    20: "left_big_toe", 21: "right_big_toe", 22: "left_small_toe",
    23: "right_small_toe", 24: "left_heel", 25: "right_heel"
}

LEFT_HAND_KEYPOINT_WHOLEBODY_NAMES = {
    91: "left_hand_root", 92: "left_thumb1", 93: "left_thumb2", 94: "left_thumb3", 95: "left_thumb4",
    96: "left_forefinger1",
    97: "left_forefinger2", 98: "left_forefinger3", 99: "left_forefinger4", 100: "left_middle_finger1",
    101: "left_middle_finger2", 102: "left_middle_finger3", 103: "left_middle_finger4",
    104: "left_ring_finger1", 105: "left_ring_finger2", 106: "left_ring_finger3", 107: "left_ring_finger4",
    108: "left_pinky_finger1", 109: "left_pinky_finger2", 110: "left_pinky_finger3", 111: "left_pinky_finger4"
}

RIGHT_HAND_KEYPOINT_WHOLEBODY_NAMES = {
    112: "right_hand_root", 113: "right_thumb1", 114: "right_thumb2", 115: "right_thumb3", 116: "right_thumb4",
    117: "right_forefinger1", 118: "right_forefinger2", 119: "right_forefinger3", 120: "right_forefinger4",
    121: "right_middle_finger1", 122: "right_middle_finger2", 123: "right_middle_finger3",
    124: "right_middle_finger4", 125: "right_ring_finger1", 126: "right_ring_finger2", 127: "right_ring_finger3",
    128: "right_ring_finger4", 129: "right_pinky_finger1", 130: "right_pinky_finger2", 131: "right_pinky_finger3",
    132: "right_pinky_finger4"
}


def get_keypoint_config(num_joints: int) -> Dict[str, Any]:
    keypoint_config = {}
    if num_joints == 26:
        correspondence = {
            "body": BODY_KEYPOINT_HALPE26_NAMES,
            "foot": FOOT_KEYPOINT_HALPE26_NAMES
        }
    elif num_joints == 133:
        correspondence = {
            "body": BODY_KEYPOINT_COCO_NAMES, "foot": FOOT_KEYPOINT_WHOLEBODY_NAMES,
            "face": {i: f"face_{i - 23:02d}" for i in range(23, 91)}, "left_hand": LEFT_HAND_KEYPOINT_WHOLEBODY_NAMES,
            "right_hand": RIGHT_HAND_KEYPOINT_WHOLEBODY_NAMES
        }
    else:
        raise Exception(f"Doesnt found processing for such number of joints: {num_joints}.")

    for part, names in correspondence.items():
        keypoint_config[part] = {
            "names": names,
            "from_name": f"label_{part}_keypoints",
            "prefix": part
        }
    return keypoint_config


class KeypointProcessor(object):
    def __init__(self, image_info: Dict, bbox_id: str) -> None:
        self.image_info = image_info
        self.bbox_id = bbox_id
        self.width = image_info["width"]
        self.height = image_info["height"]

    @staticmethod
    def _is_valid_keypoint(kp_x: float, kp_y: float, visibility: float) -> bool:
        return visibility > 0 and kp_x > 0 and kp_y > 0

    @staticmethod
    def _is_in_bounds(x_percent: float, y_percent: float) -> bool:
        return 0 <= x_percent <= 100 and 0 <= y_percent <= 100

    def _to_percent_coord(self, kp_x: float, kp_y: float) -> Tuple[float, float]:
        x_percent = (kp_x / self.width) * 100
        y_percent = (kp_y / self.height) * 100
        return x_percent, y_percent

    def _create_keypoint_annotation(
            self,
            x_percent: float,
            y_percent: float,
            labels: List[str],
            point_id: str,
            from_name: str,
            width: float = 0.5
    ) -> Dict[str, Any]:
        return {
            "id": point_id,
            "type": "keypointlabels",
            "value": {
                "x": x_percent,
                "y": y_percent,
                "width": width,
                "keypointlabels": labels
            },
            "parentID": self.bbox_id,
            "to_name": "image",
            "from_name": from_name,
            "original_width": self.width,
            "original_height": self.height
        }

    def process_joints(
            self,
            keypoints: List[float],
            config: Dict,
            image_id: int,
            ann_idx: int,
            point_ids: Dict
    ) -> List[Dict[str, Any]]:
        results = []
        prefix = config["prefix"]

        for i in range(min(config["names"]) * 3, (max(config["names"]) + 1) * 3, 3):
            point_index = i // 3
            kp_x, kp_y, visibility = keypoints[i], keypoints[i + 1], keypoints[i + 2]

            if not self._is_valid_keypoint(kp_x, kp_y, visibility):
                continue

            x_percent, y_percent = self._to_percent_coord(kp_x, kp_y)

            if not self._is_in_bounds(x_percent, y_percent):
                continue

            if point_index not in config["names"]:
                continue
            label = config["names"][point_index]

            point_id = f"{prefix}-{image_id}-{ann_idx}-{point_index:03d}"

            annotation = self._create_keypoint_annotation(
                x_percent, y_percent, [label], point_id, config["from_name"], config.get("width", 0.5)
            )
            results.append(annotation)
            point_ids[prefix][point_index] = point_id

        return results


@click.command()
@click.option("--coco-file", type=str,
              default="anns/coco/2025-10-20 12-50-24_halpe-x.json",
              help="Path to file which consider labels in coco format.")
@click.option("--output-file", type=str,
              default="anns/ls/2025-10-20 12-50-24.json", help="Path to dump file.")
@click.option("--local-storage-path", type=str,
              default="/data/local-files/?d=NewPoseCustom/2025-10-20 12-50-24",
              help="Url for local storage path from Label Studio.")
@click.option("--num-joints", type=int, default=26, help="Number of keypoints per person.")
@click.option("--frames-per-task", type=int, default=1000, help="Annotations per label studio description file.")
def coco_wholebody_to_label_studio_predictions(
        coco_file: str,
        output_file: str,
        local_storage_path: str,
        num_joints: int,
        frames_per_task: Optional[int] = 100
) -> List[Dict]:
    coco_file = Path(coco_file)
    output_file = Path(output_file)

    click.echo(f"Loading COCO file: {coco_file}")
    with coco_file.open("r", encoding="utf-8") as file:
        coco_data = json.load(file)
    image_mapping = {img["id"]: img for img in coco_data["images"]}

    keypoint_config = get_keypoint_config(num_joints)

    annotations_by_image = {}
    for ann in coco_data["annotations"]:
        image_id = ann["image_id"]
        annotations_by_image.setdefault(image_id, []).append(ann)

    image_ids = list(annotations_by_image.keys())
    groups = [image_ids[i: i + frames_per_task] for i in range(0, len(image_ids), frames_per_task)]

    tasks = []
    for idx, group in tqdm(enumerate(groups), total=(len(groups)), colour="blue", desc="Processing group"):
        for image_id in tqdm(group, colour="green", desc="Processing annotation"):
            image_info = image_mapping[image_id]
            annotations = annotations_by_image[image_id]

            task = {
                "data": {
                    "image": f"{local_storage_path}/{image_info['file_name']}"
                },
                "predictions": [{
                    "result": [],
                    "model_version": "sapiens_1b_coco_wholebody_best_coco_wholebody_AP_727"
                }]
            }

            result = task["predictions"][0]["result"]

            for ann_idx, ann in enumerate(annotations):
                bbox_id = f"bbox_{image_id}_{ann_idx}"
                bbox_annotation = _create_bbox_annotation(ann, image_info, bbox_id)
                if bbox_annotation:
                    result.append(bbox_annotation)

                processor = KeypointProcessor(image_info, bbox_id)
                point_ids = {key: {} for key in keypoint_config.keys()}

                for keypoint_type, config in keypoint_config.items():
                    results = processor.process_joints(
                        ann["keypoints"], config, image_id, ann_idx, point_ids
                    )
                    result.extend(results)
            tasks.append(task)

        save_path = output_file.parent.joinpath(f"{output_file.stem}/{idx}.{len(groups) - 1}-{frames_per_task}.json")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", encoding="utf-8") as file:
            json.dump(tasks, file, indent=2, ensure_ascii=False)
        tasks.clear()

    click.echo(f"Convertation completed! Created {len(tasks)} tasks.")
    click.echo(f"Results saved to: {output_file}")
    return tasks


def _create_bbox_annotation(ann: Dict, image_info: Dict, bbox_id: str) -> Optional[Dict]:
    bbox = ann.get("bbox", [])
    if len(bbox) != 4:
        return None

    x, y, width, height = bbox
    img_width, img_height = image_info["width"], image_info["height"]

    return {
        "id": bbox_id,
        "type": "rectanglelabels",
        "value": {
            "x": max(0, (x / img_width) * 100),
            "y": max(0, (y / img_height) * 100),
            "width": max(0, (width / img_width) * 100),
            "height": max(0, (height / img_height) * 100),
            "rotation": 0,
            "rectanglelabels": ["person"]
        },
        "to_name": "image",
        "from_name": "label_rectangles",
        "original_width": img_width,
        "original_height": img_height
    }


if __name__ == "__main__":
    coco_wholebody_to_label_studio_predictions()
