import click
import json
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Optional

BODY_KEYPOINT_NAMES = {
    0: "nose", 1: "left_eye", 2: "right_eye", 3: "left_ear", 4: "right_ear",
    5: "left_shoulder", 6: "right_shoulder", 7: "left_elbow", 8: "right_elbow",
    9: "left_wrist", 10: "right_wrist", 11: "left_hip", 12: "right_hip",
    13: "left_knee", 14: "right_knee", 15: "left_ankle", 16: "right_ankle"
}

FOOT_KEYPOINT_NAMES = {
    0: "left_big_toe", 1: "left_small_toe", 2: "left_heel",
    3: "right_big_toe", 4: "right_small_toe", 5: "right_heel"
}

LEFT_HAND_KEYPOINT_NAMES = {
    0: "left_hand_root", 1: "left_thumb1", 2: "left_thumb2", 3: "left_thumb3", 4: "left_thumb4", 5: "left_forefinger1",
    6: "left_forefinger2", 7: "left_forefinger3", 8: "left_forefinger4", 9: "left_middle_finger1",
    10: "left_middle_finger2", 11: "left_middle_finger3", 12: "left_middle_finger4",
    13: "left_ring_finger1", 14: "left_ring_finger2", 15: "left_ring_finger3", 16: "left_ring_finger4",
    17: "left_pinky_finger1", 18: "left_pinky_finger2", 19: "left_pinky_finger3", 20: "left_pinky_finger4"
}

RIGHT_HAND_KEYPOINT_NAMES = {
   0: "right_hand_root", 1: "right_thumb1", 2: "right_thumb2", 3: "right_thumb3", 4: "right_thumb4",
    5: "right_forefinger1", 6: "right_forefinger2", 7: "right_forefinger3", 8: "right_forefinger4",
    9: "right_middle_finger1", 10: "right_middle_finger2", 11: "right_middle_finger3",
    12: "right_middle_finger4", 13: "right_ring_finger1", 14: "right_ring_finger2", 15: "right_ring_finger3",
    16: "right_ring_finger4", 17: "right_pinky_finger1", 18: "right_pinky_finger2", 19: "right_pinky_finger3",
    20: "right_pinky_finger4"
}


KEYPOINT_CONFIGS = {
    "body": {
        "names": BODY_KEYPOINT_NAMES,
        "from_name": "label_body_keypoints",
        "prefix": "body"
    },
    "face": {
        "names_func": lambda i: f"face_{i:02d}",
        "from_name": "label_face_keypoints",
        "prefix": "face",
        "valid_key": "face_valid",
        "kpts_key": "face_kpts"
    },
    "left_hand": {
        "names": LEFT_HAND_KEYPOINT_NAMES,
        "from_name": "label_lefthand_keypoints",
        "prefix": "left_hand",
        "valid_key": "lefthand_valid",
        "kpts_key": "lefthand_kpts"
    },
    "right_hand": {
        "names": RIGHT_HAND_KEYPOINT_NAMES,
        "from_name": "label_righthand_keypoints",
        "prefix": "right_hand",
        "valid_key": "righthand_valid",
        "kpts_key": "righthand_kpts"
    },
    "foot": {
        "names": FOOT_KEYPOINT_NAMES,
        "from_name": "label_foot_keypoints",
        "prefix": "foot",
        "valid_key": "foot_valid",
        "kpts_key": "foot_kpts"
    }
}


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
    
    def _to_percent_coords(self, kp_x: float, kp_y: float) -> Tuple[float, float]:
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

    def process_keypoints(
        self,
        keypoints: List[float],
        config: Dict,
        image_id: int,
        ann_idx: int,
        point_ids: Dict
    ) -> List[Dict[str, Any]]:
        results = []
        prefix = config["prefix"]

        for i in range(0, len(keypoints), 3):
            if i + 2 >= len(keypoints):
                continue

            point_index = i // 3
            kp_x, kp_y, visibility = keypoints[i], keypoints[i + 1], keypoints[i + 2]

            if not self._is_valid_keypoint(kp_x, kp_y, visibility):
                continue

            x_percent, y_percent = self._to_percent_coords(kp_x, kp_y)

            if not self._is_in_bounds(x_percent, y_percent):
                continue

            if "names" in config:
                if point_index not in config["names"]:
                    continue
                label = config["names"][point_index]
            else:
                label = config["names_func"](point_index)

            point_id = f"{prefix}_{image_id}_{ann_idx}_{point_index:02d}"
            width = config.get("width", 0.5)

            annotation = self._create_keypoint_annotation(
                x_percent, y_percent, [label], point_id,
                config["from_name"], width
            )
            results.append(annotation)
            point_ids[prefix][point_index] = point_id

        return results

@click.command()
@click.option("--coco-file", type=str,
              default="sapiens_coco_wholebody.json", help="Path to file which consider labels in coco format.")
@click.option("--output-file", type=str,
              default="ls-pose-predictions.json", help="Path to dump file.")
@click.option("--local-storage-path", type=str,
              default="/data/local-files/?d=images", help="Url for local storage path from Label Studio.")
@click.option("--separate-threshold", type=int, default=1000, help="Annotations per label studio description file.")
def coco_wholebody_to_label_studio_predictions(
    coco_file: str,
    output_file: str,
    local_storage_path: str,
    separate_threshold: Optional[int] = 5
) -> List[Dict]:
    coco_file = Path(coco_file)
    output_file = Path(output_file)

    click.echo(f"Loading COCO file: {coco_file}")
    with coco_file.open("r", encoding="utf-8") as file:
        coco_data = json.load(file)
    image_mapping = {img["id"]: img for img in coco_data["images"]}

    annotations_by_image = {}
    for ann in coco_data["annotations"]:
        image_id = ann["image_id"]
        annotations_by_image.setdefault(image_id, []).append(ann)

    image_ids = list(annotations_by_image.keys())
    groups = [image_ids[i:i + separate_threshold] for i in range(0, len(image_ids), separate_threshold)]

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
                point_ids = {key: {} for key in KEYPOINT_CONFIGS.keys()}

                for keypoint_type, config in KEYPOINT_CONFIGS.items():
                    if keypoint_type == "body":
                        keypoints = ann.get("keypoints", [])
                        results = processor.process_keypoints(
                            keypoints, config, image_id, ann_idx, point_ids
                        )
                        result.extend(results)
                    else:
                        if ann.get(config.get("valid_key", False)):
                            keypoints = ann.get(config.get("kpts_key", []), [])
                            results = processor.process_keypoints(
                                keypoints, config, image_id, ann_idx, point_ids
                            )
                            result.extend(results)
            tasks.append(task)

        save_path = output_file.parent.joinpath(f"{output_file.stem}_{idx}.json")
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