import argparse
import os
import json
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import xml.etree.ElementTree as ET
from label_studio_sdk import LabelStudio
import time
from typing import Any
from urllib.parse import unquote
from dotenv import load_dotenv
from collections import defaultdict
import re
from tqdm import tqdm


logger = logging.getLogger(__name__)
logging.basicConfig(level=getattr(logging, os.getenv("LOG_LEVEL", default="INFO")), format="[%(levelname)s] %(message)s")


class LSConverter(object):
    def __init__(self, url: str | None = None, api_key: str | None = None) -> None:
        self.client = LabelStudio(
            base_url=url or os.getenv("LABEL_STUDIO_URL"),
            api_key=api_key or os.getenv("LABEL_STUDIO_API_KEY")
        )
        self.label_values: None | list[str] = None # represent ordered names of keypoints

    @staticmethod
    def _parse_labels(config_path: str) -> list[str | None]:
        root = ET.fromstring(config_path)
        return [label.get("value") for label in root.findall(".//KeyPointLabels/Label")]

    def tasks2json(self, tasks, project_name, output_dir="outputs"):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        labels_file = output_dir.joinpath(f"{project_name}.json")
        images: list[dict[str, int | str]] = []
        annotations: list[dict[str, Any]] = []

        assert self.label_values is not None, "Skipped step for parsing label config from 'project.label_config'."
        joints_number = len(self.label_values)
        label_order = {name: i for i, name in enumerate(self.label_values)}

        seen_image_ids: set[int] = set()

        pb = tqdm(total=len(tasks), desc="Progress", leave=True, ncols=80, unit="task")
        for task in tasks:
            image_name = Path(unquote(task["data"]["image"])).relative_to(args.root_dataset_path).as_posix()
            image_id = task["id"]

            if not task.get(args.data_type):
                logger.debug(f"No {args.data_type} for project: '{project_name}', task {pb.n}, '{task['data']['image']}'")
                continue

            for annotation in task[args.data_type]:
                if annotation.get("was_cancelled"):
                    logger.debug(f"Skipping cancelled annotation for task {pb.n}, '{image_name}'")
                    continue

                results = annotation["result"]

                boxes = {}
                keypoints_by_parent = defaultdict(list)
                polygons = []

                # Sorting each bounding boxes and search them children by using parentID connection
                for label in results:
                    if label["type"] in ("rectanglelabels", "labels"):
                        boxes[label["id"]] = label
                    elif label["type"] == "keypointlabels":
                        parent_id = label.get("parentID")
                        if not parent_id:
                            logger.warning(f"Keypoint without parentID for {image_name}, annotation will be skipped")
                            continue
                        keypoints_by_parent[parent_id].append(label)
                    elif label["type"] == "polygonlabels":
                        polygons.append(label)

                for box_id, box_label in boxes.items():
                    category_name = None
                    key = box_label["type"]
                    if len(box_label["value"].get(key, [])) > 0:
                        category_name = box_label["value"][key][0]

                    if category_name is None:
                        logger.warning(f"Unknown label or empty for image {image_name}")
                        continue

                    width = box_label.get("original_width")
                    height = box_label.get("original_height")
                    if width is None or height is None:
                        logger.warning(f"Width/height missing for {image_name}")
                        continue

                    if image_id not in seen_image_ids:
                        images.append({
                            "id": image_id,
                            "file_name": image_name,
                            "width": width,
                            "height": height
                        })
                        seen_image_ids.add(image_id)

                    self.process_rectangle(box_label, annotations, image_id, joints_number)

                    joints_tensor = np.zeros((joints_number, 3))

                    for kp_label in keypoints_by_parent.get(box_id, []):
                        kp_values = kp_label["value"].get("keypointlabels", [])

                        if not kp_values:
                            continue

                        name = kp_values[0]
                        if name not in label_order:
                            logger.warning(f"Unknown keypoint label '{name}' for image {image_name}")
                            continue

                        ind = label_order[name] + 1

                        try:
                            self.process_keypoints(kp_label, joints_tensor, annotations, ind)
                        except KeyError as error:
                            logger.warning(error)

                for label in polygons:
                    self.process_polygon(label, annotations)
            pb.update()

        description = {
            "images": images,
            "categories": [{"id": 1, "name": "person"}],
            "annotations": annotations,
            "info": {
                "year": datetime.now().year,
                "version": "1.0",
                "description": f'Converted from Label Studio project {project_name}',
                "contributor": 'LSConverter',
                "date_created": str(datetime.now())
            }
        }
        labels_file.write_text(json.dumps(description, indent=2), encoding="utf-8")
        logger.info(f"Annotation json saved by: {labels_file}")
        return labels_file

    @staticmethod
    def process_rectangle(label: dict[str, Any], annotations: list[dict[str, Any]], image_id: int,
                          joints_number: int) -> None:
        value = label["value"]
        w = value["width"] * label["original_width"] / 100
        h = value["height"] * label["original_height"] / 100

        annotations.append({
            "id": len(annotations),
            "image_id": image_id,
            "bbox": [
                value["x"] * label["original_width"] / 100,
                value["y"] * label["original_height"] / 100,
                w, h
            ],
            "area": w * h,
            "category_id": 1,
            "iscrowd": 0,
            "ignore": 0,
            "num_keypoints": 0,
            "keypoints": [0] * (joints_number * 3),
        })

    @staticmethod
    def process_keypoints(
        label: dict[str, Any],
        joints_tensor: np.ndarray,
        annotations: list[dict[str, Any]],
        category_id: int
    ) -> None:
        value = label["value"]
        x = value["x"] * label["original_width"] / 100
        y = value["y"] * label["original_height"] / 100
        joints_tensor[category_id - 1] = [x, y, 2]

        annotations[-1]["keypoints"] = joints_tensor.flatten().tolist()
        annotations[-1]["num_keypoints"] += 1

    #FIXME: Not tested, at now optional
    @staticmethod
    def process_polygon(label: dict[str, Any], annotations: list[dict[str, Any]]) -> None:
        width = label["original_width"]
        height = label["original_height"]
        points_abs = [(x / 100 * width, y / 100 * height) for x, y in label["value"]["points"]]
        x, y = zip(*points_abs)
        x1, y1, x2, y2 = min(x), min(y), max(x), max(y)
        bbox = [x1, y1, x2 - x1, y2 - y1]
        area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

        annotations[-1].update({
            "bbox": bbox,
            "area": area,
            "segmentation": [[coord for point in points_abs for coord in point]]
        })

    def _export_project_annotations(self, project_id: int) -> list[dict[str, Any]]:
        export_job = self.client.projects.exports.create(
            id=project_id,
            title=f"Export_{project_id}",
            serialization_options={
                "predictions": {"only_id": False},
                "drafts": {"only_id": False},
            },
        )
        export_id = export_job.id
        logger.info(f"Snapshot export created: {export_id}")

        while True:
            job = self.client.projects.exports.get(id=project_id, export_pk=export_id)
            if job.status == "completed":
                logger.info(f"Export completed: {export_id}")
                break
            elif job.status == "failed":
                raise RuntimeError(f"Export failed for project {project_id}")
            else:
                logger.info(f"Waiting for export {export_id}, status: {job.status}")
                time.sleep(2)

        tasks = self.client.projects.exports.download(id=project_id, export_pk=export_id, export_type="JSON")
        tasks_json = json.loads(b"".join(tasks).decode("utf-8"))
        return tasks_json

    def process_annotations(self, output_dir: str = "outputs") -> None:
        patterns = args.patterns.split(",")
        compilers = [re.compile(pattern) for pattern in patterns]
        for project in self.client.projects.list().items:
            if any(compiler.search(project.title) for compiler in compilers):
                counter = project.total_annotations_number if args.data_type == "annotations" else project.total_predictions_number
                if not counter:
                    logger.warning(f"Skip empty project: '{project.title}', doesn't detect any {args.data_type}.")
                    continue

                logger.info(f"Processing project: {project.title} (id={project.id})")
                # Fetch label config directly from project settings
                self.label_values = self._parse_labels(project.label_config)
                tasks = self._export_project_annotations(project.id)
                self.tasks2json(tasks, project.title, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Label Studio projects to COCO via snapshot export")
    parser.add_argument("--root-dataset-path", default="s3://datasets",
                        help="The base path relative to which paths to the data described in the annotation are constructed.")
    parser.add_argument("--output_folder", default="../annotations/labelstudio",
                        help="Folder to save COCO JSONs")
    # parser.add_argument("--name_pattern", default=".*",
    #                     help="Regular expression for filtering project by them name.")
    parser.add_argument("--patterns", default="^Pose Annotation,^Foots Pose Annotation",
                        help="Regular expression for filtering project by them name.")
    # parser.add_argument("--name_pattern", default=r"Hands\s\+\sBody Pose Annotation",
    #                     help="Regular expression for filtering project by them name.")
    parser.add_argument("--data_type", choices=["predictions", "annotations"], default="annotations",
                        help="'Predictions' are the type of data obtained from LS from an auto-labeler, "
                             "'annotations' are data from LS that are marked up by people. ")
    args = parser.parse_args()

    load_dotenv()
    converter = LSConverter()
    converter.process_annotations(output_dir=args.output_folder)
