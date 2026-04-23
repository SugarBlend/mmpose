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

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


class LSConverter(object):
    def __init__(self, config_path: str) -> None:
        load_dotenv()
        self.client = LabelStudio(base_url=os.getenv("LABEL_STUDIO_URL"), api_key=os.getenv("LABEL_STUDIO_API_KEY"))
        self.label_values = self._parse_labels(config_path)

        # Joints mapping from ls .xml file
        self.categories = [{"id": i + 1, "name": name} for i, name in enumerate(self.label_values)]
        self.category_name_to_id = {cat["name"]: cat["id"] for cat in self.categories}

        # Add 'person' category with equal for bounding box
        self.categories.append({"id": 0, "name": "person"})
        self.category_name_to_id["person"] = 0

    @staticmethod
    def _parse_labels(config_path: str) -> list[str]:
        root = ET.parse(config_path).getroot()
        return [label.get("value") for label in root.findall(".//KeyPointLabels/Label")]

    def tasks2json(self, tasks, project_name, output_dir="outputs"):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        labels_file = output_dir.joinpath(f"{project_name}.json")

        images: list[dict[str, int | str]] = []
        annotations: list[dict[str, Any]] = []
        joints_number = len(self.label_values)

        for idx, task in enumerate(tasks):
            image_name = Path(unquote(task["data"]["image"])).name
            image_id = task["id"]

            if not task.get("annotations"):
                logger.warning(f"No annotations for project: '{project_name}', task {idx}, '{task['data']['image']}'")
                continue

            joints_tensor = np.zeros((joints_number, 3))

            for annotation in task["annotations"]:
                results = annotation["result"]
                results.sort(key=lambda x: 0 if x["type"] == "keypointlabels" else 1)

                width, height = None, None

                for label in results:
                    category_name = None
                    for key in ["rectanglelabels", "polygonlabels", "labels", "keypointlabels"]:
                        if key == label["type"] and len(label["value"].get(key, [])) > 0:
                            category_name = label['value'][key][0]
                            break

                    if category_name is None:
                        logger.warning(f"Unknown label or empty for image {image_name}")
                        continue

                    if width is None or height is None:
                        width = label.get("original_width")
                        height = label.get("original_height")
                        if width is None or height is None:
                            logger.warning(f"Width/height missing for {image_name}")
                            continue
                        images.append({
                            "id": image_id,
                            "file_name": image_name,
                            "width": width,
                            "height": height
                        })

                    if label["type"] in ["rectanglelabels", "labels"]:
                        self.process_rectangle(label, annotations)
                    elif label["type"] == "polygonlabels":
                        self.process_polygon(label, annotations)
                    elif label["type"] == "keypointlabels":
                        self.process_keypoints(
                            label, joints_tensor, annotations, image_id, self.category_name_to_id[category_name]
                        )

        description = {
            "images": images,
            "categories": [
                {
                  "id": 1,
                  "name": "person"
                }
              ],
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
    def process_rectangle(label: dict[str, Any], annotations: list[dict[str, Any]]) -> None:
        value = label["value"]
        w = value["width"] * label["original_width"] / 100
        h = value["height"] * label["original_height"] / 100

        annotations[-1].update({
            "bbox": [
                value["x"] * label["original_width"] / 100,
                value["y"] * label["original_height"] / 100,
                w, h
            ],
            "area": w * h,
            "category_id": 1
        })

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

    @staticmethod
    def process_keypoints(
        label: dict[str, Any],
        joints_tensor: np.ndarray,
        annotations: list[dict[str, Any]],
        image_id: int,
        category_id: int
    ) -> None:
        value = label["value"]
        x = value["x"] * label["original_width"] / 100
        y = value["y"] * label["original_height"] / 100
        joints_tensor[category_id - 1] = [x, y, 2]

        if not annotations or annotations[-1]["image_id"] != image_id:
            annotation = {
                "id": len(annotations),
                "image_id": image_id,
                "keypoints": joints_tensor.flatten().tolist(),
                "iscrowd": 0,
                "ignore": 0,
                "num_keypoints": 1
            }
            annotations.append(annotation)
        else:
            annotations[-1]["keypoints"] = joints_tensor.flatten().tolist()
            annotations[-1]["num_keypoints"] += 1

    def _export_project_annotations(self, project_id: int) -> list[dict[str, Any]]:
        export_job = self.client.projects.exports.create(id=project_id, title=f"Export_{project_id}")
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

        tasks = self.client.projects.exports.download(id=project_id, export_pk=export_id)
        tasks_json = json.loads(b"".join(tasks).decode("utf-8"))
        return tasks_json

    def process_annotations(self, output_dir: str = "outputs") -> None:
        for project in self.client.projects.list().items:
            logger.info(f"Processing project: {project.title} (id={project.id})")
            tasks = self._export_project_annotations(project.id)
            self.tasks2json(tasks, project.title, output_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert Label Studio projects to COCO via snapshot export")
    parser.add_argument("--config", default="configs/config-halpe.xml", help="Label Studio config XML")
    parser.add_argument("--output_folder", default="../annotations", help="Folder to save COCO JSONs")
    args = parser.parse_args()

    converter = LSConverter(args.config)
    converter.process_annotations(output_dir=args.output_folder)
