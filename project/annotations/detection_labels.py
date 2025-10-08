import click
import cv2
from datetime import datetime
import glob
import json
import os
from pathlib import Path
from typing import List
from tqdm import tqdm
import torch
from ultralytics import YOLO


@click.command()
@click.option(
    "--files",
    type=str,
    default="",
    help="Folder of images to annotate."
)
@click.option(
    "--output-json",
    type=str,
    default="",
    help="Path to the annotation file in 'json' format."
)
@click.option(
    "--yolo-version",
    type=str,
    default="yolo12x.pt",
    help="Version of the yolo used."
)
@click.option(
    "--no-preview",
    is_flag=True,
    help="Disable image preview."
)
@click.option(
    "--vis-delay",
    type=int,
    default=30,
    help="Hold to frame display in ms."
)
def annotate(
    files: List[str],
    output_json: str,
    yolo_version: str,
    no_preview: bool,
    vis_delay: int,
):
    files = glob.glob(f"{files}/*")
    if not len(files):
        raise Exception("Folder is empty.")

    model = YOLO(yolo_version, verbose=False)
    model.cuda()
    model.eval()
    model.fuse()

    coco_struct = {
        "info": {
            "description": "Custom dataset annotations",
            "url": "",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "",
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "licenses": [{
            "url": "",
            "id": 1,
        }],
        "images": [],
        "annotations": [],
        "categories": [{
            "id": 1,
            "name": "person",
            "supercategory": "person"
        }]
    }

    image_id = 1
    annotation_id = 1

    with torch.inference_mode():
        for file in tqdm(files, desc="Processed files"):
            image = cv2.imread(file)
            if image is None:
                continue

            height, width = image.shape[:2]
            filename = os.path.basename(file)

            image_info = {
                "id": image_id,
                "file_name": filename,
                "width": width,
                "height": height
            }
            coco_struct["images"].append(image_info)

            detections = model(image, verbose=False)[0].boxes.data
            person_detections = detections[detections[:, 5] == 0.].cpu().numpy()

            for detection in person_detections:
                x1, y1, x2, y2, conf, cls = detection

                bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": bbox,
                    "area": float((x2 - x1) * (y2 - y1)),
                    "segmentation": [],
                    "iscrowd": 0,
                    "score": float(conf)
                }
                coco_struct["annotations"].append(annotation)
                annotation_id += 1

                if not no_preview:
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    cv2.putText(image, f"{conf:.2f}", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (127, 127, 127), 1)

            if not no_preview:
                cv2.imshow("Annotated", image)
                cv2.waitKey(vis_delay)
            image_id += 1

    if not no_preview:
        cv2.destroyAllWindows()

    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open(mode="w", encoding="utf-8") as file:
        json.dump(coco_struct, file, indent=2, ensure_ascii=False)

    click.echo(f"Annotation save by path: {output_json}")
    click.echo(f"Total images: {len(coco_struct['images'])}")
    click.echo(f"Total annotations: {len(coco_struct['annotations'])}")


if __name__ == "__main__":
    annotate()
