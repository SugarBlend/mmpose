import itertools
import json
from datetime import datetime
from pathlib import Path
from typing import Iterator, Any

import click
import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
PERSON_CLASS_ID  = 0


def _iter_images(folder: str) -> Iterator[Path]:
    root = Path(folder)
    if not root.is_dir():
        raise click.BadParameter(f"Not a directory: {folder}", param_hint="--files")
    paths = sorted(p for p in root.iterdir()
                   if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise click.UsageError(f"No images found in: {folder}")
    return iter(paths)


def _coco_info(model: str) -> dict:
    now = datetime.now()
    return {
        "info": {
            "description": f"Automatic annotations by {model}",
            "version": "1.0",
            "year": now.year,
            "date_created": now.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "licenses": [{"id": 1, "url": ""}],
        "categories": [{"id": 1, "name": "person", "supercategory": "person"}],
        "images": [],
        "annotations": [],
    }


def _draw_detections(image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    out = image.copy()
    for x1, y1, x2, y2, conf in boxes:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(out, (x1, y1), (x2, y2), (50, 200, 80), 2)
        label = f"{conf:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - lh - 8), (x1 + lw + 6, y1), (50, 200, 80), -1)
        cv2.putText(out, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    return out


def _save_json(path: Path, struct: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as file:
        json.dump(struct, file, indent=2, ensure_ascii=False)
    tmp.replace(path)


@click.command()
@click.option("--files", type=str, default="D:/NewPoseCustom/stable",
              help="Folder of images to annotate.")
@click.option("--output-dir", type=str, default="../annotations/detection/coco",
              help="Output COCO JSON path.")
@click.option("--yolo-model", type=str, default="yolo12x.pt",
              help="YOLO model weights (name or path).")
@click.option("--conf", type=float, default=0.25,
              show_default=True, help="Minimum detection confidence threshold.")
@click.option("--batch-size", type=int, default=100,
              show_default=True, help="Inference batch size.")
@click.option("--preview", is_flag=True, default=False,
              help="Show annotated preview window while processing.")
@click.option("--vis-delay", type=int, default=30,
              show_default=True, help="cv2.waitKey delay in ms (only with --preview).")
@click.option("--save-every", type=int, default=500,
              show_default=True, help="Save JSON incrementally every N images.")
@click.option("--device", type=str, default="0",
              show_default=True, help="Inference device: 'cpu', '0', '0,1', etc. Auto-detected if empty.")
def annotate(
    files: str,
    output_dir: str,
    yolo_model: str,
    conf: float,
    batch_size: int,
    preview: bool,
    vis_delay: int,
    save_every: int,
    device: str,
) -> None:
    if not device:
        device = "0" if torch.cuda.is_available() else "cpu"
    click.echo(f"Device: {device}")

    model = YOLO(yolo_model, verbose=False)
    out_path = Path(output_dir).joinpath(f"{Path(files).stem}.json")
    coco = _coco_info(Path(yolo_model).stem)
    image_ids = itertools.count(1)
    ann_ids = itertools.count(1)

    if preview:
        cv2.namedWindow("Annotated", cv2.WINDOW_GUI_EXPANDED)

    image_paths = list(_iter_images(files))
    click.echo(f"Found {len(image_paths)} images in {files!r}")

    for batch_start in tqdm(range(0, len(image_paths), batch_size),
                            desc="Batches", unit="batch", ncols=70):
        batch_paths = image_paths[batch_start : batch_start + batch_size]

        # Load images for this batch
        batch_images: list[np.ndarray] = []
        valid_paths: list[Path]  = []
        for p in batch_paths:
            img = cv2.imread(str(p))
            if img is None:
                click.echo(f"Cannot read image, skipping: {p}")
                continue
            batch_images.append(img)
            valid_paths.append(p)

        if not batch_images:
            continue

        results = model(
            batch_images,
            conf=conf,
            device=device,
            verbose=False,
            classes=[PERSON_CLASS_ID],
        )

        for img, path, result in zip(batch_images, valid_paths, results):
            img_id = next(image_ids)
            h, w = img.shape[:2]

            coco["images"].append({
                "id": img_id,
                "file_name": path.name,
                "width": w,
                "height": h,
            })

            boxes = result.boxes
            if boxes is not None and len(boxes):
                xyxy = boxes.xyxy.cpu().numpy() # (N, 4)
                confs = boxes.conf.cpu().numpy() # (N,)

                person_boxes = []
                for (x1, y1, x2, y2), score in zip(xyxy, confs):
                    bw, bh = float(x2 - x1), float(y2 - y1)
                    coco["annotations"].append({
                        "id": next(ann_ids),
                        "image_id": img_id,
                        "category_id": 1,
                        "bbox": [float(x1), float(y1), bw, bh],
                        "area": bw * bh,
                        "segmentation": [],
                        "iscrowd": 0,
                        "score": float(score),
                    })
                    person_boxes.append((x1, y1, x2, y2, score))

                if preview and person_boxes:
                    vis = _draw_detections(img, np.array(person_boxes))
                    cv2.imshow("Annotated", vis)
                    if cv2.waitKey(vis_delay) == ord("q"):
                        click.echo("Preview quit by user")
                        preview = False
                        cv2.destroyAllWindows()

        # Incremental save
        processed = batch_start + len(batch_paths)
        if processed % save_every < batch_size:
            _save_json(out_path, coco)
            click.echo(f"Incremental save at {processed} images → {out_path}")

    if preview:
        cv2.destroyAllWindows()

    _save_json(out_path, coco)

    click.echo(f"Annotation saved → {out_path}")
    click.echo(f"Total images: {len(coco['images'])}")
    click.echo(f"Total annotations: {len(coco['annotations'])}")


if __name__ == "__main__":
    annotate()
