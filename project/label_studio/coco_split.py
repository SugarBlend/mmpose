import json
import random
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def load_coco_files(annotations_dir: Path) -> tuple[list[dict], list[dict], list[dict]]:
    json_files = sorted(annotations_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {annotations_dir}")

    all_images: list[dict] = []
    all_annotations: list[dict] = []
    categories: list[dict] = []

    # Global counters — every file continues from where the previous one left off
    next_image_id = 1
    next_ann_id = 1

    for json_file in json_files:
        logger.info(f"Loading: {json_file.name}")
        data = json.loads(json_file.read_text(encoding="utf-8"))

        if not categories and data.get("categories"):
            categories = data["categories"]

        file_images: list[dict] = data.get("images", [])
        file_annotations: list[dict] = data.get("annotations", [])

        if not file_images:
            logger.info("  → 0 images, 0 annotations (skipped)")
            continue

        # Build a local old -> new map for this file only
        old_to_new: dict[int, int] = {}
        new_images: list[dict] = []
        for img in file_images:
            old_to_new[img["id"]] = next_image_id
            new_images.append({**img, "id": next_image_id})
            next_image_id += 1

        new_annotations: list[dict] = []
        dropped = 0
        for ann in file_annotations:
            new_img_id = old_to_new.get(ann["image_id"])
            if new_img_id is None:
                dropped += 1
                continue
            new_annotations.append({**ann, "id": next_ann_id, "image_id": new_img_id})
            next_ann_id += 1

        if dropped:
            logger.warning(f"  ! Dropped {dropped} annotations with unknown image_id")

        all_images.extend(new_images)
        all_annotations.extend(new_annotations)

        logger.info(
            f"  → {len(new_images)} images, {len(new_annotations)} annotations"
        )

    logger.info(
        f"Total merged: {len(all_images)} images, {len(all_annotations)} annotations"
    )
    return all_images, all_annotations, categories


def split_images(
    images: list[dict],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    shuffled = images[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train: n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }

    for name, imgs in splits.items():
        logger.info(f"Split  {name:>5}: {len(imgs)} images")

    return splits


def build_coco_doc(
    images: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    categories: list[dict[str, Any]],
    split_name: str,
) -> dict[str, Any]:
    return {
        "info": {
            "year": datetime.now().year,
            "version": "1.0",
            "description": f"Mixed COCO dataset – {split_name}",
            "contributor": "coco_split.py",
            "date_created": str(datetime.now()),
        },
        "categories": categories,
        "images": images,
        "annotations": annotations,
    }


def save_splits(
    splits: dict[str, list[dict]],
    all_annotations: list[dict],
    categories: list[dict],
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    ann_by_image: dict[int, list[dict]] = {}
    for ann in all_annotations:
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    saved: dict[str, Path] = {}
    for split_name, split_images in splits.items():
        image_ids = {img["id"] for img in split_images}
        split_anns = [
            ann
            for img_id in image_ids
            for ann in ann_by_image.get(img_id, [])
        ]

        doc = build_coco_doc(split_images, split_anns, categories, split_name)
        out_path = output_dir / f"{split_name}.json"
        out_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        logger.info(
            f"Saved {out_path}  "
            f"({len(split_images)} images, {len(split_anns)} annotations)"
        )
        saved[split_name] = out_path

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge COCO JSON files and split into train / val / test"
    )
    parser.add_argument(
        "annotations_dir",
        help="Directory containing COCO JSON annotation files to mix",
    )
    parser.add_argument(
        "--output_dir",
        default="split_output",
        help="Directory to save train.json / val.json / test.json (default: split_output)",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Fraction of images for train (default: 0.7)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Fraction of images for val (default: 0.2)",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Fraction of images for test (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible shuffle (default: 42)",
    )
    args = parser.parse_args()

    # Validate ratios
    total = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total - 1.0) > 1e-6:
        parser.error(f"Ratios must sum to 1.0, got {total:.4f}")

    annotations_dir = Path(args.annotations_dir)
    if not annotations_dir.is_dir():
        parser.error(f"Not a directory: {annotations_dir}")

    # Run pipeline
    images, annotations, categories = load_coco_files(annotations_dir)
    # IDs are already globally unique after load_coco_files — no separate reindex needed
    splits = split_images(images, args.train_ratio, args.val_ratio, args.seed)
    save_splits(splits, annotations, categories, Path(args.output_dir))


if __name__ == "__main__":
    main()
