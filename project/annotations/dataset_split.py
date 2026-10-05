import os
import re
import json
import hashlib
import logging
import argparse
from pathlib import Path
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)
logging.basicConfig(level=getattr(logging, os.getenv("LOG_LEVEL", default="INFO")),
                    format="[%(levelname)s] %(message)s")

SPLITS = ("train", "val", "test")
# on ties / conflicts prefer the split where leakage hurts the most
SPLIT_PRIORITY = {"test": 2, "val": 1, "train": 0}
FRAME_REGEX = re.compile(r"(\d+)\.[A-Za-z]+$")  # frame number at the end of the file name: .../000150.jpg


def load_coco_files(annotations_dir: Path, patterns: str) -> tuple[list[dict], list[dict], list[dict]]:
    compilers = [re.compile(p) for p in patterns.split(",") if p]
    json_files = sorted(
        (path for path in annotations_dir.glob("*.json") if any(c.search(path.name) for c in compilers)),
        key=lambda path: path.name,
    )
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in '{annotations_dir}' matching '{patterns}'")

    images: list[dict] = []
    annotations: list[dict] = []
    categories: list[dict] | None = None

    for json_file in json_files:
        data = json.loads(json_file.read_text(encoding="utf-8"))

        file_categories = data.get("categories") or []
        if categories is None and file_categories:
            categories = file_categories
        elif file_categories and file_categories != categories:
            logger.warning(f"{json_file.name}: categories differ from the first file, using the first ones")

        file_images = data.get("images", [])
        if not file_images:
            logger.warning(f"{json_file.name}: 0 images, skipped")
            continue

        seen_ids: set = set()
        for img in file_images:
            if img["id"] in seen_ids:
                logger.warning(f"{json_file.name}: duplicate image id {img['id']}, skipped")
                continue
            seen_ids.add(img["id"])
            images.append({**img, "_src": json_file.name, "_src_id": img["id"]})

        dropped = 0
        for ann in data.get("annotations", []):
            if ann["image_id"] not in seen_ids:
                dropped += 1
                continue
            annotations.append({**ann, "_src": json_file.name, "_src_image_id": ann["image_id"],
                                "_src_id": ann.get("id", -1)})
        if dropped:
            logger.warning(f"{json_file.name}: dropped {dropped} annotations with unknown image_id")

        logger.info(f"{json_file.name}: {len(seen_ids)} images")

    logger.info(f"Total merged: {len(images)} images, {len(annotations)} annotations from {len(json_files)} files")
    return images, annotations, categories or []


def reindex(images: list[dict], annotations: list[dict]) -> None:
    images.sort(key=lambda i: (i["file_name"], i["_src"], str(i["_src_id"])))
    id_map: dict[tuple, int] = {}
    for new_id, img in enumerate(images, start=1):
        id_map[(img["_src"], img["_src_id"])] = new_id
        img["id"] = new_id

    for ann in annotations:
        ann["image_id"] = id_map[(ann["_src"], ann["_src_image_id"])]
    annotations.sort(key=lambda a: (a["image_id"], a["_src"], str(a["_src_id"])))
    for new_id, ann in enumerate(annotations, start=1):
        ann["id"] = new_id


def group_key(file_name: str, group_regex: re.Pattern | None, chunk_frames: int = 0) -> str:
    # Distribute the data into groups. These groups will subsequently be split into samples within the manifest.
    # Chunk-based separation is supported to maximize the temporal spacing between samples (which are essentially
    # consecutive frames) and thereby prevent data leakage.
    if group_regex is None:
        return file_name

    match = group_regex.search(file_name)
    if not match:
        return file_name

    key = match.group(1) if match.groups() else match.group(0)
    if chunk_frames > 0:
        frame_match = FRAME_REGEX.search(file_name)
        if frame_match:
            key = f"{key}#chunk{int(frame_match.group(1)) // chunk_frames}"
    return key


def hash_split(key: str, salt: int, val_ratio: float, test_ratio: float) -> str:
    h = int(hashlib.sha1(f"{salt}:{key}".encode("utf-8")).hexdigest()[:15], 16) / 16 ** 15  # [0, 1)
    return  "test" if h < test_ratio else "val" if h < test_ratio + val_ratio else "train"


def load_manifest(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    data = json.loads(path.read_text(encoding="utf-8"))
    assignments: dict[str, str] = data.get("assignments", data)
    bad = {k: v for k, v in assignments.items() if v not in SPLITS}
    if bad:
        raise ValueError(f"Manifest {path} has unknown splits: {list(bad.items())[:5]}")

    return assignments


def save_manifest(path: Path, manifest: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"assignments": dict(sorted(manifest.items()))}
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def assign_splits(
    images: list[dict],
    manifest: dict[str, str],
    group_regex: re.Pattern | None,
    salt: int,
    val_ratio: float,
    test_ratio: float,
    chunk_frames: int = 0,
    stratify: bool = True,
) -> Counter:
    files_by_group: dict[str, set[str]] = defaultdict(set)
    for img in images:
        files_by_group[group_key(img["file_name"], group_regex, chunk_frames)].add(img["file_name"])

    # splits already used by each group (a new frame of a known video follows its video)
    existing_by_group: dict[str, Counter] = defaultdict(Counter)
    for fn, split in manifest.items():
        existing_by_group[group_key(fn, group_regex, chunk_frames)][split] += 1

    added: Counter = Counter()
    mixed_groups = 0
    targets: dict[str, str] = {}
    new_groups: list[str] = []
    for group in sorted(files_by_group):
        existing = existing_by_group.get(group)
        if existing:
            if len(existing) > 1:
                mixed_groups += 1
            targets[group] = max(existing.items(), key=lambda kv: (kv[1], SPLIT_PRIORITY[kv[0]]))[0]
        else:
            new_groups.append(group)

    if stratify and chunk_frames > 0:
        # Every recording gets its own ~test/val share: new chunks of one recording are ordered by
        # their hash and filled test -> val -> train by frame count. No recording can fall entirely
        # into one split just by chance, while chunks themselves stay intact.
        by_recording: dict[str, list[str]] = defaultdict(list)
        for group in new_groups:
            by_recording[group.split("#chunk")[0]].append(group)

        for rec_groups in by_recording.values():
            # ability to change the hash by altering the seed, sorting by deterministic hash
            rec_groups.sort(key=lambda gr_name: hashlib.sha1(f"{salt}:{gr_name}".encode("utf-8")).hexdigest())
            samples = sum(len(files_by_group[g]) for g in rec_groups)
            accumulation: float = 0
            for group in rec_groups:
                size = len(files_by_group[group])
                pos = (accumulation + size / 2) / samples  # chunk midpoint in [0, 1)
                targets[group] = "test" if pos < test_ratio else "val" if pos < test_ratio + val_ratio else "train"
                accumulation += size
    else:
        for group in new_groups:
            targets[group] = hash_split(group, salt, val_ratio, test_ratio)

    for group, files in sorted(files_by_group.items()):
        target = targets[group]
        for fn in sorted(files):
            if fn not in manifest:
                manifest[fn] = target
                added[target] += 1

    if mixed_groups:
        logger.warning(f"{mixed_groups} groups are already spread over several splits in the manifest "
                       f"(legacy leakage); their new images follow the majority split")
    return added


def build_coco_doc(images: list[dict], annotations: list[dict], categories: list[dict], split: str) -> dict:
    # no timestamps: identical input must give an identical file (stable md5 for DVC / MLflow)
    return {
        "info": {"description": f"Merged Label Studio annotations – {split}", "contributor": "coco_split.py"},
        "categories": categories,
        "images": [{k: v for k, v in img.items() if not k.startswith("_")} for img in images],
        "annotations": [{k: v for k, v in ann.items() if not k.startswith("_")} for ann in annotations],
    }


def save_splits(
    images: list[dict],
    annotations: list[dict],
    categories: list[dict],
    manifest: dict[str, str],
    output_dir: Path,
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)

    split_images: dict[str, list[dict]] = {s: [] for s in SPLITS}
    split_of_image: dict[int, str] = {}
    for img in images:
        split = manifest[img["file_name"]]
        split_images[split].append(img)
        split_of_image[img["id"]] = split

    split_anns: dict[str, list[dict]] = {s: [] for s in SPLITS}
    for ann in annotations:
        split_anns[split_of_image[ann["image_id"]]].append(ann)

    counts: dict[str, int] = {}
    for split in SPLITS:
        doc = build_coco_doc(split_images[split], split_anns[split], categories, split)
        out_path = output_dir / f"{split}.json"
        out_path.write_text(json.dumps(doc), encoding="utf-8")
        counts[split] = len(split_images[split])
        logger.info(f"Saved {out_path} ({len(split_images[split])} images, {len(split_anns[split])} annotations)")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge COCO JSON files and split into train / val / test with a stable, manifest-based assignment"
    )
    parser.add_argument("--annotations_dir", nargs="+", default=["./labelstudio_anns/halpe26"],
                        help="Directory(ies) with COCO JSON files. Several dirs (halpe26 + halpe136) "
                             "only together with --assign_only")
    parser.add_argument("--output_dir", default="./custom_anns/halpe26",
                        help="Where to save train.json / val.json / test.json")
    parser.add_argument("--manifest", default="split_manifest.json",
                        help="Path to split manifest (file_name -> split).")
    parser.add_argument("--assign_only", action="store_true",
                        help="Only add new images of ALL --annotations_dir to the shared manifest, write no splits")
    parser.add_argument("--frozen_manifest", action="store_true",
                        help="Read-only manifest: never add images, fail if an image is missing in it")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42, help="Hash salt for NEW images")
    parser.add_argument("--patterns", default="^Pose Annotation*,^Foots Pose Annotation*,^Outsource*",
                        help="Comma-separated regexes for annotation file names")
    parser.add_argument("--group_regex", default="^([^/]+/[^/]+)/",
                        help="Regex applied to file_name; group 1 (or the whole match) is the group key. "
                             "All images of one group go to one split. Example for 'videos/<video>/frame.jpg': "
                             r"'^(videos/[^/]+)/'. Default: each image is its own group")
    parser.add_argument("--chunk_frames", type=int, default=30,
                        help="Split every recording (group_regex) into chunks of N frame numbers; chunks are "
                             "assigned to splits independently. Frame number = last digits in the file name. "
                             "Frame numbers are indices of EXTRACTED frames (00000, 00001, ...), so at 6 fps "
                             "30 = 5 s. 0 = whole recording is one group")
    parser.add_argument("--no_stratify", action="store_true",
                        help="With --chunk_frames: assign chunks by pure hash instead of giving every "
                             "recording its own train/val/test share")
    parser.add_argument("--split_mode", choices=["hash"], default="hash",
                        help="Kept for compatibility with dvc.yaml; only 'hash' is supported")
    args = parser.parse_args()

    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        parser.error("Ratios must sum to 1.0")

    annotation_dirs = [Path(d) for d in args.annotations_dir]
    for d in annotation_dirs:
        if not d.is_dir():
            parser.error(f"Not a directory: {d}")
    if args.assign_only and args.frozen_manifest:
        parser.error("--assign_only and --frozen_manifest are mutually exclusive")
    if not args.assign_only and len(annotation_dirs) > 1:
        parser.error("Several --annotations_dir are only supported with --assign_only")

    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest)
    group_regex = re.compile(args.group_regex) if args.group_regex else None

    # read the old split BEFORE anything is overwritten (bootstrap_from may equal output_dir)
    if manifest_path.exists():
        manifest = load_manifest(manifest_path)
        logger.info(f"Manifest: {len(manifest)} known images from {manifest_path}")
    else:
        manifest = {}
        logger.info("Manifest not found, starting a new one (all images are new)")

    if args.frozen_manifest and not manifest_path.exists():
        parser.error(f"--frozen_manifest: {manifest_path} does not exist")

    # shared manifest for several annotation sets: one image -> one split in every set
    if args.assign_only:
        all_images: list[dict] = []
        for d in annotation_dirs:
            imgs, _, _ = load_coco_files(d, args.patterns)
            all_images.extend(imgs)
        before = len(manifest)
        added = assign_splits(all_images, manifest, group_regex, args.seed, args.val_ratio, args.test_ratio,
                              args.chunk_frames, not args.no_stratify)
        save_manifest(manifest_path, manifest)
        logger.info(f"Manifest {manifest_path}: {before} -> {len(manifest)} images; new: "
                    + (", ".join(f"{s}={added[s]}" for s in SPLITS) if added else "none"))
        return

    images, annotations, categories = load_coco_files(annotation_dirs[0], args.patterns)
    reindex(images, annotations)

    if args.frozen_manifest:
        unknown = sorted({img["file_name"] for img in images} - set(manifest))
        if unknown:
            raise SystemExit(f"{len(unknown)} images are not in the frozen manifest {manifest_path} "
                             f"(run the assign stage first), e.g. {unknown[:3]}")
        added = Counter()
    else:
        added = assign_splits(images, manifest, group_regex, args.seed, args.val_ratio, args.test_ratio,
                              args.chunk_frames, not args.no_stratify)
    present = {img["file_name"] for img in images}
    missing = sum(1 for fn in manifest if fn not in present)

    counts = save_splits(images, annotations, categories, manifest, output_dir)
    if not args.frozen_manifest:
        save_manifest(manifest_path, manifest)

    total = sum(counts.values()) or 1
    logger.info("New images: " + (", ".join(f"{s}={added[s]}" for s in SPLITS) if added else "none"))
    logger.info("Actual ratio: " + " / ".join(f"{s} {counts[s] / total:.1%}" for s in SPLITS))
    if missing and not args.frozen_manifest:
        logger.warning(f"{missing} images from the manifest are absent in current annotations "
                       f"(kept in the manifest: if they come back, they return to the same split)")


if __name__ == "__main__":
    main()