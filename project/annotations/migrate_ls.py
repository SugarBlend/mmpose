import json
import argparse
import os
from pathlib import Path
import logging


logger = logging.getLogger(__name__)
logging.basicConfig(level=getattr(logging, os.getenv("LOG_LEVEL", default="INFO")),
                    format="[%(levelname)s] %(message)s")


ANN_KEEP = ("result", "was_cancelled", "ground_truth", "lead_time", "created_at")
PRED_KEEP = ("result", "score", "model_version")


def clean_task(task: dict, completed_by: int | None, keep_predictions: bool) -> dict:
    out = {"data": task["data"]}
    if task.get("meta"):
        out["meta"] = task["meta"]

    anns = []
    for ann in task.get("annotations", []):
        a = {field: ann[field] for field in ANN_KEEP if ann.get(field) is not None}
        if completed_by is not None:
            a["completed_by"] = completed_by
        anns.append(a)
    if anns:
        out["annotations"] = anns

    if keep_predictions:
        # exports made with "only_id" contain plain ints here: nothing portable to keep
        preds = [{k: p[k] for k in PRED_KEEP if p.get(k) is not None}
                 for p in task.get("predictions", []) if isinstance(p, dict)]
        if preds:
            out["predictions"] = preds
    return out


def migrate_labels(path: Path, completed_by: int, no_predictions: bool = False) -> None:
    tasks = json.loads(path.read_text(encoding="utf-8"))
    cleaned = [clean_task(task, completed_by, not no_predictions) for task in tasks]

    anns = sum(len(task.get("annotations", [])) for task in cleaned)
    preds = sum(len(task.get("predictions", [])) for task in cleaned)
    path.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"{len(cleaned)} tasks, {anns} annotations, {preds} predictions")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, type=str)
    parser.add_argument("--completed_by", type=int, default=1,
                    help="The user identifier in the migration service that will be marked as having performed the "
                         "operation. Default 1: owner")
    parser.add_argument("--no_predictions", action="store_true", help="Drop predictions")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    src = Path(args.src)
    if src.is_dir():
        for file in src.glob("*.json"):
            migrate_labels(file, args.completed_by, args.no_predictions)
    elif src.is_file():
        migrate_labels(src, args.completed_by, args.no_predictions)
