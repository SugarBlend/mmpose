import json
import os
import re
import threading
from pathlib import Path
from typing import Any, List, Dict, Optional, Tuple, Union
import logging

import cv2
import numpy as np

from project.visualizer.models import FilterParams, ImageEntry


class DataService:
    logger = logging.getLogger("DataService")

    def __init__(self) -> None:
        self.all_entries: List[ImageEntry] = []
        self.filtered_entries: List[ImageEntry] = []
        self.total_anns: int = 0
        self._img_info: Dict[Union[str, int], Dict[str, Any]] = {}

    def load(self, path: str) -> Tuple[int, int]:
        with Path(path).open(mode="r", encoding="utf-8") as file:
            data = json.load(file)

        if "annotations" not in data:
            raise ValueError("Doesn't have key 'annotations'. Your file may not be in COCO format.")

        self._img_info = {sample["id"]: sample for sample in data.get("images", [])}
        groups: Dict[Union[str, int], ImageEntry] = {}
        for ann in data["annotations"]:
            image_id = ann["image_id"]
            if image_id not in groups:
                info = self._img_info.get(image_id, {})
                groups[image_id] = ImageEntry(image_id, info.get("file_name"))
            ann["file_name"] = groups[image_id].file_name
            groups[image_id].annotations.append(ann)

        self.all_entries = sorted(groups.values(), key=lambda e: e.image_id)
        self.total_anns = len(data["annotations"])
        return len(self.all_entries), self.total_anns

    def apply_filter(self, params: FilterParams) -> int:
        compiled: List[re.Pattern] = []

        for pattern in params.patterns:
            pattern = pattern.strip()
            if not pattern:
                self.logger.warning(f"Catch invalid regex pattern: '{pattern}'")
                continue
            try:
                compiled.append(re.compile(pattern))
            except re.error as exc:
                self.logger.debug(f"Caught exception: {exc}")
                continue

        if not compiled:
            self.filtered_entries = list(self.all_entries)
            return len(self.filtered_entries)

        ops = list(params.operators)

        def matches(filename: str) -> bool:
            result = bool(compiled[0].search(filename))
            for i, cp in enumerate(compiled[1:], 0):
                op = ops[i] if i < len(ops) else "OR"
                m = bool(cp.search(filename))
                result = (result and m) if op == "AND" else (result or m)
            return result

        self.filtered_entries = [e for e in self.all_entries if matches(e.file_name)]
        return len(self.filtered_entries)

    def get(self, i: int) -> Optional[ImageEntry]:
        return self.filtered_entries[i] if 0 <= i < len(self.filtered_entries) else None

    def __len__(self) -> int:
        return len(self.filtered_entries)


class TrackBuilder:
    def __init__(self) -> None:
        self._data: Dict[int, Dict[int, List[Tuple[int, int, int]]]] = {}

    def build(self, entries: List[ImageEntry]) -> None:
        self._data.clear()
        for fi, entry in enumerate(entries):
            for ai, ann in enumerate(entry.annotations):
                kpts: List[Union[int, float]] = ann.get('keypoints', [])
                if not kpts or len(kpts) % 3 != 0:
                    continue
                nk = len(kpts) // 3
                if ai not in self._data:
                    self._data[ai] = {}
                for ki in range(nk):
                    x, y, v = int(kpts[ki * 3]), int(kpts[ki * 3 + 1]), kpts[ki * 3 + 2]
                    if v > 0:
                        self._data[ai].setdefault(ki, []).append((fi, x, y))

    def get_window(self, frame_idx: int, length: int) -> Dict[int, Dict[int, List[Tuple[int, int, int]]]]:
        lo = frame_idx - length
        res: Dict[int, Dict[int, List[Tuple[int, int, int]]]] = {}
        for ai, kd in self._data.items():
            for ki, hist in kd.items():
                pts = [(fi, x, y) for fi, x, y in hist if lo <= fi <= frame_idx]
                if pts:
                    res.setdefault(ai, {})[ki] = pts
        return res


class ImageLoader:
    def __init__(self) -> None:
        self.directory: str = ""
        self._cache: Dict[str, np.ndarray] = {}
        self._order: List[str] = []
        self._limit: int = 32
        self._lock = threading.Lock()

    def set_directory(self, path: str) -> None:
        self.directory = path
        with self._lock:
            self._cache.clear()
            self._order.clear()

    def load(self, filename: str) -> Optional[np.ndarray]:
        with self._lock:
            if filename in self._cache:
                return self._cache[filename]

        if not self.directory:
            return None

        image_path = os.path.join(self.directory, Path(filename).name)
        if not os.path.exists(image_path):
            image_path = self._search(filename)

        if not image_path:
            return None

        img = cv2.imread(image_path)
        if img is None:
            return None

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        with self._lock:
            if len(self._order) >= self._limit:
                self._cache.pop(self._order.pop(0), None)
            self._cache[filename] = rgb
            self._order.append(filename)
        return rgb

    def _search(self, filename: str) -> Optional[str]:
        name = Path(filename).name
        for root, _, files in os.walk(self.directory):
            if name in files:
                return os.path.join(root, name)
        return None