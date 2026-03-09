from pathlib import Path
from typing import List, Optional

import cv2
from PIL import ImageTk
import numpy as np

from project.visualizer.drawing import (
    resize,
    draw_frame,
    fit_image,
    outlined,
    rgb_to_photoimage,
)
from project.visualizer.models import DrawParams, FilterParams, ViewGroupConfig
from project.visualizer.services import DataService, ImageLoader, TrackBuilder


class ViewGroup:
    def __init__(self, cfg: ViewGroupConfig, shared_loader: ImageLoader) -> None:
        self.cfg: ViewGroupConfig = cfg
        self.loader: ImageLoader = shared_loader
        self.data: DataService = DataService()
        self.tracker: TrackBuilder = TrackBuilder()
        self._idx: int = 0
        self._loaded: bool = False
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._last_cw: int = 0
        self._last_ch: int = 0

    def load_from(self, all_entries: List, total_ann: int) -> None:
        self.data.all_entries = all_entries
        self.data.total_ann = total_ann
        self._apply_filter()
        self._loaded = True

    def _apply_filter(self) -> None:
        params: FilterParams = FilterParams(
            patterns=self.cfg.patterns,
            operators=self.cfg.operators
        )
        self.data.apply_filter(params)
        self.tracker.build(self.data.filtered_entries)
        self._idx = 0

    def total(self) -> int:
        return len(self.data)

    def current_entry(self):
        return self.data.get(self._idx)

    def set_idx(self, i: int) -> None:
        self._idx = self._clamp_index(i)

    def sync_ratio(self, ratio: float) -> None:
        t = self.total()
        if t:
            self._idx = self._clamp_index(int(ratio * t))

    def _clamp_index(self, i: int) -> int:
        t = self.total()
        return max(0, min(i, t - 1)) if t else 0

    def render(self, cw: int, ch: int, dp: DrawParams) -> Optional[ImageTk.PhotoImage]:
        entry = self.current_entry()
        if entry is None:
            return None

        img: Optional[np.ndarray] = self.loader.load(entry.file_name)
        if img is None:
            return None

        img = img.copy()
        oh, ow = img.shape[:2]

        tw: Optional[List] = (
            self.tracker.get_window(self._idx, dp.track_length) if dp.show_tracks else None
        )
        draw_frame(img, entry.annotations, dp, tw)

        if dp.show_frame_label:
            try:
                fn: int = int(Path(entry.file_name).stem)
                outlined(img, f"ID={fn + 1}", (10, 34),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.75, (120, 210, 255), 2)
            except ValueError:
                pass
            outlined(img, self.cfg.name, (10, oh - 10),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 80), 1)

        nw, nh = fit_image(ow, oh, cw, ch)
        resized: np.ndarray = resize(img, nw, nh)
        photo: ImageTk.PhotoImage = rgb_to_photoimage(resized)

        self._photo = photo
        self._last_cw, self._last_ch = cw, ch

        return photo
