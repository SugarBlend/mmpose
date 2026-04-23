import argparse
import sys
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image, ImageTk
from mmpose.structures import PoseDataSample
from pycocotools.coco import COCO
from tqdm import tqdm

from deploy2serve.deployment.projects.sapiens.utils.palettes import COCO_WHOLEBODY_SKELETON_INFO

sys.path.insert(0, Path(__file__).parents[2].as_posix())
from project.label_studio.palettes import COCO_HALPE26_SKELETON_INFO
from project.label_studio.pipelines.pipeline import MMPipeline
from config import EvalConfig
from overlay import (
    MODEL_JOINT_COLORS, MODEL_LIMB_COLORS,
    draw_skeleton_pretty, render_hud,
)
from mmpose.utils.logger import MMLogger


logger = MMLogger.get_instance("Comparison")


def _skeleton_pairs(info: dict) -> list[tuple[int, int]]:
    return [(v["link"][0], v["link"][1]) for v in info.values()]


SKELETON_META = {
    26: _skeleton_pairs(COCO_HALPE26_SKELETON_INFO),
    133: _skeleton_pairs(COCO_WHOLEBODY_SKELETON_INFO),
}


@dataclass
class ModelVisState:
    pipeline: MMPipeline
    legend: str
    color: tuple[float, float, float]
    limb_color: tuple[float, float, float]
    skeleton: list[tuple[int, int]]


@dataclass
class FrameCache:
    """Pre-rendered BGR frames ready for display."""
    frames: list[np.ndarray] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.frames[idx]


def _build_model_states(cfg: EvalConfig) -> list[ModelVisState]:
    first_pipeline = MMPipeline(
        pose_checkpoint=cfg.models[0].model_path,
        pose_config=cfg.models[0].config_path,
    )
    num_joints = first_pipeline.model_cfg.num_keypoints

    if num_joints not in SKELETON_META:
        raise ValueError(
            f"No skeleton metadata for {num_joints} joints. "
            f"Supported: {list(SKELETON_META)}"
        )

    skeleton = SKELETON_META[num_joints]
    states: list[ModelVisState] = []

    for idx, model_cfg in enumerate(cfg.models):
        pipeline = first_pipeline if idx == 0 else MMPipeline(
            pose_checkpoint=model_cfg.model_path,
            pose_config=model_cfg.config_path,
        )
        color = MODEL_JOINT_COLORS[idx % len(MODEL_JOINT_COLORS)]
        limb_color = MODEL_LIMB_COLORS[idx % len(MODEL_LIMB_COLORS)]
        states.append(ModelVisState(pipeline, model_cfg.legend, color, limb_color, skeleton))
        logger.info(f"[{idx}] {model_cfg.legend}")

    return states


# def _infer(
#     states: list[ModelVisState],
#     image_path: str,
#     bboxes: list,
# ) -> list[list[PoseDataSample]]:
#     results: list[Optional[list[PoseDataSample]]] = [None] * len(states)
#
#     if torch.cuda.is_available() and len(states) > 1:
#         streams = [torch.cuda.Stream() for _ in states]
#         default = torch.cuda.default_stream()
#         for idx, (state, stream) in enumerate(zip(states, streams)):
#             stream.wait_stream(default)
#             with torch.cuda.stream(stream):
#                 results[idx] = state.pipeline(image_path, bboxes)
#         for stream in streams:
#             default.wait_stream(stream)
#     else:
#         for idx, state in enumerate(states):
#             results[idx] = state.pipeline(image_path, bboxes)
#
#     return results


def _infer(
    states: list[ModelVisState],
    image_path: str,
    bboxes: list,
) -> list[list[PoseDataSample]]:
    results: list[Optional[list[PoseDataSample]]] = [None] * len(states)

    for idx, state in enumerate(states):
        results[idx] = state.pipeline(image_path, bboxes)

    return results


def preprocess_all_frames(cfg: EvalConfig) -> FrameCache:
    """
    Run inference on every image upfront and render each frame into a BGR
    numpy array.  All results are stored in a FrameCache object.
    """
    logger.info("Loading models…")
    states = _build_model_states(cfg)

    ann_file = cfg.models[0].ann_file
    dataset_path = cfg.models[0].dataset_folder

    coco = COCO(ann_file)
    cat_ids = coco.getCatIds(catNms=["person"])
    img_ids = coco.getImgIds(catIds=cat_ids)

    legends = [s.legend for s in states]
    joint_colors = [s.color for s in states]
    limb_colors = [s.limb_color for s in states]
    total = len(img_ids)

    cache = FrameCache()

    logger.info(f"Pre-processing {total} images…")
    for frame_idx, image_id in enumerate(tqdm(img_ids, desc="Pre-processing"), start=1):
        for instance in coco.loadImgs(image_id):
            bboxes = [ann["bbox"] for ann in coco.imgToAnns.get(image_id, [])]
            if not bboxes:
                continue

            image_path = f"{dataset_path}/{instance['file_name']}"
            image = cv2.imread(image_path)
            if image is None:
                logger.warning(f"Cannot read: {image_path}")
                continue

            all_results = _infer(states, image_path, bboxes)

            for state, samples in zip(states, all_results):
                for sample in samples:
                    kpts = sample.pred_instances.keypoints[0]
                    draw_skeleton_pretty(
                        image, kpts, state.skeleton,
                        state.color, state.limb_color,
                    )

            image = render_hud(image, legends, joint_colors, limb_colors, frame_idx, total)
            cache.frames.append(image)

    logger.info(f"Pre-processing done. Cached {len(cache)} frames.")
    return cache


class PosePlayer(object):
    """
    Tkinter-based frame player with progress bar and navigation controls.

    Keyboard shortcuts
    ------------------
    ← - previous frame
    → - next frame
    Shift + ← - jump back  10 frames
    Shift + → - jump forward 10 frames
    Home - first frame
    End - last frame
    Space - toggle auto-play
    Esc - quit
    """

    AUTOPLAY_DELAY_MS: int = 16  # ms between frames during auto-play
    JUMP_STEP: int = 10  # frames to skip with Shift+arrow

    # Slider colours
    _SLIDER_BG = "#acbce6"
    _SLIDER_TROUGH = "#41427a"
    _SLIDER_FG = "#6578a8"
    _PAD: int = 16  # symmetric padding around the image in pixels

    def __init__(self, cache: FrameCache, title: str = "Pose Comparison Player") -> None:
        if len(cache) == 0:
            raise ValueError("FrameCache is empty – nothing to display.")

        self._cache = cache
        self._index: int = 0
        self._playing: bool = False
        self._after_id: Optional[str] = None
        # PhotoImage cache: invalidated when canvas size changes
        self._photo_cache: list[Optional[ImageTk.PhotoImage]] = [None] * len(cache)
        self._photo_cache_size: tuple[int, int] = (0, 0)  # (cw, ch) the cache was built for
        self._resize_after_id: Optional[str] = None  # debounce resize events

        # root window
        self._root = tk.Tk()
        self._root.title(title)
        self._root.configure(bg="#1a1a1a")
        self._root.resizable(True, True)

        # Constrain max size to the full screen; start as a small window
        sw = self._root.winfo_screenwidth()
        sh = self._root.winfo_screenheight()
        self._root.maxsize(sw, sh)
        self._root.minsize(640, 480)
        # Initial size: 30 % of screen, centred
        init_w = max(1280, sw // 3)
        init_h = max(720, sh // 3)
        x0 = (sw - init_w) // 2
        y0 = (sh - init_h) // 2
        self._root.geometry(f"{init_w}x{init_h}+{x0}+{y0}")

        # canvas
        self._canvas = tk.Canvas(self._root, bg="#e8e8e1", highlightthickness=1)
        self._canvas.pack(fill=tk.BOTH, expand=True)
        # anchor=center so we only need to update the (cx, cy) coordinate
        self._img_id = self._canvas.create_image(0, 0, anchor=tk.CENTER)
        self._current_photo: Optional[ImageTk.PhotoImage] = None

        # slider
        self._slider_var = tk.IntVar(value=0)
        self._slider = tk.Scale(
            self._root,
            variable=self._slider_var,
            from_=0,
            to=max(0, len(cache) - 1),
            orient=tk.HORIZONTAL,
            showvalue=False,
            sliderlength=16,
            width=8,
            bg=self._SLIDER_BG,
            troughcolor=self._SLIDER_TROUGH,
            fg=self._SLIDER_FG,
            activebackground=self._SLIDER_FG,
            highlightthickness=0,
            bd=0,
            cursor="hand2",
            command=self._on_slider_move,
        )
        self._slider.pack(fill=tk.X, side=tk.BOTTOM, padx=4, pady=2)
        self._slider_updating: bool = False   # guard against feedback loops

        # control bar
        ctrl = tk.Frame(self._root, bg="#41427a", pady=5)
        ctrl.pack(fill=tk.X, side=tk.BOTTOM)

        btn_cfg = dict(
            bg="#282852", fg="#e0e0e0",
            activebackground="#505050", activeforeground="#ffffff",
            pady=3, padx=3,
        )

        tk.Button(ctrl, text="⏮️", command=self._go_first, **btn_cfg).pack(side=tk.LEFT, padx=(10, 2))
        tk.Button(ctrl, text="⏪", command=self._prev, **btn_cfg).pack(side=tk.LEFT, padx=2)
        self._btn_play = tk.Button(ctrl, text="▶️", command=self._toggle_play, **btn_cfg)
        self._btn_play.pack(side=tk.LEFT, padx=2)
        tk.Button(ctrl, text="⏩", command=self._next, **btn_cfg).pack(side=tk.LEFT, padx=2)
        tk.Button(ctrl, text="⏩", command=self._go_last, **btn_cfg).pack(side=tk.LEFT, padx=2)

        # frame counter
        self._label = tk.Label(
            ctrl, text=self._frame_label(),
            fg="#e0e0e0", bg="#41427a",
            font=("SegoeUI", 10), width=14,
        )
        self._label.pack(side=tk.RIGHT, padx=12)

        # key bindings
        self._root.bind("<Left>", lambda _: self._prev())
        self._root.bind("<Right>", lambda _: self._next())
        self._root.bind("<Shift-Left>", lambda _: self._jump_back())
        self._root.bind("<Shift-Right>", lambda _: self._jump_fwd())
        self._root.bind("<Home>", lambda _: self._go_first())
        self._root.bind("<End>", lambda _: self._go_last())
        self._root.bind("<space>", lambda _: self._toggle_play())
        self._root.bind("<Escape>", lambda _: self._quit())

        self._canvas.bind("<Configure>",  self._on_canvas_resize)
        self._root.protocol("WM_DELETE_WINDOW", self._quit)

        # Draw first frame once window is ready
        self._root.after(50, self._refresh)

    def _render_frame(self, bgr: np.ndarray, canvas_w: int, canvas_h: int) -> ImageTk.PhotoImage:
        """Scale *bgr* to fit inside the padded canvas area, keeping aspect ratio."""
        src_h, src_w = bgr.shape[:2]
        avail_w = max(1, canvas_w - 2 * self._PAD)
        avail_h = max(1, canvas_h - 2 * self._PAD)
        scale = min(avail_w / src_w, avail_h / src_h)
        nw = max(1, int(src_w * scale))
        nh = max(1, int(src_h * scale))
        resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return ImageTk.PhotoImage(Image.fromarray(rgb))

    def _get_photo(self, idx: int, cw: int, ch: int) -> ImageTk.PhotoImage:
        """Return cached PhotoImage for *idx*, re-rendering if size changed."""
        if self._photo_cache_size != (cw, ch):
            # Canvas was resized — drop entire cache
            self._photo_cache = [None] * len(self._cache)
            self._photo_cache_size = (cw, ch)

        if self._photo_cache[idx] is None:
            self._photo_cache[idx] = self._render_frame(self._cache[idx], cw, ch)

        return self._photo_cache[idx]

    def _warm_next(self, idx: int, cw: int, ch: int) -> None:
        """Pre-render the next frame in the background after a short delay."""
        nxt = idx + 1
        if nxt < len(self._cache) and self._photo_cache[nxt] is None:
            self._photo_cache[nxt] = self._render_frame(self._cache[nxt], cw, ch)

    def _frame_label(self) -> str:
        return f"{self._index + 1:>5} / {len(self._cache)}"

    def _refresh(self) -> None:
        """Redraw image centred on canvas and sync slider position."""
        cw = max(1, self._canvas.winfo_width())
        ch = max(1, self._canvas.winfo_height())
        photo = self._get_photo(self._index, cw, ch)
        self._current_photo = photo
        self._canvas.coords(self._img_id, cw // 2, ch // 2)
        self._canvas.itemconfig(self._img_id, image=photo)
        self._label.config(text=self._frame_label())
        self._slider_updating = True
        self._slider_var.set(self._index)
        self._slider_updating = False
        # Pre-render next frame while idle
        self._root.after_idle(self._warm_next, self._index, cw, ch)

    def _on_slider_move(self, value: str) -> None:
        """Called by tk.Scale when user drags the slider."""
        if self._slider_updating:
            return

        idx = int(value)
        if idx != self._index:
            self._index = idx
            self._refresh()

    def _on_canvas_resize(self, event: tk.Event) -> None:
        # Debounce: wait 80 ms after last resize event before re-rendering
        if self._resize_after_id is not None:
            self._root.after_cancel(self._resize_after_id)
        self._resize_after_id = self._root.after(80, self._refresh)

    def _go_first(self) -> None:
        self._index = 0
        self._refresh()

    def _go_last(self) -> None:
        self._index = len(self._cache) - 1
        self._refresh()

    def _prev(self) -> None:
        if self._index > 0:
            self._index -= 1
            self._refresh()

    def _next(self) -> None:
        if self._index < len(self._cache) - 1:
            self._index += 1
            self._refresh()

    def _jump_back(self) -> None:
        self._index = max(0, self._index - self.JUMP_STEP)
        self._refresh()

    def _jump_fwd(self) -> None:
        self._index = min(len(self._cache) - 1, self._index + self.JUMP_STEP)
        self._refresh()

    def _toggle_play(self) -> None:
        self._playing = not self._playing
        self._btn_play.config(text="⏸️" if self._playing else "▶️")
        if self._playing:
            self._autoplay()

    def _autoplay(self) -> None:
        if not self._playing:
            return

        if self._index < len(self._cache) - 1:
            self._index += 1
            self._refresh()
            self._after_id = self._root.after(self.AUTOPLAY_DELAY_MS, self._autoplay)
        else:
            self._playing = False
            self._btn_play.config(text="▶️")

    def _quit(self) -> None:
        if self._after_id is not None:
            self._root.after_cancel(self._after_id)
        self._root.destroy()

    def run(self) -> None:
        self._root.mainloop()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visual pose model comparison (tkinter player)")
    parser.add_argument("--config", "-c", type=str, required=True,
                        help="Path to eval-config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = EvalConfig.load(args.config)
    frame_cache = preprocess_all_frames(cfg)
    player = PosePlayer(frame_cache, title="Pose Comparison Player")
    player.run()
