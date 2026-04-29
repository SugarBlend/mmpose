"""
pose_comparison_player.py — visual pose model comparison with Tkinter UI.

New features vs original:
- Toolbar with Settings button (dropdown popover).
- Settings panel:
    • Checkboxes: Enable Face / Enable Hands / Enable Feet / Enable Body
    • Sliders: Joint Size (radius 1-15 px) and Connection Thickness (1-12 px)
    • Side-based colouring toggle: colour joints/limbs by left / right / center side.
- All settings apply live on the next frame render (no pre-processing required).
- Supports COCO-17, HALPE-26, HALPE-136, COCO-Wholebody-133 formats.
"""
from __future__ import annotations

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
    MODEL_JOINT_COLORS, MODEL_LIMB_COLORS, MODEL_SIDE_COLORS,
    BodyPartFilter, body_filter,
    draw_skeleton_pretty,
)
from mmpose.utils.logger import MMLogger


logger = MMLogger.get_instance("Comparison")


# ──────────────────────────────────────────────────────────────────────────────
# Skeleton metadata
# ──────────────────────────────────────────────────────────────────────────────

def _skeleton_pairs(info: dict) -> list[tuple[int, int]]:
    return [(v["link"][0], v["link"][1]) for v in info.values()]


SKELETON_META: dict[int, list[tuple[int, int]]] = {
    26:  _skeleton_pairs(COCO_HALPE26_SKELETON_INFO),
    133: _skeleton_pairs(COCO_WHOLEBODY_SKELETON_INFO),
}


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelVisState:
    pipeline:   MMPipeline
    legend:     str
    color:      tuple[float, float, float]
    limb_color: tuple[float, float, float]
    side_colors: dict[str, tuple[float, float, float]]
    skeleton:   list[tuple[int, int]]


@dataclass
class RawFrame:
    """One image with all model keypoints — before rendering."""
    base_bgr:   np.ndarray                          # original image
    kpt_sets:   list[np.ndarray]                    # one kpt array per model
    states:     list[ModelVisState]                 # same order as kpt_sets
    legends:    list[str]
    joint_colors: list[tuple[float, float, float]]
    limb_colors:  list[tuple[float, float, float]]
    side_colors_list: list[dict[str, tuple[float, float, float]]]


# ──────────────────────────────────────────────────────────────────────────────
# Render settings (mutable, shared)
# ──────────────────────────────────────────────────────────────────────────────

class RenderSettings:
    """All user-adjustable draw parameters. Reads happen in the render thread."""

    DEFAULT_JOINT_R   = 5.0
    DEFAULT_LIMB_W    = 3.0
    DEFAULT_SCORE_THR = 0.3

    def __init__(self) -> None:
        self.joint_r:        float = self.DEFAULT_JOINT_R
        self.limb_w:         float = self.DEFAULT_LIMB_W
        self.score_thr:      float = self.DEFAULT_SCORE_THR
        self.use_side_color: bool  = True
        self.bpart_filter: BodyPartFilter = body_filter


render_settings = RenderSettings()


# ──────────────────────────────────────────────────────────────────────────────
# Model building & inference
# ──────────────────────────────────────────────────────────────────────────────

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
        color       = MODEL_JOINT_COLORS[idx % len(MODEL_JOINT_COLORS)]
        limb_color  = MODEL_LIMB_COLORS[idx % len(MODEL_LIMB_COLORS)]
        side_colors = MODEL_SIDE_COLORS[idx % len(MODEL_SIDE_COLORS)]
        states.append(ModelVisState(pipeline, model_cfg.legend, color, limb_color,
                                    side_colors, skeleton))
        logger.info(f"[{idx}] {model_cfg.legend}")

    return states


def _infer(
    states: list[ModelVisState],
    image_path: str,
    bboxes: list,
) -> list[list[PoseDataSample]]:
    results: list[Optional[list[PoseDataSample]]] = [None] * len(states)
    for idx, state in enumerate(states):
        results[idx] = state.pipeline(image_path, bboxes)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Pre-processing: store raw data, render on-demand
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_all_frames(cfg: EvalConfig) -> list[RawFrame]:
    """Run inference on every image and store keypoints. Rendering is deferred."""
    logger.info("Loading models…")
    states = _build_model_states(cfg)

    ann_file     = cfg.models[0].ann_file
    dataset_path = cfg.models[0].dataset_folder
    coco         = COCO(ann_file)
    cat_ids      = coco.getCatIds(catNms=["person"])
    img_ids      = coco.getImgIds(catIds=cat_ids)

    legends           = [s.legend      for s in states]
    joint_colors      = [s.color       for s in states]
    limb_colors       = [s.limb_color  for s in states]
    side_colors_list  = [s.side_colors for s in states]

    raw_frames: list[RawFrame] = []

    logger.info(f"Running inference on {len(img_ids)} images…")
    for image_id in tqdm(img_ids[:10], desc="Inference"):
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

            kpt_sets: list[np.ndarray] = []
            for samples in all_results:
                if samples:
                    kpt_sets.append(samples[0].pred_instances.keypoints[0])
                else:
                    kpt_sets.append(np.zeros((0, 3)))

            raw_frames.append(RawFrame(
                base_bgr=image.copy(),
                kpt_sets=kpt_sets,
                states=states,
                legends=legends,
                joint_colors=joint_colors,
                limb_colors=limb_colors,
                side_colors_list=side_colors_list,
            ))

    logger.info(f"Inference done. {len(raw_frames)} frames stored.")
    return raw_frames


def render_frame(rf: RawFrame, rs: RenderSettings) -> np.ndarray:
    """Render one RawFrame using current RenderSettings. Returns a new BGR array."""
    img = rf.base_bgr.copy()
    flt = rs.bpart_filter

    for state, kpts, sc in zip(rf.states, rf.kpt_sets, rf.side_colors_list):
        if kpts.shape[0] == 0:
            continue
        draw_skeleton_pretty(
            img, kpts, state.skeleton,
            joint_color=state.color,
            limb_color=state.limb_color,
            side_colors=sc if rs.use_side_color else None,
            kpt_thr=rs.score_thr,
            joint_r=rs.joint_r,
            limb_w=rs.limb_w,
            bpart_filter=flt,
        )

    return img


# ──────────────────────────────────────────────────────────────────────────────
# Settings panel (Toplevel popover)
# ──────────────────────────────────────────────────────────────────────────────

class SettingsPanel(tk.Toplevel):
    """
    Floating settings window anchored near the toolbar button.
    Closing it hides rather than destroys so it can be re-opened instantly.
    """

    _BG       = "#1e2030"
    _FG       = "#c8cce8"
    _ACCENT   = "#5070c8"
    _CARD_BG  = "#252840"
    _SEP      = "#3a3d58"
    _SLIDER_BG    = "#acbce6"
    _SLIDER_TR    = "#41427a"

    def __init__(self, parent: tk.Tk, rs: RenderSettings,
                 on_change: callable) -> None:
        super().__init__(parent)
        self._rs = rs
        self._on_change = on_change

        self.title("Settings")
        self.configure(bg=self._BG)
        self.resizable(False, False)
        self.overrideredirect(False)  # keep title bar for easy dragging
        self.withdraw()               # hidden by default

        # Keep on top of main window
        self.transient(parent)

        self._build()

    # ── construction ─────────────────────────────────────────────────────────

    def _section(self, parent: tk.Widget, title: str) -> tk.Frame:
        """Titled card section."""
        outer = tk.Frame(parent, bg=self._BG)
        outer.pack(fill=tk.X, padx=12, pady=(10, 0))
        tk.Label(outer, text=title, bg=self._BG, fg=self._ACCENT,
                 font=("SegoeUI", 9, "bold")).pack(anchor=tk.W, pady=(0, 4))
        card = tk.Frame(outer, bg=self._CARD_BG, padx=12, pady=10)
        card.pack(fill=tk.X)
        return card

    def _separator(self, parent: tk.Widget) -> None:
        tk.Frame(parent, bg=self._SEP, height=1).pack(fill=tk.X, padx=12,
                                                       pady=(12, 0))

    def _build(self) -> None:
        root_frame = tk.Frame(self, bg=self._BG)
        root_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # ── Body parts ───────────────────────────────────────────────────────
        parts_card = self._section(root_frame, "BODY PARTS")

        self._part_vars: dict[str, tk.BooleanVar] = {}
        part_cfg = [
            ("body",  "🦴  Body  (COCO-17 skeleton)"),
            ("face",  "😶  Face  (face mesh, 68 pts)"),
            ("hands", "🤚  Hands  (palm keypoints)"),
            ("feet",  "🦶  Feet  (toe & heel points)"),
        ]
        for part_key, label in part_cfg:
            var = tk.BooleanVar(value=self._rs.bpart_filter.is_enabled(part_key))
            self._part_vars[part_key] = var
            cb = tk.Checkbutton(
                parts_card,
                text=label,
                variable=var,
                command=lambda k=part_key, v=var: self._on_part_toggle(k, v),
                bg=self._CARD_BG,
                fg=self._FG,
                selectcolor="#3a3e60",
                activebackground=self._CARD_BG,
                activeforeground=self._FG,
                font=("SegoeUI", 10),
                anchor=tk.W,
                cursor="hand2",
            )
            cb.pack(fill=tk.X, pady=1)

        self._separator(root_frame)

        # ── Side colouring ───────────────────────────────────────────────────
        colour_card = self._section(root_frame, "SIDE COLOURING")
        self._side_var = tk.BooleanVar(value=self._rs.use_side_color)
        tk.Checkbutton(
            colour_card,
            text="Colour by side  (Left / Right / Center)",
            variable=self._side_var,
            command=self._on_side_toggle,
            bg=self._CARD_BG,
            fg=self._FG,
            selectcolor="#3a3e60",
            activebackground=self._CARD_BG,
            activeforeground=self._FG,
            font=("SegoeUI", 10),
            anchor=tk.W,
            cursor="hand2",
        ).pack(fill=tk.X)

        self._separator(root_frame)

        # ── Size sliders ─────────────────────────────────────────────────────
        size_card = self._section(root_frame, "SIZES")

        # Joint radius: 1 – 15 px, default 5
        self._joint_r_var = tk.DoubleVar(value=self._rs.joint_r)
        self._make_slider(
            size_card,
            label="Joint radius",
            var=self._joint_r_var,
            from_=1.0, to=15.0,
            resolution=0.5,
            unit="px",
            command=self._on_joint_r,
        )

        # Limb width: 1 – 12 px, default 3
        self._limb_w_var = tk.DoubleVar(value=self._rs.limb_w)
        self._make_slider(
            size_card,
            label="Connection thickness",
            var=self._limb_w_var,
            from_=1.0, to=12.0,
            resolution=0.5,
            unit="px",
            command=self._on_limb_w,
        )

        self._separator(root_frame)

        # ── Score threshold ───────────────────────────────────────────────────
        thr_card = self._section(root_frame, "VISIBILITY")

        self._score_thr_var = tk.DoubleVar(value=self._rs.score_thr)
        self._make_slider(
            thr_card,
            label="Score threshold",
            var=self._score_thr_var,
            from_=0.0, to=1.0,
            resolution=0.01,
            unit="",
            command=self._on_score_thr,
        )

        # bottom padding
        tk.Frame(root_frame, bg=self._BG, height=10).pack()

    def _make_slider(
        self,
        parent: tk.Widget,
        label: str,
        var: tk.DoubleVar,
        from_: float,
        to: float,
        resolution: float,
        unit: str,
        command: callable,
    ) -> None:
        row = tk.Frame(parent, bg=self._CARD_BG)
        row.pack(fill=tk.X, pady=(6, 0))

        tk.Label(row, text=label, bg=self._CARD_BG, fg=self._FG,
                 font=("SegoeUI", 10), width=22, anchor=tk.W).pack(side=tk.LEFT)

        val_lbl = tk.Label(row, text=f"{var.get():.1f} {unit}",
                           bg=self._CARD_BG, fg="#8899cc",
                           font=("SegoeUI", 9), width=7)
        val_lbl.pack(side=tk.RIGHT)

        def _cb(v: str) -> None:
            val_lbl.config(text=f"{float(v):.1f} {unit}")
            command(float(v))

        s = tk.Scale(
            parent,
            variable=var,
            from_=from_, to=to,
            resolution=resolution,
            orient=tk.HORIZONTAL,
            showvalue=False,
            sliderlength=14,
            width=7,
            bg=self._SLIDER_BG,
            troughcolor=self._SLIDER_TR,
            fg=self._SLIDER_BG,
            activebackground=self._ACCENT,
            highlightthickness=0,
            bd=0,
            cursor="hand2",
            command=_cb,
        )
        s.pack(fill=tk.X, pady=(2, 4))

    # ── callbacks ────────────────────────────────────────────────────────────

    def _on_part_toggle(self, part: str, var: tk.BooleanVar) -> None:
        self._rs.bpart_filter.set(part, var.get())
        self._on_change()

    def _on_side_toggle(self) -> None:
        self._rs.use_side_color = self._side_var.get()
        self._on_change()

    def _on_joint_r(self, value: float) -> None:
        self._rs.joint_r = value
        self._on_change()

    def _on_limb_w(self, value: float) -> None:
        self._rs.limb_w = value
        self._on_change()

    def _on_score_thr(self, value: float) -> None:
        self._rs.score_thr = value
        self._on_change()

    # ── show/hide ────────────────────────────────────────────────────────────

    def toggle(self, x: int, y: int) -> None:
        if self.winfo_viewable():
            self.withdraw()
        else:
            self.geometry(f"+{x}+{y}")
            self.deiconify()
            self.lift()
            self.focus_force()


# ──────────────────────────────────────────────────────────────────────────────
# Legend panel (Toplevel popover, mirrors SettingsPanel pattern)
# ──────────────────────────────────────────────────────────────────────────────

class LegendPanel(tk.Toplevel):
    """
    Dropdown popover showing model names with coloured joint swatches.
    Each row: coloured circle (joint colour) + model name label.
    When side-colouring is active, shows left/right/center dot trio instead.
    """

    _BG      = "#1e2030"
    _FG      = "#c8cce8"
    _ACCENT  = "#5070c8"
    _CARD_BG = "#252840"
    _SEP     = "#3a3d58"

    def __init__(self, parent: tk.Tk, rs: RenderSettings) -> None:
        super().__init__(parent)
        self._rs = rs
        self._dot_photos: list = []  # keep PhotoImage refs alive

        self.title("Legend")
        self.configure(bg=self._BG)
        self.resizable(False, False)
        self.overrideredirect(False)
        self.withdraw()
        self.transient(parent)

        # Content is built lazily on first show (model list not known yet)
        self._built = False
        self._frame: Optional[tk.Frame] = None

    # ── public API ────────────────────────────────────────────────────────────

    def populate(
        self,
        legends: list[str],
        joint_colors: list[tuple[float, float, float]],
        side_colors_list: list[dict[str, tuple[float, float, float]]],
    ) -> None:
        """Call once after model states are known."""
        self._legends          = legends
        self._joint_colors     = joint_colors
        self._side_colors_list = side_colors_list

    def toggle(self, x: int, y: int) -> None:
        if self.winfo_viewable():
            self.withdraw()
        else:
            self._rebuild()
            self.geometry(f"+{x}+{y}")
            self.deiconify()
            self.lift()
            self.focus_force()

    # ── internal ─────────────────────────────────────────────────────────────

    def _make_dot(self, color_rgb_f: tuple[float, float, float], size: int = 14) -> tk.PhotoImage:
        """Create a tiny circular PhotoImage for a joint colour swatch."""
        from PIL import Image as PILImage, ImageDraw as PILDraw
        img = PILImage.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = PILDraw.Draw(img)
        r, g, b = (int(c * 255) for c in color_rgb_f)
        # white ring
        draw.ellipse([0, 0, size - 1, size - 1], fill=(255, 255, 255, 220))
        # coloured fill
        draw.ellipse([2, 2, size - 3, size - 3], fill=(r, g, b, 255))
        # specular
        hs = size // 4
        draw.ellipse([3, 3, 3 + hs, 3 + hs], fill=(255, 255, 255, 160))
        return ImageTk.PhotoImage(img)

    def _rebuild(self) -> None:
        """Destroy and recreate content (rebuilds when side-colour mode changes)."""
        if self._frame is not None:
            self._frame.destroy()
        self._dot_photos.clear()

        outer = tk.Frame(self, bg=self._BG)
        outer.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self._frame = outer

        use_side = self._rs.use_side_color

        # Section header
        hdr_frame = tk.Frame(outer, bg=self._BG)
        hdr_frame.pack(fill=tk.X, padx=12, pady=(10, 4))
        tk.Label(hdr_frame, text="MODELS", bg=self._BG, fg=self._ACCENT,
                 font=("SegoeUI", 9, "bold")).pack(anchor=tk.W)

        card = tk.Frame(outer, bg=self._CARD_BG, padx=14, pady=10)
        card.pack(fill=tk.X, padx=12)

        # Left accent stripe inside card (visual flair)
        stripe = tk.Frame(card, bg="#5070c8", width=3)
        stripe.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        rows = tk.Frame(card, bg=self._CARD_BG)
        rows.pack(side=tk.LEFT, fill=tk.X, expand=True)

        for i, legend in enumerate(self._legends):
            row = tk.Frame(rows, bg=self._CARD_BG)
            row.pack(fill=tk.X, pady=4)

            if use_side and self._side_colors_list:
                # Show 3 small dots: left / center / right
                sc = self._side_colors_list[i]
                for side_key in ("left", "center", "right"):
                    col = sc.get(side_key, self._joint_colors[i])
                    ph = self._make_dot(col, size=12)
                    self._dot_photos.append(ph)
                    lbl = tk.Label(row, image=ph, bg=self._CARD_BG)
                    lbl.pack(side=tk.LEFT, padx=1)
            else:
                # Single dot in model joint colour
                col = self._joint_colors[i]
                ph = self._make_dot(col, size=14)
                self._dot_photos.append(ph)
                lbl = tk.Label(row, image=ph, bg=self._CARD_BG)
                lbl.pack(side=tk.LEFT, padx=(0, 6))

            tk.Label(row, text=legend, bg=self._CARD_BG, fg=self._FG,
                     font=("SegoeUI", 11, "bold")).pack(side=tk.LEFT)

        # Side colour legend key (only when active)
        if use_side:
            sep = tk.Frame(outer, bg=self._SEP, height=1)
            sep.pack(fill=tk.X, padx=12, pady=(10, 0))

            key_frame = tk.Frame(outer, bg=self._BG)
            key_frame.pack(fill=tk.X, padx=24, pady=(6, 8))

            # Use colours from first model as example
            example_sc = self._side_colors_list[0] if self._side_colors_list else {}
            key_items = [
                ("left",   "Left"),
                ("center", "Center"),
                ("right",  "Right"),
            ]
            for side_key, side_label in key_items:
                col = example_sc.get(side_key, (0.6, 0.6, 0.6))
                ph = self._make_dot(col, size=10)
                self._dot_photos.append(ph)
                krow = tk.Frame(key_frame, bg=self._BG)
                krow.pack(side=tk.LEFT, padx=8)
                tk.Label(krow, image=ph, bg=self._BG).pack(side=tk.LEFT, padx=(0, 3))
                tk.Label(krow, text=side_label, bg=self._BG, fg="#7788aa",
                         font=("SegoeUI", 9)).pack(side=tk.LEFT)

        tk.Frame(outer, bg=self._BG, height=6).pack()


# ──────────────────────────────────────────────────────────────────────────────
# Player
# ──────────────────────────────────────────────────────────────────────────────

class PosePlayer:
    """
    Tkinter-based frame player with toolbar, settings panel, and live re-render.

    Keyboard shortcuts
    ------------------
    ←           previous frame
    →           next frame
    Shift+←     jump back 10 frames
    Shift+→     jump forward 10 frames
    Home        first frame
    End         last frame
    Space       toggle auto-play
    Esc         quit
    """

    AUTOPLAY_DELAY_MS = 16
    JUMP_STEP         = 10

    _BG          = "#1a1a2e"
    _TOOLBAR_BG  = "#16213e"
    _CTRL_BG     = "#0f3460"
    _SLIDER_BG   = "#acbce6"
    _SLIDER_TR   = "#41427a"
    _SLIDER_FG   = "#6578a8"
    _BTN_BG      = "#282852"
    _BTN_FG      = "#e0e0e0"
    _PAD         = 16

    def __init__(self, raw_frames: list[RawFrame], rs: RenderSettings,
                 title: str = "Pose Comparison Player") -> None:
        if not raw_frames:
            raise ValueError("No frames to display.")

        self._raw      = raw_frames
        self._rs       = rs
        self._index    = 0
        self._playing  = False
        self._after_id: Optional[str] = None

        # Rendered cache — invalidated when settings or canvas size change
        self._rendered:   list[Optional[np.ndarray]]       = [None] * len(raw_frames)
        self._photo_cache: list[Optional[ImageTk.PhotoImage]] = [None] * len(raw_frames)
        self._canvas_size: tuple[int, int] = (0, 0)
        self._settings_sig: tuple = ()   # snapshot of settings used to build cache

        self._resize_after_id: Optional[str] = None

        # ── root ─────────────────────────────────────────────────────────────
        self._root = tk.Tk()
        self._root.title(title)
        self._root.configure(bg=self._BG)
        self._root.resizable(True, True)

        sw, sh = self._root.winfo_screenwidth(), self._root.winfo_screenheight()
        self._root.maxsize(sw, sh)
        self._root.minsize(640, 480)
        init_w = max(1280, sw // 3)
        init_h = max(720, sh // 3)
        self._root.geometry(f"{init_w}x{init_h}+"
                            f"{(sw - init_w) // 2}+{(sh - init_h) // 2}")

        # ── toolbar ───────────────────────────────────────────────────────────
        toolbar = tk.Frame(self._root, bg=self._TOOLBAR_BG, pady=4)
        toolbar.pack(fill=tk.X, side=tk.TOP)

        tk.Label(toolbar, text="🎬 Pose Comparison",
                 bg=self._TOOLBAR_BG, fg="#7090e0",
                 font=("SegoeUI", 11, "bold")).pack(side=tk.LEFT, padx=12)

        # Settings button (right side of toolbar)
        self._btn_settings = tk.Button(
            toolbar,
            text="⚙  Settings",
            command=self._toggle_settings,
            bg="#2a2d4a", fg="#c0c8f0",
            activebackground="#3a3d68", activeforeground="#ffffff",
            relief=tk.FLAT, padx=10, pady=3,
            cursor="hand2",
            font=("SegoeUI", 10),
        )
        self._btn_settings.pack(side=tk.RIGHT, padx=(4, 10))

        # Legend button (right side, left of Settings)
        self._btn_legend = tk.Button(
            toolbar,
            text="🎨  Legend",
            command=self._toggle_legend,
            bg="#2a2d4a", fg="#c0c8f0",
            activebackground="#3a3d68", activeforeground="#ffffff",
            relief=tk.FLAT, padx=10, pady=3,
            cursor="hand2",
            font=("SegoeUI", 10),
        )
        self._btn_legend.pack(side=tk.RIGHT, padx=4)

        # ── canvas ────────────────────────────────────────────────────────────
        self._canvas = tk.Canvas(self._root, bg="#e8e8e1", highlightthickness=1)
        self._canvas.pack(fill=tk.BOTH, expand=True)
        self._img_id = self._canvas.create_image(0, 0, anchor=tk.CENTER)
        self._current_photo: Optional[ImageTk.PhotoImage] = None

        # ── progress slider ───────────────────────────────────────────────────
        self._slider_var = tk.IntVar(value=0)
        self._slider = tk.Scale(
            self._root,
            variable=self._slider_var,
            from_=0, to=max(0, len(raw_frames) - 1),
            orient=tk.HORIZONTAL,
            showvalue=False,
            sliderlength=16, width=8,
            bg=self._SLIDER_BG, troughcolor=self._SLIDER_TR,
            fg=self._SLIDER_FG, activebackground=self._SLIDER_FG,
            highlightthickness=0, bd=0, cursor="hand2",
            command=self._on_slider_move,
        )
        self._slider.pack(fill=tk.X, side=tk.BOTTOM, padx=4, pady=2)
        self._slider_updating = False

        # ── control bar ───────────────────────────────────────────────────────
        ctrl = tk.Frame(self._root, bg=self._CTRL_BG, pady=5)
        ctrl.pack(fill=tk.X, side=tk.BOTTOM)

        btn_cfg = dict(
            bg=self._BTN_BG, fg=self._BTN_FG,
            activebackground="#505050", activeforeground="#ffffff",
            pady=3, padx=3,
        )
        tk.Button(ctrl, text="⏮️", command=self._go_first,   **btn_cfg).pack(side=tk.LEFT, padx=(10, 2))
        tk.Button(ctrl, text="⏪", command=self._prev,        **btn_cfg).pack(side=tk.LEFT, padx=2)
        self._btn_play = tk.Button(ctrl, text="▶️", command=self._toggle_play, **btn_cfg)
        self._btn_play.pack(side=tk.LEFT, padx=2)
        tk.Button(ctrl, text="⏩", command=self._next,        **btn_cfg).pack(side=tk.LEFT, padx=2)
        tk.Button(ctrl, text="⏭️", command=self._go_last,    **btn_cfg).pack(side=tk.LEFT, padx=2)

        self._label = tk.Label(ctrl, text=self._frame_label(),
                               fg=self._BTN_FG, bg=self._CTRL_BG,
                               font=("SegoeUI", 10), width=14)
        self._label.pack(side=tk.RIGHT, padx=12)

        # ── settings panel ────────────────────────────────────────────────────
        self._settings_panel = SettingsPanel(
            self._root, self._rs, on_change=self._on_settings_change
        )

        # ── legend panel ──────────────────────────────────────────────────────
        self._legend_panel = LegendPanel(self._root, self._rs)
        if raw_frames:
            rf0 = raw_frames[0]
            self._legend_panel.populate(
                legends=rf0.legends,
                joint_colors=rf0.joint_colors,
                side_colors_list=rf0.side_colors_list,
            )

        # ── bindings ──────────────────────────────────────────────────────────
        self._root.bind("<Left>",       lambda _: self._prev())
        self._root.bind("<Right>",      lambda _: self._next())
        self._root.bind("<Shift-Left>", lambda _: self._jump_back())
        self._root.bind("<Shift-Right>",lambda _: self._jump_fwd())
        self._root.bind("<Home>",       lambda _: self._go_first())
        self._root.bind("<End>",        lambda _: self._go_last())
        self._root.bind("<space>",      lambda _: self._toggle_play())
        self._root.bind("<Escape>",     lambda _: self._quit())
        self._canvas.bind("<Configure>", self._on_canvas_resize)
        self._root.protocol("WM_DELETE_WINDOW", self._quit)

        self._root.after(50, self._refresh)

    # ── settings ─────────────────────────────────────────────────────────────

    def _toggle_settings(self) -> None:
        btn = self._btn_settings
        x = btn.winfo_rootx()
        y = btn.winfo_rooty() + btn.winfo_height() + 4
        self._settings_panel.toggle(x, y)

    def _toggle_legend(self) -> None:
        btn = self._btn_legend
        x = btn.winfo_rootx()
        y = btn.winfo_rooty() + btn.winfo_height() + 4
        self._legend_panel.toggle(x, y)

    def _settings_signature(self) -> tuple:
        rs = self._rs
        flt = rs.bpart_filter
        return (
            rs.joint_r,
            rs.limb_w,
            rs.score_thr,
            rs.use_side_color,
            tuple(flt.is_enabled(p) for p in BodyPartFilter.PARTS),
        )

    def _on_settings_change(self) -> None:
        """Called by SettingsPanel whenever anything changes."""
        self._rendered    = [None] * len(self._raw)
        self._photo_cache = [None] * len(self._raw)
        self._settings_sig = self._settings_signature()
        # Rebuild legend content if side-colour mode changed
        if self._legend_panel.winfo_viewable():
            self._legend_panel._rebuild()
        self._refresh()

    # ── render pipeline ───────────────────────────────────────────────────────

    def _get_rendered(self, idx: int) -> np.ndarray:
        sig = self._settings_signature()
        if sig != self._settings_sig:
            self._rendered    = [None] * len(self._raw)
            self._photo_cache = [None] * len(self._raw)
            self._settings_sig = sig
        if self._rendered[idx] is None:
            self._rendered[idx] = render_frame(self._raw[idx], self._rs)
        return self._rendered[idx]

    def _render_to_photo(self, idx: int, cw: int, ch: int) -> ImageTk.PhotoImage:
        bgr = self._get_rendered(idx)
        src_h, src_w = bgr.shape[:2]
        avail_w = max(1, cw - 2 * self._PAD)
        avail_h = max(1, ch - 2 * self._PAD)
        scale = min(avail_w / src_w, avail_h / src_h)
        nw = max(1, int(src_w * scale))
        nh = max(1, int(src_h * scale))
        resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return ImageTk.PhotoImage(Image.fromarray(rgb))

    def _get_photo(self, idx: int, cw: int, ch: int) -> ImageTk.PhotoImage:
        if self._canvas_size != (cw, ch):
            self._photo_cache  = [None] * len(self._raw)
            self._canvas_size  = (cw, ch)
        if self._photo_cache[idx] is None:
            self._photo_cache[idx] = self._render_to_photo(idx, cw, ch)
        return self._photo_cache[idx]

    def _warm_next(self, idx: int, cw: int, ch: int) -> None:
        nxt = idx + 1
        if nxt < len(self._raw) and self._photo_cache[nxt] is None:
            self._photo_cache[nxt] = self._render_to_photo(nxt, cw, ch)

    # ── display ───────────────────────────────────────────────────────────────

    def _frame_label(self) -> str:
        return f"{self._index + 1:>5} / {len(self._raw)}"

    def _refresh(self) -> None:
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
        self._root.after_idle(self._warm_next, self._index, cw, ch)

    # ── navigation ────────────────────────────────────────────────────────────

    def _on_slider_move(self, value: str) -> None:
        if self._slider_updating:
            return
        idx = int(value)
        if idx != self._index:
            self._index = idx
            self._refresh()

    def _on_canvas_resize(self, event: tk.Event) -> None:
        if self._resize_after_id is not None:
            self._root.after_cancel(self._resize_after_id)
        self._resize_after_id = self._root.after(80, self._refresh)

    def _go_first(self) -> None:
        self._index = 0;                                 self._refresh()

    def _go_last(self) -> None:
        self._index = len(self._raw) - 1;               self._refresh()

    def _prev(self) -> None:
        if self._index > 0:
            self._index -= 1;                            self._refresh()

    def _next(self) -> None:
        if self._index < len(self._raw) - 1:
            self._index += 1;                            self._refresh()

    def _jump_back(self) -> None:
        self._index = max(0, self._index - self.JUMP_STEP); self._refresh()

    def _jump_fwd(self) -> None:
        self._index = min(len(self._raw) - 1, self._index + self.JUMP_STEP)
        self._refresh()

    def _toggle_play(self) -> None:
        self._playing = not self._playing
        self._btn_play.config(text="⏸️" if self._playing else "▶️")
        if self._playing:
            self._autoplay()

    def _autoplay(self) -> None:
        if not self._playing:
            return
        if self._index < len(self._raw) - 1:
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


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visual pose model comparison (tkinter player)"
    )
    parser.add_argument("--config", "-c", type=str, required=True,
                        help="Path to eval-config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args       = _parse_args()
    cfg        = EvalConfig.load(args.config)
    raw_frames = preprocess_all_frames(cfg)
    rs         = RenderSettings()
    player     = PosePlayer(raw_frames, rs, title="Pose Comparison Player")
    player.run()