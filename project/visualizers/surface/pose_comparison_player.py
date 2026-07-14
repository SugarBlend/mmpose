import argparse
import sys
import tkinter as tk
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk
from dotenv import load_dotenv

sys.path.insert(0, Path(__file__).parents[2].as_posix())
from config import EvalConfig
from overlay import (
    BodyPartFilter,
    draw_skeleton_pretty,
)
from preprocessing import RenderSettings, RawFrame, preprocess_all_frames
from widgets.legend_panel import LegendPanel
from widgets.settings_panel import SettingsPanel


def render_frame(rf: RawFrame, rs: RenderSettings, counter: int, writers: list[cv2.VideoWriter]) -> np.ndarray:
    img = rf.base_bgr.copy()
    flt = rs.bpart_filter

    for i, state in enumerate(rf.states):
        items = len(rf.kpt_sets) // len(rf.states)
        kpts = rf.kpt_sets[i * items: (i + 1) * items]
        sc = rf.side_colors_list[i]

        if not len(kpts) or kpts[0].shape[0] == 0:
            continue
        for skeleton in kpts:
            img = draw_skeleton_pretty(
                img.copy(), skeleton, state.skeleton,
                joint_color=state.color,
                limb_color=state.limb_color,
                side_colors=sc if rs.use_side_color else None,
                kpt_thr=rs.score_thr,
                joint_r=rs.joint_r,
                limb_w=rs.limb_w,
                bpart_filter=flt,
            )
        if i == len(rf.states) - 1:
            writers[counter % 4].write(img)
    return img


class PosePlayer(object):
    AUTOPLAY_DELAY_MS = 1
    JUMP_STEP = 10

    _BG = "#C9C9C9"
    _TOOLBAR_BG = "#C9C9C9"
    _CTRL_BG = "#C9C9C9"
    _SLIDER_BG = "#C9C9C9"
    _SLIDER_TR = "#545252"
    _SLIDER_FG = "#F5F5F5"
    _BTN_BG = "#ADA8A8"
    _BTN_FG = "#080808"
    _PAD = 16

    def __init__(self, raw_frames: list[RawFrame], rs: RenderSettings,
                 title: str = "Pose Comparison Player") -> None:
        if not raw_frames:
            raise ValueError("No frames to display.")

        self._raw = raw_frames
        self._rs = rs
        self._index = 0
        self._playing = False
        self._after_id: Optional[str] = None

        self._rendered: list[Optional[np.ndarray]] = [None] * len(raw_frames)
        self._photo_cache: list[Optional[ImageTk.PhotoImage]] = [None] * len(raw_frames)
        self._canvas_size: tuple[int, int] = (0, 0)
        self._settings_sig: tuple = ()
        self._resize_after_id: Optional[str] = None

        self._root = tk.Tk()
        self._root.title(title)
        self._root.configure(bg=self._BG)
        self._root.resizable(True, True)

        sw, sh = self._root.winfo_screenwidth(), self._root.winfo_screenheight()
        self._root.maxsize(sw, sh)
        self._root.minsize(640, 480)
        init_w = max(1280, sw // 3)
        init_h = max(720, sh // 3)
        self._root.geometry(f"{init_w}x{init_h}+{(sw - init_w) // 2}+{(sh - init_h) // 2}")

        toolbar = tk.Frame(self._root, bg=self._TOOLBAR_BG, pady=4)
        toolbar.pack(fill=tk.X, side=tk.TOP)

        btn_cfg = dict(bg=self._BTN_BG, fg=self._BTN_FG, activebackground="#505050", activeforeground="#ffffff")

        self._btn_settings = tk.Button(
            toolbar, text="Settings", command=self._toggle_settings,
            padx=10, pady=3, cursor="hand2", font=("SegoeUI", 10), **btn_cfg,
        )
        self._btn_settings.pack(side=tk.RIGHT, padx=(4, 10))

        self._btn_legend = tk.Button(
            toolbar, text="Legend", command=self._toggle_legend,
            padx=10, pady=3, cursor="hand2", font=("SegoeUI", 10), **btn_cfg,
        )
        self._btn_legend.pack(side=tk.RIGHT, padx=4)

        self._canvas = tk.Canvas(self._root, bg="#e8e8e1", highlightthickness=1, highlightbackground="#080808")
        self._canvas.pack(fill=tk.BOTH, expand=True)
        self._img_id = self._canvas.create_image(0, 0, anchor=tk.CENTER)
        self._current_photo: Optional[ImageTk.PhotoImage] = None

        self._slider_var = tk.IntVar(value=0)
        self._slider = tk.Scale(
            self._root, variable=self._slider_var,
            from_=0, to=max(0, len(raw_frames) - 1),
            orient=tk.HORIZONTAL,
            showvalue=False, sliderlength=16, width=8,
            bg=self._SLIDER_BG, troughcolor=self._SLIDER_TR,
            fg=self._SLIDER_FG, activebackground=self._SLIDER_FG,
            highlightthickness=1, bd=0, cursor="hand2",
            highlightbackground="#080808",
            command=self._on_slider_move,
        )
        self._slider.pack(fill=tk.X, side=tk.BOTTOM, padx=14, pady=2)
        self._slider_updating = False

        ctrl = tk.Frame(self._root, bg=self._CTRL_BG, pady=5)
        ctrl.pack(fill=tk.X, side=tk.BOTTOM)

        tk.Button(ctrl, text="⏮️", command=self._go_first,  padx=15, **btn_cfg).pack(side=tk.LEFT, padx=(10, 2))
        tk.Button(ctrl, text="⏪", command=self._prev, padx=15, **btn_cfg).pack(side=tk.LEFT, padx=2)
        self._btn_play = tk.Button(ctrl, text="▶️", command=self._toggle_play, padx=15, **btn_cfg)
        self._btn_play.pack(side=tk.LEFT, padx=2)
        tk.Button(ctrl, text="⏩", command=self._next, padx=15, **btn_cfg).pack(side=tk.LEFT, padx=2)
        tk.Button(ctrl, text="⏭️", command=self._go_last, padx=15, **btn_cfg).pack(side=tk.LEFT, padx=2)

        self._label = tk.Label(
            ctrl, text=self._frame_label(),
            fg=self._BTN_FG, bg=self._CTRL_BG,
            font=("SegoeUI", 10), width=14,
        )
        self._label.pack(side=tk.RIGHT, padx=12)

        self._settings_panel = SettingsPanel(
            self._root, self._rs, on_change=self._on_settings_change,
        )
        self._legend_panel = LegendPanel(self._root, self._rs)
        if raw_frames:
            rf0 = raw_frames[0]
            self._legend_panel.populate(
                legends=rf0.legends,
                joint_colors=rf0.joint_colors,
                side_colors_list=rf0.side_colors_list,
            )

        # ── keyboard / window bindings ────────────────────────────────────
        self._root.bind("<Left>", lambda _: self._prev())
        self._root.bind("<Right>", lambda _: self._next())
        self._root.bind("<Shift-Left>", lambda _: self._jump_back())
        self._root.bind("<Shift-Right>", lambda _: self._jump_fwd())
        self._root.bind("<Home>", lambda _: self._go_first())
        self._root.bind("<End>", lambda _: self._go_last())
        self._root.bind("<space>", lambda _: self._toggle_play())
        self._root.bind("<Escape>", lambda _: self._quit())
        self._canvas.bind("<Configure>", self._on_canvas_resize)
        self._canvas.bind("<Button-1>", self._close_panels, add="+")
        self._slider.bind("<Button-1>", self._close_panels, add="+")
        self._root.protocol("WM_DELETE_WINDOW", self._quit)

        # reposition panels when main window moves / resizes
        self._root.bind("<Configure>", self._on_root_configure, add="+")

        self._root.after(50, self._refresh)
        self.writers: list[cv2.VideoWriter] | None = None

    def _init_writers(self, shape: tuple[int, int], cameras: list = range(4)) -> None:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_folder = Path("recordings")
        output_folder.mkdir(parents=True, exist_ok=True)
        self.writers = [cv2.VideoWriter(output_folder.joinpath(f'output_{cam}.avi').as_posix(), fourcc, 6, shape)
                        for cam in cameras]

    def _panel_pos(self, btn: tk.Button, panel: tk.Toplevel) -> tuple[int, int]:
        # Ensure panel geometry is up-to-date
        panel.update_idletasks()
        pw = panel.winfo_reqwidth()

        btn_left = btn.winfo_rootx()
        y = btn.winfo_rooty() + btn.winfo_height() + 4
        sw = self._root.winfo_screenwidth()

        x = btn_left
        if x + pw > sw:
            x = sw - pw - 4  # shift left so panel fits on screen
        return x, y

    def _toggle_settings(self) -> None:
        # Close the other panel first
        self._legend_panel.close()
        # If already visible — just close (button was clicked again)
        if self._settings_panel.winfo_viewable():
            self._settings_panel.close()
            return

        x, y = self._panel_pos(self._btn_settings, self._settings_panel)
        self._settings_panel.toggle(x, y)
        # Keep keyboard focus on main window so arrow keys keep working
        self._root.focus_force()

    def _toggle_legend(self) -> None:
        # Close the other panel first
        self._settings_panel.close()
        # If already visible — just close (button was clicked again)
        if self._legend_panel.winfo_viewable():
            self._legend_panel.close()
            return

        x, y = self._panel_pos(self._btn_legend, self._legend_panel)
        self._legend_panel.toggle(x, y)
        # Keep keyboard focus on main window so arrow keys keep working
        self._root.focus_force()

    def _close_panels(self, event: tk.Event = None) -> None:
        self._settings_panel.close()
        self._legend_panel.close()

    def _on_root_configure(self, event: tk.Event) -> None:
        if event.widget is not self._root:
            return

        if self._settings_panel.winfo_viewable():
            x, y = self._panel_pos(self._btn_settings, self._settings_panel)
            self._settings_panel.geometry(f"+{x}+{y}")

        if self._legend_panel.winfo_viewable():
            x, y = self._panel_pos(self._btn_legend, self._legend_panel)
            self._legend_panel.geometry(f"+{x}+{y}")

    def _settings_signature(self) -> tuple:
        rs = self._rs
        flt = rs.bpart_filter
        return (
            rs.joint_r, rs.limb_w, rs.score_thr, rs.use_side_color,
            tuple(flt.is_enabled(p) for p in BodyPartFilter.PARTS),
        )

    def _on_settings_change(self) -> None:
        self._rendered = [None] * len(self._raw)
        self._photo_cache = [None] * len(self._raw)
        self._settings_sig = self._settings_signature()
        if self._legend_panel.winfo_viewable():
            self._legend_panel._rebuild()

        self._refresh()

    def _get_rendered(self, idx: int) -> np.ndarray:
        sig = self._settings_signature()
        if sig != self._settings_sig:
            self._rendered = [None] * len(self._raw)
            self._photo_cache = [None] * len(self._raw)
            self._settings_sig = sig

        if self._rendered[idx] is None:

            if self.writers is None:
                h, w, c = self._raw[idx].base_bgr.shape
                #FIXME: hardcode 4 sequentially cameras
                self._init_writers((w, h))
            self._rendered[idx] = render_frame(self._raw[idx], self._rs, self._index + 1, self.writers)

        return self._rendered[idx]

    def _render_to_photo(self, idx: int, cw: int, ch: int) -> ImageTk.PhotoImage:
        bgr = self._get_rendered(idx)
        src_h, src_w = bgr.shape[:2]
        avail_w = max(1, cw - 2 * self._PAD)
        avail_h = max(1, ch - 2 * self._PAD)
        scale = min(avail_w / src_w, avail_h / src_h)
        nw, nh = max(1, int(src_w * scale)), max(1, int(src_h * scale))
        resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return ImageTk.PhotoImage(Image.fromarray(rgb))

    def _get_photo(self, idx: int, cw: int, ch: int) -> ImageTk.PhotoImage:
        if self._canvas_size != (cw, ch):
            self._photo_cache = [None] * len(self._raw)
            self._canvas_size = (cw, ch)

        if self._photo_cache[idx] is None:
            self._photo_cache[idx] = self._render_to_photo(idx, cw, ch)

        return self._photo_cache[idx]

    def _warm_next(self, idx: int, cw: int, ch: int) -> None:
        nxt = idx + 1
        if nxt < len(self._raw) and self._photo_cache[nxt] is None:
            self._photo_cache[nxt] = self._render_to_photo(nxt, cw, ch)

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
        self._index = 0
        self._refresh()

    def _go_last(self) -> None:
        self._index = len(self._raw) - 1
        self._refresh()

    def _prev(self) -> None:
        if self._index > 0:
            self._index -= 1
            self._refresh()

    def _next(self) -> None:
        if self._index < len(self._raw) - 1:
            self._index += 1
            self._refresh()

    def _jump_back(self) -> None:
        self._index = max(0, self._index - self.JUMP_STEP)
        self._refresh()

    def _jump_fwd(self)  -> None:
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
        [item.release() for item in self.writers]

        if self._after_id is not None:
            self._root.after_cancel(self._after_id)
        self._root.destroy()

    def run(self) -> None:
        self._root.mainloop()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visual pose model comparison (tkinter player)"
    )
    parser.add_argument("--config", "-c", type=str, default="./eval-config.yaml",
                        help="Path to eval-config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    load_dotenv("../../../tools/.env")
    args = _parse_args()
    cfg = EvalConfig.load(args.config)
    raw_frames = preprocess_all_frames(cfg)
    rs = RenderSettings()
    player = PosePlayer(raw_frames, rs, title="Pose Comparison Player")
    player.run()
