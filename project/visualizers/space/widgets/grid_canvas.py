import math
import threading
from pathlib import Path
from typing import List, Tuple, Optional

import tkinter as tk
import cv2
import numpy as np

from project.visualizers.space.constants import W
from project.visualizers.space.drawing import resize, draw_frame, fit_image, outlined, rgb_to_photoimage
from project.visualizers.space.models import DrawParams
from project.visualizers.space.widgets.group_panel import GroupPanel
from project.visualizers.space.widgets.view_group import ViewGroup


class GridCanvas(tk.Frame):
    def __init__(self, parent: tk.Widget, ctrl: object) -> None:
        super().__init__(parent, bg=W["bg"])
        self.ctrl: object = ctrl
        self._panels: List[GroupPanel] = []
        self._expanded: Optional[GroupPanel] = None
        self._layout_str: str = "auto"

    def set_groups(self, groups: List[ViewGroup], layout: str) -> None:
        # clear old panels
        for panel in self._panels:
            try:
                panel.canvas.unbind("<Configure>")
            except Exception:
                pass
            panel.destroy()
        self._panels.clear()
        self._expanded = None
        self._layout_str = layout

        for grp in groups:
            panel = GroupPanel(self, grp, self.ctrl)
            self._panels.append(panel)

        self._do_layout()

    def _do_layout(self) -> None:
        for panel in self._panels:
            panel.grid_forget()
            panel.pack_forget()

        if self._expanded is not None:
            self._expanded.pack(fill=tk.BOTH, expand=True)
            return

        n = len(self._panels)
        if n == 0:
            return

        # reset weights
        for i in range(20):
            self.columnconfigure(i, weight=0, uniform="", minsize=0)
            self.rowconfigure(i, weight=0, uniform="", minsize=0)

        rows, cols = self._calc_grid(n)

        for c in range(cols):
            self.columnconfigure(c, weight=1, uniform="gcol")
        for r in range(rows):
            self.rowconfigure(r, weight=1, uniform="grow")

        for i, panel in enumerate(self._panels):
            r, c = divmod(i, cols)
            panel.grid(row=r, column=c, sticky="nsew", padx=2, pady=2)

    def _calc_grid(self, n: int) -> Tuple[int, int]:
        layout = getattr(self, "_layout_str", "auto")
        if n == 0:
            return 1, 1
        if layout == "1xN":
            return 1, n
        if layout == "Nx1":
            return n, 1
        if layout == "2x2":
            cols = 2
            rows = math.ceil(n / cols)
            return rows, cols
        if layout == "2x3":
            cols = 3
            rows = math.ceil(n / cols)
            return rows, cols
        # auto — square grid
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        return rows, cols

    def expand(self, panel: GroupPanel) -> None:
        self._expanded = panel
        self._do_layout()
        self.update_idletasks()

    def collapse(self) -> None:
        self._expanded = None
        self._do_layout()
        self.update_idletasks()

    def draw_all(self, dp: DrawParams) -> None:
        if self._expanded:
            self._expanded.draw(dp)
            return

        panels = self._panels
        n = len(panels)
        if n == 0:
            return

        sizes: List[Tuple[int, int]] = [(max(p.canvas.winfo_width(), 100),
                                         max(p.canvas.winfo_height(), 100)) for p in panels]

        np_results: List[Optional[np.ndarray]] = [None] * n

        def render_numpy(i: int, group: ViewGroup, cw: int, ch: int) -> None:
            try:
                entry = group.current_entry()
                if entry is None:
                    return
                img = group.loader.load(entry.file_name)
                if img is None:
                    return
                img = img.copy()
                oh, ow = img.shape[:2]
                tw = group.tracker.get_window(group._idx, dp.track_length) if dp.show_tracks else None
                draw_frame(img, entry.annotations, dp, tw)
                if dp.show_frame_label:
                    try:
                        fn = int(Path(entry.file_name).stem)
                        outlined(img, f"ID={fn+1}", (10, 34),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.75, (120, 210, 255), 2)
                    except ValueError:
                        pass
                    outlined(img, group.cfg.name, (10, oh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                             (255, 220, 80), 1)
                nw, nh = fit_image(ow, oh, cw, ch)
                resized = resize(img, nw, nh)
                np_results[i] = resized
            except Exception:
                pass

        threads: List[threading.Thread] = [
            threading.Thread(target=render_numpy, args=(i, panels[i].group, sizes[i][0], sizes[i][1]), daemon=True)
            for i in range(n)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # update canvas
        for i, panel in enumerate(panels):
            cw, ch = sizes[i]
            arr = np_results[i]
            panel.canvas.delete("all")
            if arr is None:
                msg = "Folder with images doesn't chose" if not panel.group.loader.directory else "File not found"
                panel.canvas.create_text(cw // 2, ch // 2, text=msg, fill="#555",
                                         font=("Comic Sans MS", 11), justify="center")
            else:
                nh, nw = arr.shape[:2]
                photo = rgb_to_photoimage(arr)
                panel.group._photo = photo
                panel.canvas.create_image((cw - nw) // 2, (ch - nh) // 2, anchor="nw", image=photo)

            t_total = panel.group.total()
            panel._overlay.config(text=f"{panel.group._idx + 1}/{t_total}" if t_total else "0/0")

    def get_panels(self) -> List[GroupPanel]:
        return self._panels