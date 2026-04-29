import time
import tkinter
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import List, Optional
import cv2

from constants import W, FPS_PRESETS
from models import DrawParams, FilterParams, ImageEntry, ViewGroupConfig
from services import DataService, ImageLoader, TrackBuilder
from widgets.view_group import ViewGroup
from style import apply_style, hsep
from drawing import draw_frame, fit_image, resize, rgb_to_photoimage, outlined
from widgets import TopBar, PlayerBar, InfoPanel, GridCanvas, GroupManagerDialog


class AppController:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Annotation Visualizer")
        root.geometry("1520x940")
        root.minsize(900, 600)
        self.font = "Comic Sans MS"

        self._init_vars()

        # Shared loader and tracker
        self.loader = ImageLoader()
        self.data_service = DataService()
        self.tracker = TrackBuilder()

        # Grid mode: list of ViewGroups; single mode: None
        self.view_groups: Optional[List[ViewGroup]] = None
        self.grid_layout: str = "auto"

        self._idx = 0
        self._playing = False
        self._fps_lock = False
        self._rsz_tm = None
        self._layout_in_progress = False
        self._photo = None
        self._slider_updating = False

        apply_style(root)
        root.configure(bg=W["bg"])
        self._build_ui()
        self._bind_keys()

    def _init_vars(self):
        self.regex_patterns: list = [".*"]
        self.regex_operators: list = []
        self._filter_rows: list = []

        self.show_bbox_var = tk.BooleanVar(value=True)
        self.show_bbox_fill_var = tk.BooleanVar(value=True)
        self.show_kpts_var = tk.BooleanVar(value=True)
        self.show_skel_var = tk.BooleanVar(value=True)
        self.show_jids_var = tk.BooleanVar(value=False)
        self.show_aids_var = tk.BooleanVar(value=False)
        self.show_label_var = tk.BooleanVar(value=True)
        self.show_tracks_var = tk.BooleanVar(value=False)

        self.radius_var = tk.DoubleVar(value=9.0)
        self.font_var = tk.DoubleVar(value=0.4)
        self.skel_thick_var = tk.DoubleVar(value=5.0)
        self.bbox_alpha_var = tk.DoubleVar(value=0.15)
        self.track_len_var = tk.DoubleVar(value=6.0)
        self.track_alpha_var = tk.DoubleVar(value=0.7)

        self.play_direction_var = tk.StringVar(value="forward")
        self.loop_var = tk.BooleanVar(value=True)
        self._play_delay = 33

    def _build_ui(self) -> None:
        self.topbar = TopBar(self.root, self)
        self.topbar.pack(fill=tk.X, side=tk.TOP)
        hsep(self.root, W["border"]).pack(fill=tk.X)

        work = tk.Frame(self.root, bg=W["bg"])
        work.pack(fill=tk.BOTH, expand=True)

        self.vpaned = ttk.PanedWindow(work, orient=tk.VERTICAL)
        self.vpaned.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        top_wrap = tk.Frame(self.vpaned, bg=W["bg"])
        self.vpaned.add(top_wrap, weight=4)

        # Single canvas mode
        self.canvas = tk.Canvas(top_wrap, bg=W["canvas"], highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Double-Button-3>", lambda e: self.toggle_fullscreen())
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # Grid mode canvas
        self.grid_canvas = GridCanvas(top_wrap, self)

        hsep(top_wrap, W["border"]).pack(fill=tk.X)
        self.player = PlayerBar(top_wrap, self)
        self.player.pack(fill=tk.X, side=tk.BOTTOM)

        self.info = InfoPanel(self.vpaned, self)
        self.vpaned.add(self.info, weight=1)

    def _bind_keys(self) -> None:
        self.root.bind("<Left>", lambda event: self.previous_image())
        self.root.bind("<Right>", lambda event: self.next_image())
        self.root.bind("<Up>", lambda event: self.next_image(step=10))
        self.root.bind("<Down>", lambda event: self.previous_image(step=10))
        self.root.bind("<space>", self.toggle_play)
        self.root.bind("<F11>", self.toggle_fullscreen)
        self.root.bind("<Escape>", self.exit_fullscreen)
        self.root.bind("<Configure>", self._on_resize)
        self.root.focus_set()

    def _is_grid_mode(self) -> bool:
        return self.view_groups is not None and len(self.view_groups) > 0

    def _show_grid_mode(self) -> None:
        self.canvas.pack_forget()
        self.grid_canvas.pack(fill=tk.BOTH, expand=True)

    def _show_single_mode(self) -> None:
        self.grid_canvas.pack_forget()
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def open_group_manager(self) -> None:
        GroupManagerDialog(self.root, self)

    def apply_view_groups(self, cfgs: List[ViewGroupConfig], layout: str) -> None:
        self._layout_in_progress = True
        if self._rsz_tm:
            self.root.after_cancel(self._rsz_tm)
            self._rsz_tm = None

        try:
            self.grid_layout = layout
            if not cfgs:
                self.clear_view_groups()
                return

            self.view_groups = []
            for cfg in cfgs:
                grp = ViewGroup(cfg, self.loader)
                if self.data_service.all_entries:
                    grp.load_from(self.data_service.all_entries, self.data_service.total_anns)
                self.view_groups.append(grp)

            self.grid_canvas.set_groups(self.view_groups, layout)
            self._show_grid_mode()
            self.root.update_idletasks()

            for p in self.grid_canvas.get_panels():
                p.canvas.bind("<Configure>", lambda e: self._on_panel_resize())

            max_total = max((g.total() for g in self.view_groups), default=0)
            self._set_slider(0, max(max_total, 1))

            names = ", ".join(c.name for c in cfgs)
            self.topbar.status_label.config(text=f"Groups: {len(cfgs)}  [{names}]", fg=W["accent"])
        finally:
            self._layout_in_progress = False

        self.refresh()

    def clear_view_groups(self) -> None:
        self._layout_in_progress = True
        if self._rsz_tm:
            self.root.after_cancel(self._rsz_tm)
            self._rsz_tm = None
        try:
            for p in self.grid_canvas.get_panels():
                try:
                    p.canvas.unbind("<Configure>")
                except (Exception, ):
                    pass
            self.view_groups = None
            self.grid_canvas.set_groups([], "auto")
            self._show_single_mode()
            self.topbar.status_label.config(text="Single view", fg=W["green"])
        finally:
            self._layout_in_progress = False
        self.refresh()

    def expand_group(self, group: ViewGroup) -> None:
        for panel in self.grid_canvas.get_panels():
            if panel.group is group:
                self.grid_canvas.expand(panel)
                self.refresh()
                return

    def collapse_group(self) -> None:
        self.grid_canvas.collapse()
        self.refresh()

    def load_annotations(self) -> None:
        path = filedialog.askopenfilename(
            title="Select annotation file", filetypes=[("JSON", "*.json"), ("All", "*.*")]
        )
        if not path:
            return

        try:
            ni, na = self.data_service.load(path)
            nm = Path(path).name
            self.topbar.file_info.config(text=f"🧾 {nm} {ni}/{na} frames/anns", fg=W["green"])
            self.topbar.status_label.config(text=f"{ni} frames", fg=W["green"])
            if self.view_groups:
                for grp in self.view_groups:
                    grp.load_from(self.data_service.all_entries, self.data_service.total_anns)
            self.apply_filter()
        except Exception as ex:
            messagebox.showerror("Loading error", str(ex))
            self.topbar.file_info.config(text=f"✗ {ex}", fg=W["red"])

    def select_images_dir(self) -> None:
        path = filedialog.askdirectory(title="Folder with images")
        if path:
            self.loader.set_directory(path)
            cur = self.topbar.file_info.cget("text")
            self.topbar.file_info.config(text=cur + f"\n📂 {Path(path).name}")
            self.refresh()

    def apply_filter(self) -> None:
        if self._filter_rows:
            self.regex_patterns = [r["var"].get() for r in self._filter_rows]
            self.regex_operators = [r["op_var"].get() for r in self._filter_rows[1:]]
        if not self.regex_patterns or all(not p.strip() for p in self.regex_patterns):
            self.regex_patterns = [".*"]
            self.regex_operators = []

        p = FilterParams(patterns=self.regex_patterns, operators=self.regex_operators)
        count = self.data_service.apply_filter(p)
        self._idx = 0
        if count:
            self.player.progress.config(from_=0, to=max(0, count - 1))
        self.tracker.build(self.data_service.filtered_entries)
        self.topbar.filter_lbl.config(text=f"{count} frames")
        self.refresh()

    def next_image(self, step: int = 1) -> None:
        if self._is_grid_mode():
            self._grid_step(step)
        else:
            total = len(self.data_service)
            if not total:
                return
            self._idx = min(self._idx + step, total - 1)
            self.refresh()

    def previous_image(self, step: int = 1) -> None:
        if self._is_grid_mode():
            self._grid_step(-step)
        else:
            if not len(self.data_service):
                return
            self._idx = max(self._idx - step, 0)
            self.refresh()

    def _grid_step(self, delta: int) -> None:
        max_total = max((g.total() for g in self.view_groups), default=0)
        if not max_total:
            return
        self._idx = max(0, min(self._idx + delta, max_total - 1))
        ratio = self._idx / max(1, max_total - 1)
        for grp in self.view_groups:
            grp.sync_ratio(ratio)
        self.refresh()

    def first_image(self) -> None:
        self._idx = 0
        if self._is_grid_mode():
            for g in self.view_groups:
                g.set_idx(0)
        self.refresh()

    def last_image(self) -> None:
        if self._is_grid_mode():
            max_total = max((g.total() for g in self.view_groups), default=0)
            self._idx = max(0, max_total - 1)
            for g in self.view_groups:
                g.set_idx(g.total() - 1)
        else:
            self._idx = max(0, len(self.data_service) - 1)
        self.refresh()

    def on_nav_slider(self, val) -> None:
        if self._slider_updating:
            return
        idx = int(float(val))
        if self._is_grid_mode():
            max_total = max((g.total() for g in self.view_groups), default=0)
            if not max_total:
                return
            ratio = idx / max(1, max_total - 1)
            self._idx = idx
            for g in self.view_groups:
                g.sync_ratio(ratio)
        else:
            if not len(self.data_service):
                return
            self._idx = max(0, min(idx, len(self.data_service) - 1))
        self.refresh()

    def toggle_play(self) -> None:
        if self._playing:
            self._stop()
        else:
            self._start()

    def _start(self) -> None:
        self._playing = True
        self.player.play_btn.config(text="⏸")
        self.player.play_status.config(text="▶", fg=W["green"])
        self._tick()

    def _stop(self) -> None:
        self._playing = False
        self.player.play_btn.config(text="▶")
        self.player.play_status.config(text="⏸", fg=W["dim"])

    def _tick(self) -> None:
        if not self._playing:
            return
        t0 = time.perf_counter()
        fwd = self.play_direction_var.get() == "forward"

        if self._is_grid_mode():
            max_total = max((g.total() for g in self.view_groups), default=0)
            if not max_total:
                self._stop()
                return
            if fwd:
                if self._idx >= max_total - 1:
                    if self.loop_var.get():
                        self._idx = 0
                    else:
                        self._stop()
                        return
                else:
                    self._idx += 1
            else:
                if self._idx <= 0:
                    if self.loop_var.get():
                        self._idx = max_total - 1
                    else:
                        self._stop()
                        return
                else:
                    self._idx -= 1
            ratio = self._idx / max(1, max_total - 1)
            for g in self.view_groups:
                g.sync_ratio(ratio)
        else:
            total = len(self.data_service)
            if not total:
                self._stop()
                return
            if fwd:
                if self._idx >= total - 1:
                    if self.loop_var.get():
                        self._idx = 0
                    else:
                        self._stop()
                        return
                else:
                    self._idx += 1
            else:
                if self._idx <= 0:
                    if self.loop_var.get():
                        self._idx = total - 1
                    else:
                        self._stop()
                        return
                else:
                    self._idx -= 1

        self.refresh()
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        next_delay = max(1, self._play_delay - elapsed_ms)
        self.root.after(next_delay, self._tick)

    def on_fps_move(self, value) -> None:
        if self._fps_lock:
            return

        closest = min(FPS_PRESETS, key=lambda k: abs(k - float(value)))
        txt, delay = FPS_PRESETS[closest]
        self._play_delay = delay
        try:
            self.player.fps_label.config(text=txt.split()[0])
        except AttributeError:
            pass

    def on_fps_release(self, event: tkinter.Event) -> None:
        cur = self.player.fps_slider.get()
        closest = min(FPS_PRESETS, key=lambda k: abs(k - cur))
        self._fps_lock = True
        self.player.fps_slider.set(closest)
        self._fps_lock = False
        txt, delay = FPS_PRESETS[closest]
        self._play_delay = delay
        self.player.fps_label.config(text=txt.split()[0])

    def toggle_fullscreen(self, event: tkinter.Event | None = None) -> None:
        cur = self.root.attributes('-fullscreen')
        self.root.attributes('-fullscreen', not cur)
        self.topbar.fs_btn.config(text="⊡" if not cur else "⛶")
        self.refresh()

    def exit_fullscreen(self, event: tkinter.Event | None = None) -> None:
        self.root.attributes('-fullscreen', False)
        self.topbar.fs_btn.config(text="⛶")
        self.refresh()

    def _on_canvas_resize(self, event: tk.Event) -> None:
        if self._layout_in_progress:
            return
        if self._rsz_tm:
            self.root.after_cancel(self._rsz_tm)
        self._rsz_tm = self.root.after(40, self.refresh)

    def _on_panel_resize(self) -> None:
        if self._layout_in_progress:
            return
        if self._rsz_tm:
            self.root.after_cancel(self._rsz_tm)
        self._rsz_tm = self.root.after(60, self.refresh)

    def _on_resize(self, event: tk.Event) -> None:
        if self._layout_in_progress:
            return
        if event.widget != self.root:
            return
        if self._rsz_tm:
            self.root.after_cancel(self._rsz_tm)
        self._rsz_tm = self.root.after(40, self.refresh)

    def _make_dp(self) -> DrawParams:
        return DrawParams(
            show_bbox=self.show_bbox_var.get(),
            show_bbox_fill=self.show_bbox_fill_var.get(),
            bbox_fill_alpha=self.bbox_alpha_var.get(),
            show_keypoints=self.show_kpts_var.get(),
            show_skeleton=self.show_skel_var.get(),
            show_joint_ids=self.show_jids_var.get(),
            show_ann_ids=self.show_aids_var.get(),
            show_frame_label=self.show_label_var.get(),
            show_tracks=self.show_tracks_var.get(),
            track_length=max(1, int(self.track_len_var.get())),
            track_alpha=self.track_alpha_var.get(),
            point_radius=self.radius_var.get(),
            font_scale=self.font_var.get(),
            skel_thickness=max(1, int(self.skel_thick_var.get())),
        )

    def _set_slider(self, idx: int, total: int) -> None:
        self._slider_updating = True
        try:
            hi = max(1, total - 1)
            self.player.progress.config(from_=0, to=hi)
            self.player.progress.set(max(0, min(idx, hi)))
        finally:
            self._slider_updating = False

    def refresh(self) -> None:
        dp = self._make_dp()
        if self._is_grid_mode():
            self._refresh_grid(dp)
        else:
            self._refresh_single(dp)

    def _refresh_grid(self, dp: DrawParams) -> None:
        max_total = max((g.total() for g in self.view_groups), default=0)
        if max_total == 0:
            self.player.frame_label.config(text="0 / 0")
            self.player.id_label.config(text="No data")
            for panel in self.grid_canvas.get_panels():
                cw = max(panel.canvas.winfo_width(), 100)
                ch = max(panel.canvas.winfo_height(), 100)
                panel.canvas.delete("all")
                panel.canvas.create_text(
                    cw // 2, ch // 2,
                    text="Load annotation file\n(Toolbar → Files → Open annotation)",
                    fill="#555", font=(self.font, 11), justify=tk.CENTER)
            return

        self.player.frame_label.config(text=f"{self._idx + 1} / {max_total}")
        self._set_slider(self._idx, max_total)

        visible_groups = [self.grid_canvas._expanded.group] if self.grid_canvas._expanded else [p.group for p in self.grid_canvas.get_panels()]
        self.info.update_groups(visible_groups)

        if visible_groups:
            grp = visible_groups[0]
            entry = grp.current_entry()
            if entry:
                try:
                    fn = int(Path(entry.file_name).stem)
                    self.player.id_label.config(text=f"ID:{fn+1}  {grp.cfg.name}")
                except ValueError:
                    self.player.id_label.config(text=Path(entry.file_name).name)

        self.grid_canvas.draw_all(dp)

    def _refresh_single(self, dp: DrawParams) -> None:
        total = len(self.data_service)
        entry = self.data_service.get(self._idx)
        if entry is None:
            self.player.frame_label.config(text="0 / 0")
            self.canvas.delete("all")
            if not self.data_service.all_entries:
                cw = max(self.canvas.winfo_width(), 100)
                ch = max(self.canvas.winfo_height(), 100)
                self.canvas.create_text(
                    cw // 2, ch // 2,
                    text="Load annotation file\n(Toolbar → Files → Open annotation)",
                    fill="#555", font=(self.font, 13), justify=tk.CENTER)
            return

        self.player.frame_label.config(text=f"{self._idx + 1} / {total}")
        self._set_slider(self._idx, total)
        try:
            fn = int(Path(entry.file_name).stem)
            self.player.id_label.config(text=f"ID:{fn+1}  CAM:—")
        except ValueError:
            self.player.id_label.config(text=Path(entry.file_name).name)

        self.info.update_ann(entry)
        self._render_single(entry, dp)

    def _render_single(self, entry: ImageEntry, dp: DrawParams) -> None:
        canvas = self.canvas
        canvas.delete("all")
        cw = max(canvas.winfo_width(), 100)
        ch = max(canvas.winfo_height(), 100)

        img = self.loader.load(entry.file_name)
        if img is None:
            msg = ("Select the folder with images (Toolbar → Files → Open images folder)"
                   if not self.loader.directory
                   else f"File not found:\n{entry.file_name}")
            canvas.create_text(cw // 2, ch // 2, text=msg, fill="#555", font=(self.font, 13), justify=tk.CENTER)
            return

        img = img.copy()
        oh, ow = img.shape[:2]
        tw = (self.tracker.get_window(self._idx, dp.track_length) if dp.show_tracks else None)
        draw_frame(img, entry.annotations, dp, tw)
        if dp.show_frame_label:
            try:
                fn = int(Path(entry.file_name).stem)
                outlined(img, f"ID={fn+1}", (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (120, 210, 255), 2)
            except ValueError:
                pass

        nw, nh = fit_image(ow, oh, cw, ch)
        resized = resize(img, nw, nh)
        self._photo = rgb_to_photoimage(resized)
        canvas.create_image((cw - nw) // 2, (ch - nh) // 2, anchor=tk.NW, image=self._photo)
