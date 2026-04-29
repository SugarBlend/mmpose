import tkinter as tk
from tkinter import ttk
from typing import Callable, Any


from project.visualizers.space.constants import W
from project.visualizers.space.style import SliderRow, Toggle, TogRow, vsep


class _LabelProxy:

    def __init__(
        self,
        str_var: tk.StringVar,
        color_setter: Callable[[str], None] | None = None,
    ) -> None:

        self._var = str_var
        self._color_setter = color_setter

    def config(
        self,
        text: str | None = None,
        fg: str | None = None,
        **_: Any,
    ) -> None:

        if text is not None:
            self._var.set(text)

        if fg is not None and self._color_setter:
            self._color_setter(fg)

    def cget(self, key: str) -> str:
        return self._var.get() if key == "text" else ""


class TopBar(tk.Frame):
    H = 22

    def __init__(self, parent: tk.Frame, controller: Any) -> None:
        super().__init__(parent, bg=W["toolbar"], height=self.H)
        self.pack_propagate(False)
        self.controller: Any = controller

        self.font = "Comic Sans MS"
        self._file_text = tk.StringVar(value="No files selected")
        self._file_color: str = W["dim"]
        self._filter_text = tk.StringVar(value="")

        self.file_info = _LabelProxy(
            self._file_text,
            lambda c: setattr(self, "_file_color", c),
        )

        self.filter_lbl = _LabelProxy(self._filter_text)

        self._cur_btn: tk.Label | None = None
        self._cur_pop: tk.Toplevel | None = None

        self._build()

    def _build(self) -> None:
        bg = W["toolbar"]
        # tk.Label(self, text="Joints Vis", bg=bg, fg=W["accent"], font=(self.font, 8, "bold"), padx=8).pack(side=tk.LEFT)
        vsep(self, W["border"]).pack(side=tk.LEFT, fill=tk.Y, pady=3)

        menus: list[tuple[str, Callable[[tk.Frame], None]]] = [
            ("Files", self._build_files),
            ("Filters", self._build_filter),
            ("Groups", self._build_groups_menu),
            ("Display", self._build_display),
            ("Style", self._build_style),
            ("Tracks", self._build_tracks),
            ("Help", self._build_keys),
        ]

        for header, builder in menus:
            label = tk.Label(self, text=header, bg=bg, fg=W["label"], font=(self.font, 8), padx=9, cursor="hand2")
            label.pack(side=tk.LEFT, fill=tk.Y)
            label.bind("<Button-1>", lambda e, bl=builder, bt=label: self._toggle(bl, bt))
            label.bind("<Enter>", lambda e, bt=label: self._btn_hover(bt, True))
            label.bind("<Leave>", lambda e, bt=label: self._btn_hover(bt, False))

        self.fs_btn = tk.Label(self, text="⛶", bg=bg, fg=W["dim"], font=(self.font, 10), padx=6, cursor="hand2")
        self.fs_btn.pack(side=tk.RIGHT, fill=tk.Y)
        self.fs_btn.bind("<Button-1>", lambda e: self.controller.toggle_fullscreen())
        self.fs_btn.bind("<Enter>", lambda e: self.fs_btn.config(fg=W["accent"]))
        self.fs_btn.bind("<Leave>", lambda e: self.fs_btn.config(fg=W["dim"]))

        vsep(self, W["border"]).pack(side=tk.RIGHT, fill=tk.Y, pady=3)

        self.status_label = tk.Label(self, text="Files not uploaded", bg=bg, fg=W["dim"], font=(self.font, 8), padx=6)
        self.status_label.pack(side=tk.RIGHT)

    def _btn_hover(self, btn: tk.Label, entering: bool) -> None:

        if btn is self._cur_btn:
            return

        btn.config(bg=W["tab_h"] if entering else W["toolbar"])

    def _toggle(
        self,
        builder: Callable[[tk.Frame], None],
        btn: tk.Label,
    ) -> None:

        if self._cur_btn is btn:
            self._close_popup()
        else:
            self._close_popup()
            self._open_popup(builder, btn)

    def _open_popup(
        self,
        builder: Callable[[tk.Frame], None],
        btn: tk.Label,
    ) -> None:

        self._cur_btn = btn

        btn.config(bg=W["tab_act"], fg=W["accent"])

        pop = tk.Toplevel(self.winfo_toplevel())

        pop.overrideredirect(True)
        pop.attributes("-topmost", True)
        pop.configure(bg=W["border"])

        body = tk.Frame(
            pop,
            bg=W["surface"],
            highlightbackground=W["border"],
            highlightthickness=1,
        )

        body.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)

        builder(body)

        pop.update_idletasks()
        self.winfo_toplevel().update_idletasks()

        bx = btn.winfo_rootx()
        by = btn.winfo_rooty() + btn.winfo_height() + 1

        pw = pop.winfo_reqwidth()
        sw = self.winfo_screenwidth()

        x = max(0, min(bx, sw - pw - 4))

        pop.geometry(f"+{x}+{by}")

        pop.bind("<FocusOut>", lambda e: self._on_focusout(pop))
        pop.focus_set()

        self._cur_pop = pop

    def _on_focusout(self, pop: tk.Toplevel) -> None:
        pop.after(80, lambda: self._check_focusout(pop))

    def _check_focusout(self, pop: tk.Toplevel) -> None:
        try:
            fw = self.winfo_toplevel().focus_get()
            if fw is not None and str(fw).startswith(str(pop)):
                return
        except Exception:
            pass
        self._close_popup()

    def _close_popup(self) -> None:
        if self._cur_pop:
            try:
                self._cur_pop.destroy()
            except Exception:
                pass
            self._cur_pop = None

        if self._cur_btn:
            self._cur_btn.config(bg=W["toolbar"], fg=W["label"])
            self._cur_btn = None

    def _sec(self, parent: tk.Frame, title: str) -> tk.Frame:
        tk.Label(parent, text=title, bg=W["surf2"], fg=W["dim"], font=(self.font, 7, "bold"), padx=8, pady=2,
                 anchor="w").pack(fill=tk.X)
        frame = tk.Frame(parent, bg=W["surface"])
        frame.pack(fill=tk.X, padx=8, pady=(3, 6))
        return frame

    def _action(
        self,
        parent: tk.Frame,
        text: str,
        func: Callable[[], None],
    ) -> None:
        label = tk.Label(parent, text=text, bg=W["surf2"], fg=W["text"], font=(self.font, 8), padx=10, pady=5,
                         cursor="hand2", anchor="w")
        label.pack(fill=tk.X)

        label.bind("<Button-1>", lambda e: func())
        label.bind("<Enter>", lambda e: label.config(bg=W["border"]))
        label.bind("<Leave>", lambda e: label.config(bg=W["surf2"]))

    @staticmethod
    def _sep(parent: tk.Frame) -> None:
        tk.Frame(parent, height=1, bg=W["sep"]).pack(fill=tk.X)


    def _build_files(self, parent: tk.Frame) -> None:
        bg = W["surface"]
        self._action(parent,"Open annotation", self.controller.load_annotations)
        self._sep(parent)
        self._action(parent,"Open images folder", self.controller.select_images_dir)
        self._sep(parent)
        info = tk.Label(parent, textvariable=self._file_text, bg=bg, fg=W["dim"], font=(self.font, 8), padx=10,
                        pady=5, anchor="w", justify=tk.LEFT, wraplength=260)
        info.pack(fill=tk.X)

        def _sync(*_):
            try:
                info.config(fg=self._file_color)
            except:
                pass

        self._file_text.trace_add("write", _sync)
        _sync()

    def _build_filter(self, parent: tk.Frame) -> None:
        bg = W["surface"]
        self._sec(parent,"regex filters".upper())
        container = tk.Frame(parent, bg=bg)
        container.pack(fill=tk.X, padx=8, pady=(0, 4))
        self.controller._filter_rows = []

        def rebuild():
            for w in container.winfo_children():
                w.destroy()

            for i, rd in enumerate(self.controller._filter_rows):
                if i > 0:
                    op_var = rd["op_var"]
                    op_row = tk.Frame(container, bg=bg)
                    op_row.pack(fill=tk.X, pady=(2, 0))
                    for op_val in ("OR", "AND"):
                        is_sel = op_var.get() == op_val
                        bb = tk.Label(op_row, text=op_val, bg=W["accent"] if is_sel else W["surf2"],
                                      fg="white" if is_sel else W["dim"], font=(self.font, 7, "bold"), padx=7, pady=1,
                                      cursor="hand2")
                        bb.pack(side=tk.LEFT, padx=(0, 2))
                        bb.bind("<Button-1>", lambda e, v = op_var, val = op_val: (v.set(val), rebuild()))
                line = tk.Frame(container, bg=bg)
                line.pack(fill=tk.X, pady=(2, 0))
                entry = ttk.Entry(line, textvariable=rd["var"], font=("Consolas", 8))
                entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
                if len(self.controller._filter_rows) > 1:
                    ii = i
                    rm = tk.Label(line, text=" − ", bg=W["surf2"], fg=W["red"],
                                  font=(self.font, 9, "bold"), padx=4, cursor="hand2")
                    rm.pack(side=tk.LEFT, padx=(3, 0))
                    rm.bind("<Button-1>", lambda e, _i = ii: remove_row(_i))
                    rm.bind("<Enter>", lambda e, b = rm: b.config(bg=W["border"]))
                    rm.bind("<Leave>", lambda e, b = rm: b.config(bg=W["surf2"]))

        def add_row() -> None:
            self.controller._filter_rows.append({"var": tk.StringVar(value=""), "op_var": tk.StringVar(value="OR")})
            rebuild()

        def remove_row(idx: int) -> None:
            if len(self.controller._filter_rows) > 1:
                self.controller._filter_rows.pop(idx)
                rebuild()

        saved_pats = self.controller.regex_patterns or [".*"]
        saved_ops = self.controller.regex_operators
        for i, pat in enumerate(saved_pats):
            op_var = tk.StringVar(value=saved_ops[i - 1] if i > 0 and i - 1 < len(saved_ops) else "OR")
            self.controller._filter_rows.append({"var": tk.StringVar(value=pat), "op_var": op_var})
        rebuild()

        add_btn = tk.Label(parent, text="  +  Add pattern  ", bg=W["surf2"], fg=W["accent"],
                           font=(self.font, 8), padx=8, pady=3, cursor="hand2", anchor="w")
        add_btn.pack(fill=tk.X, padx=8, pady=(0, 4))
        add_btn.bind("<Button-1>", lambda e: add_row())
        add_btn.bind("<Enter>", lambda e: add_btn.config(bg=W["border"]))
        add_btn.bind("<Leave>", lambda e: add_btn.config(bg=W["surf2"]))

        self._sep(parent)
        foot = tk.Frame(parent, bg=W["surface"])
        foot.pack(fill=tk.X, padx=8, pady=4)
        apply = tk.Label(foot, text="Apply", bg=W["accent"], fg="white", font=(self.font, 8), padx=10, pady=3,
                         cursor="hand2")
        apply.pack(side=tk.LEFT)
        apply.bind("<Button-1>", lambda e: self.controller.apply_filter())
        apply.bind("<Enter>", lambda e: apply.config(bg=W["acc_h"]))
        apply.bind("<Leave>", lambda e: apply.config(bg=W["accent"]))
        status=tk.Label(foot, textvariable=self._filter_text, bg=W["surface"], fg=W["green"],
                        font=(self.font, 8), padx=6)
        status.pack(side=tk.LEFT)

    def _build_groups_menu(self, parent: tk.Frame) -> None:
        self._action(parent,"⊞  Open group manager", lambda: (self._close_popup(), self.controller.open_group_manager()))
        self._sep(parent)

        if self.controller.view_groups:
            frame = self._sec(parent,"active groups".upper())
            for i, group in enumerate(self.controller.view_groups):
                row = tk.Frame(frame, bg=W["surface"])
                row.pack(fill=tk.X, pady=1)
                dot = tk.Canvas(row,width=10, height=10, bg=W["surface"], highlightthickness=0)
                dot.create_oval(1, 1, 9, 9, fill=group.cfg.color, outline="")
                dot.pack(side=tk.LEFT, padx=(0, 6))
                tk.Label(row, text=group.cfg.name, bg=W["surface"], fg=W["text"],
                         font=(self.font, 8), anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)
                tk.Label(row, text=f"{group.total()}fr", bg=W["surface"], fg=W["dim"],
                         font=(self.font, 7)).pack(side=tk.RIGHT)
            self._sep(parent)
            self._action(parent,"✕  Clear groups (single view)", self.controller.clear_view_groups)
        else:
            frame = self._sec(parent,"NO GROUPS")
            tk.Label(frame, text="Use single-view filter or\nopen group manager to create groups.",
                     bg=W["surface"], fg=W["dim"], font=(self.font, 8), justify=tk.LEFT).pack(anchor="w")

    def _build_display(self, parent: tk.Frame) -> None:
        bg = W["surface"]
        frame = self._sec(parent,"show".upper())
        for lbl,var in [
            ("Bbox", self.controller.show_bbox_var),
            ("Bbox filling", self.controller.show_bbox_fill_var),
            ("Joints", self.controller.show_kpts_var),
            ("Skeletons", self.controller.show_skel_var),
            ("Joints number", self.controller.show_jids_var),
            ("Annotation number",self.controller.show_aids_var),
            ("Label ID/CAM", self.controller.show_label_var),
        ]:
            row = tk.Frame(frame, bg=bg)
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=lbl, bg=bg, fg=W["text"], font=(self.font, 8),
                     anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)
            Toggle(row, var, command=self.controller.refresh, bg=bg).pack(side=tk.RIGHT)

    def _build_style(self, parent: tk.Frame) -> None:
        bg = W["surface"]
        frame = self._sec(parent,"draw settings".upper())
        for label, var, from_, to, fmt in [
            ("Points radius", self.controller.radius_var, 0.5, 15, "{:.1f}"),
            ("Skeleton thickness", self.controller.skel_thick_var, 1, 8, "{:.0f}"),
            ("Bbox transparency", self.controller.bbox_alpha_var, 0, 0.5, "{:.2f}"),
            ("Font size", self.controller.font_var, 0.2, 1.5, "{:.1f}"),
        ]:
            SliderRow(frame, label, var, from_, to, fmt, command=self.controller.refresh, bg=bg).pack(fill=tk.X, pady=1)

    def _build_tracks(self, parent: tk.Frame) -> None:
        bg = W["surface"]
        frame = self._sec(parent,"Keypoint tracks".upper())
        TogRow(frame,"Enable tracks", self.controller.show_tracks_var,
               command=self.controller.refresh, bg=bg).pack(fill=tk.X, pady=2)
        tk.Frame(frame, height=1, bg=W["sep"]).pack(fill=tk.X, pady=3)
        SliderRow(frame,"Length (frames)", self.controller.track_len_var,5,150,"{:.0f}",
                  command=self.controller.refresh, bg=bg).pack(fill=tk.X, pady=1)
        SliderRow(frame,"Brightness", self.controller.track_alpha_var,0.05,1.0,
                  command=self.controller.refresh, bg=bg).pack(fill=tk.X, pady=1)

    def _build_keys(self, parent: tk.Frame) -> None:
        bg = W["surface"]
        frame = self._sec(parent,"Hot keys".upper())
        for key, desc in [
            ("← / →", "frame ±1"),
            ("↑ / ↓", "frame ±10"),
            ("Space", "play / pause"),
            ("F11", "Fullscreen"),
            ("Esc", "Exit from FS"),
            ("LMB ×2", "Expand panel (groups mode)"),
            ("RMB", "Back to grid"),
        ]:
            row = tk.Frame(frame, bg=bg)
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=key, bg=W["surf2"], fg=W["accent"],
                     font=("Consolas",8), padx=5, width=10, anchor="center").pack(side=tk.LEFT)
            tk.Label(row, text=desc, bg=bg, fg=W["label"], font=(self.font, 8)).pack(side=tk.LEFT, padx=8)
