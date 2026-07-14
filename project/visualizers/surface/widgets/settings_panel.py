import tkinter as tk
from project.visualizers.surface.preprocessing import RenderSettings


class SettingsPanel(tk.Toplevel):

    _BG = "#C9C9C9"
    _FG = "#080808"
    _CARD_BG = "#B8B4B4"
    _SEP = "#9A9696"
    _ACCENT = "#545252"
    _SLIDER_BG = "#C9C9C9"
    _SLIDER_TR = "#545252"
    _SLIDER_FG = "#F5F5F5"
    _BTN_BG = "#ADA8A8"

    def __init__(self, parent: tk.Tk, rs: RenderSettings, on_change: callable) -> None:
        super().__init__(parent)
        self._rs = rs
        self._on_change = on_change
        self._parent = parent

        self.title("")
        self.configure(bg=self._BG)
        self.resizable(False, False)
        self.overrideredirect(True)
        self.withdraw()
        self.transient(parent)

        # thin border frame
        border = tk.Frame(self, bg=self._ACCENT, bd=0)
        border.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        self._inner = tk.Frame(border, bg=self._BG)
        self._inner.pack(fill=tk.BOTH, expand=True)

        self._build()

        # close when main window gets focus
        parent.bind("<FocusIn>", self._on_parent_focus, add="+")
        self.bind("<FocusOut>", self._on_self_focus_out, add="+")

    def _section(self, parent: tk.Widget, title: str) -> tk.Frame:
        outer = tk.Frame(parent, bg=self._BG)
        outer.pack(fill=tk.X, padx=10, pady=(10, 0))
        tk.Label(outer, text=title, bg=self._BG, fg=self._ACCENT,
                 font=("SegoeUI", 8, "bold")).pack(anchor=tk.W, pady=(0, 3))
        card = tk.Frame(outer, bg=self._CARD_BG, padx=10, pady=8)
        card.pack(fill=tk.X)
        return card

    def _separator(self, parent: tk.Widget) -> None:
        tk.Frame(parent, bg=self._SEP, height=1).pack(fill=tk.X, padx=10, pady=(10, 0))

    def _build(self) -> None:
        root_frame = self._inner

        parts_card = self._section(root_frame, "BODY PARTS")

        self._part_vars: dict[str, tk.BooleanVar] = {}
        part_cfg = [
            ("body",  "🦴  Body (COCO-17 skeleton)"),
            ("face",  "😶  Face (face mesh, 68 pts)"),
            ("hands", "🤚  Hands (palm keypoints)"),
            ("feet",  "🦶  Feet (toe & heel points)"),
        ]
        for part_key, label in part_cfg:
            var = tk.BooleanVar(value=self._rs.bpart_filter.is_enabled(part_key))
            self._part_vars[part_key] = var
            cb = tk.Checkbutton(
                parts_card, text=label, variable=var,
                command=lambda k=part_key, v=var: self._on_part_toggle(k, v),
                bg=self._CARD_BG, fg=self._FG,
                selectcolor=self._BG,
                activebackground=self._CARD_BG, activeforeground=self._FG,
                font=("SegoeUI", 10), anchor=tk.W, cursor="hand2",
            )
            cb.pack(fill=tk.X, pady=1)

        self._separator(root_frame)

        # Side colouring
        colour_card = self._section(root_frame, "SIDE COLOURING")
        self._side_var = tk.BooleanVar(value=self._rs.use_side_color)
        tk.Checkbutton(
            colour_card, text="Colour by side  (Left / Right / Center)",
            variable=self._side_var, command=self._on_side_toggle,
            bg=self._CARD_BG, fg=self._FG,
            selectcolor=self._BG,
            activebackground=self._CARD_BG, activeforeground=self._FG,
            font=("SegoeUI", 10), anchor=tk.W, cursor="hand2",
        ).pack(fill=tk.X)

        self._separator(root_frame)

        # Size sliders
        size_card = self._section(root_frame, "SIZES")

        self._joint_r_var = tk.DoubleVar(value=self._rs.joint_r)
        self._make_slider(size_card, "Joint radius",
                          self._joint_r_var, 1.0, 15.0, 0.5, "px",
                          self._on_joint_r)

        self._limb_w_var = tk.DoubleVar(value=self._rs.limb_w)
        self._make_slider(size_card, "Connection thickness",
                          self._limb_w_var, 1.0, 12.0, 0.5, "px",
                          self._on_limb_w)

        self._separator(root_frame)

        # Score threshold
        thr_card = self._section(root_frame, "VISIBILITY")

        self._score_thr_var = tk.DoubleVar(value=self._rs.score_thr)
        self._make_slider(thr_card, "Score threshold",
                          self._score_thr_var, 0.0, 1.0, 0.01, "",
                          self._on_score_thr)

        tk.Frame(root_frame, bg=self._BG, height=10).pack()

    def _make_slider(self, parent, label, var, from_, to, resolution, unit, command) -> None:
        row = tk.Frame(parent, bg=self._CARD_BG)
        row.pack(fill=tk.X, pady=(4, 0))

        tk.Label(row, text=label, bg=self._CARD_BG, fg=self._FG, font=("SegoeUI", 10), width=22, anchor=tk.W).pack(side=tk.LEFT)

        val_lbl = tk.Label(row, text=f"{var.get():.1f} {unit}", bg=self._CARD_BG, fg=self._ACCENT,
                           font=("SegoeUI", 9), width=7)
        val_lbl.pack(side=tk.RIGHT)

        def _cb(v: str) -> None:
            val_lbl.config(text=f"{float(v):.1f} {unit}")
            command(float(v))

        tk.Scale(
            parent, variable=var, from_=from_, to=to,
            resolution=resolution, orient=tk.HORIZONTAL,
            showvalue=False, sliderlength=14, width=7,
            bg=self._SLIDER_BG, troughcolor=self._SLIDER_TR,
            fg=self._SLIDER_FG, activebackground=self._ACCENT,
            highlightthickness=0, bd=0, cursor="hand2",
            command=_cb,
        ).pack(fill=tk.X, pady=(2, 4))

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

    def _on_parent_focus(self, event: tk.Event) -> None:
        # Close only when focus lands on the main window body (not toolbar buttons)
        if event.widget is self._parent:
            self.close()

    def _on_self_focus_out(self, event: tk.Event) -> None:
        # FocusOut fires for every child widget too; only act on the window
        if event.widget is self:
            self.after(50, self._check_still_focused)

    def _check_still_focused(self) -> None:
        try:
            focused = self.focus_get()
        except (Exception,):
            focused = None
        if focused is None or str(focused) == str(self._parent):
            self.close()

    def close(self) -> None:
        if self.winfo_viewable():
            self.withdraw()

    def toggle(self, x: int, y: int) -> None:
        if self.winfo_viewable():
            self.withdraw()
        else:
            self.geometry(f"+{x}+{y}")
            self.deiconify()
            self.lift()
