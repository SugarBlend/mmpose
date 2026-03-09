import tkinter as tk
from tkinter import ttk
from typing import List, Dict, Any

from project.visualizer.constants import PANEL_BORDER_COLORS, W
from project.visualizer.models import ViewGroupConfig
from project.visualizer.widgets.view_group import ViewGroup


class GroupManagerDialog(tk.Toplevel):
    def __init__(self, parent: tk.Widget, controller: Any) -> None:
        super().__init__(parent)
        self.controller: Any = controller

        self.font: str = "Comic Sans MS"
        self.title("Groups manager")
        self.geometry("680x560")
        self.resizable(True, True)
        self.configure(bg=W["surface"])
        self.transient(parent)
        self.grab_set()
        self._rows: List[Dict[str, Any]] = []
        self._layout_var: tk.StringVar
        self._canvas: tk.Canvas
        self._inner: tk.Frame
        self._win_id: int
        self._build()
        self._load_existing()

    def _build(self) -> None:
        top = tk.Frame(self, bg=W["surface"])
        top.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

        hdr = tk.Frame(top, bg=W["surf2"])
        hdr.pack(fill=tk.X, pady=(0, 6))
        tk.Label(hdr, text="  View Groups", bg=W["surf2"], fg=W["accent"],
                 font=(self.font, 10, "bold"), pady=6).pack(side=tk.LEFT)
        tk.Label(hdr, text="Click group name to rename  •  scroll anywhere to browse",
                 bg=W["surf2"], fg=W["dim"],
                 font=(self.font, 7), padx=8).pack(side=tk.LEFT)

        scroll_wrap = tk.Frame(top, bg=W["surface"])
        scroll_wrap.pack(fill=tk.BOTH, expand=True)

        self._canvas = tk.Canvas(scroll_wrap, bg=W["surface"], highlightthickness=0)
        sb = ttk.Scrollbar(scroll_wrap, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=sb.set)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        self._inner = tk.Frame(self._canvas, bg=W["surface"])
        self._win_id = self._canvas.create_window((0, 0), window=self._inner, anchor="nw")

        self._inner.bind("<Configure>", lambda e: (
            self._canvas.configure(scrollregion=self._canvas.bbox("all")),
            self._canvas.itemconfig(self._win_id, width=self._canvas.winfo_width())
        ))
        self._canvas.bind("<Configure>", lambda e:
            self._canvas.itemconfig(self._win_id, width=e.width))

        self.bind_all("<MouseWheel>", lambda e: self._canvas.yview_scroll(int(-1 * e.delta / 120), "units"))
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

        bot = tk.Frame(self, bg=W["border"], height=1)
        bot.pack(fill=tk.X)
        btn_row = tk.Frame(self, bg=W["surface"])
        btn_row.pack(fill=tk.X, padx=12, pady=8)

        def add_group() -> None:
            i = len(self._rows)
            color = PANEL_BORDER_COLORS[i % len(PANEL_BORDER_COLORS)]
            self._add_row(ViewGroupConfig(name=f"Group {i + 1}", color=color))

        b_add = tk.Label(btn_row, text="+ Add group", bg=W["accent"], fg="white",
                         font=(self.font, 9), padx=10, pady=4, cursor="hand2")
        b_add.pack(side=tk.LEFT, padx=(0, 8))
        b_add.bind("<Button-1>", lambda e: add_group())
        b_add.bind("<Enter>", lambda e: b_add.config(bg=W["acc_h"]))
        b_add.bind("<Leave>", lambda e: b_add.config(bg=W["accent"]))

        # Apply
        b_apply = tk.Label(btn_row, text="Apply", bg=W["green"], fg="white",
                           font=(self.font, 9, "bold"), padx=14, pady=4, cursor="hand2")
        b_apply.pack(side=tk.RIGHT, padx=(8, 0))
        b_apply.bind("<Button-1>", lambda e: self._on_apply())
        b_apply.bind("<Enter>", lambda e: b_apply.config(bg="#155f2d"))
        b_apply.bind("<Leave>", lambda e: b_apply.config(bg=W["green"]))

        # cancel
        b_cancel = tk.Label(btn_row, text="Cancel", bg=W["surf2"], fg=W["text"],
                            font=(self.font, 9), padx=10, pady=4, cursor="hand2")
        b_cancel.pack(side=tk.RIGHT)
        b_cancel.bind("<Button-1>", lambda e: self._on_cancel())
        b_cancel.bind("<Enter>", lambda e: b_cancel.config(bg=W["border"]))
        b_cancel.bind("<Leave>", lambda e: b_cancel.config(bg=W["surf2"]))

        # mode switcher
        mode_row = tk.Frame(btn_row, bg=W["surface"])
        mode_row.pack(side=tk.LEFT, padx=16)
        tk.Label(mode_row, text="Layout:", bg=W["surface"], fg=W["dim"],
                 font=(self.font, 8)).pack(side=tk.LEFT)
        self._layout_var = tk.StringVar(value=self.controller.grid_layout)
        for lbl, val in [("Auto", "auto"), ("1×N", "1xN"), ("N×1", "Nx1"), ("2×2", "2x2"), ("2×3", "2x3")]:
            rb = tk.Radiobutton(mode_row, text=lbl, variable=self._layout_var,
                                value=val, bg=W["surface"], fg=W["text"],
                                font=(self.font, 8), activebackground=W["surface"],
                                selectcolor=W["accent"], cursor="hand2")
            rb.pack(side=tk.LEFT, padx=3)

    def _load_existing(self) -> None:
        existing: List[Any] = self.controller.view_groups or []
        if not existing:
            self._add_row(ViewGroupConfig(name="Group 1", color=PANEL_BORDER_COLORS[0]))
        for grp in existing:
            cfg: ViewGroupConfig = grp.cfg if isinstance(grp, ViewGroup) else grp
            self._add_row(cfg)

    def _add_row(self, cfg: ViewGroupConfig) -> None:
        row_data: Dict[str, Any] = {}
        bg = W["surface"]

        frame = tk.Frame(self._inner, bg=W["surf2"], highlightthickness=2, highlightbackground=cfg.color)
        frame.pack(fill=tk.X, pady=4, padx=2)

        # header for group
        hdr = tk.Frame(frame, bg=cfg.color)
        hdr.pack(fill=tk.X)

        # drag handle
        drag_lbl = tk.Label(hdr, text="⠿", bg=cfg.color, fg="white",
                            font=(self.font, 11), padx=6, cursor="fleur")
        drag_lbl.pack(side=tk.LEFT, pady=2)

        # pencil
        pencil = tk.Label(hdr, text="✎", bg=cfg.color, fg="white",
                          font=(self.font, 9), padx=2)
        pencil.pack(side=tk.LEFT, padx=(0, 2), pady=4)

        name_var = tk.StringVar(value=cfg.name)
        row_data["name_var"] = name_var

        name_entry = tk.Entry(hdr, textvariable=name_var, bg=cfg.color, fg="white",
                              font=(self.font, 9, "bold"), relief="flat",
                              insertbackground="white", bd=0,
                              highlightthickness=1, highlightcolor="white",
                              highlightbackground=cfg.color)
        name_entry.pack(side=tk.LEFT, padx=(2, 8), pady=4, fill=tk.X, expand=True)
        name_entry.bind("<FocusIn>", lambda e, w=name_entry: w.config(highlightbackground="white"))
        name_entry.bind("<FocusOut>", lambda e, w=name_entry, c=cfg.color: w.config(highlightbackground=c))

        # remove button
        rm = tk.Label(hdr, text=" ✕ ", bg=cfg.color, fg="white",
                      font=(self.font, 9), cursor="hand2", padx=4)
        rm.pack(side=tk.RIGHT)
        rm.bind("<Button-1>", lambda e, f=frame, rd=row_data: self._remove_row(f, rd))

        # drag-and-drop
        self._bind_drag(drag_lbl, row_data)

        # patterns frame
        pats_frame = tk.Frame(frame, bg=bg)
        pats_frame.pack(fill=tk.X, padx=8, pady=(4, 2))
        row_data["pats_frame"] = pats_frame
        row_data["pat_rows"] = []
        row_data["frame"] = frame
        row_data["color"] = cfg.color

        def rebuild_pats(rd: Dict[str, Any] = row_data) -> None:
            for w in rd["pats_frame"].winfo_children():
                w.destroy()
            for pi, pr in enumerate(rd["pat_rows"]):
                # operators
                if pi > 0:
                    op_row = tk.Frame(rd["pats_frame"], bg=bg)
                    op_row.pack(fill=tk.X, pady=(1, 0))
                    for op_val in ("OR", "AND"):
                        is_sel = pr["op_var"].get() == op_val
                        bb = tk.Label(op_row, text=op_val,
                                      bg=rd["color"] if is_sel else W["surf2"],
                                      fg="white" if is_sel else W["dim"],
                                      font=(self.font, 7, "bold"),
                                      padx=6, pady=1, cursor="hand2")
                        bb.pack(side=tk.LEFT, padx=(0, 2))
                        bb.bind("<Button-1>",
                                lambda e, v=pr["op_var"], val=op_val, _rd=rd: (v.set(val), rebuild_pats(_rd)))

                line = tk.Frame(rd["pats_frame"], bg=bg)
                line.pack(fill=tk.X, pady=(1, 0))
                e = ttk.Entry(line, textvariable=pr["var"], font=("Consolas", 8))
                e.pack(side=tk.LEFT, fill=tk.X, expand=True)

                if len(rd["pat_rows"]) > 1:
                    pii = pi
                    rm2 = tk.Label(line, text=" − ", bg=W["surf2"], fg=W["red"],
                                   font=(self.font, 9, "bold"), cursor="hand2", padx=3)
                    rm2.pack(side=tk.LEFT, padx=(3, 0))
                    rm2.bind("<Button-1>", lambda e, _pi=pii, _rd=rd: (_rd["pat_rows"].pop(_pi), rebuild_pats(_rd)))
                    rm2.bind("<Enter>", lambda e, b=rm2: b.config(bg=W["border"]))
                    rm2.bind("<Leave>", lambda e, b=rm2: b.config(bg=W["surf2"]))

        row_data["rebuild"] = rebuild_pats

        # loading patterns
        for pi, pat in enumerate(cfg.patterns):
            op_var = tk.StringVar(value=cfg.operators[pi - 1] if pi > 0 and pi - 1 < len(cfg.operators) else "OR")
            row_data["pat_rows"].append({"var": tk.StringVar(value=pat), "op_var": op_var})
        rebuild_pats()

        # add pattern button
        add_p = tk.Label(frame, text="  + pattern", bg=bg, fg=cfg.color,
                         font=(self.font, 8), padx=8, pady=2, cursor="hand2", anchor="w")
        add_p.pack(fill=tk.X)
        add_p.bind("<Button-1>", lambda e, rd=row_data: rd["pat_rows"].append({"var": tk.StringVar(value=""), "op_var": tk.StringVar(value="OR")}) or rd["rebuild"]())

        self._rows.append(row_data)

    def _bind_drag(self, handle: tk.Label, row_data: Dict[str, Any]) -> None:
        state: Dict[str, Any] = {}

        def _get_insert_pos(y_root: int) -> int:
            for i, rd in enumerate(self._rows):
                f = rd["frame"]
                fy = f.winfo_rooty()
                fh = f.winfo_height()
                if y_root < fy + fh // 2:
                    return i
            return len(self._rows)

        def on_press(e: tk.Event) -> None:
            state["active"] = True
            state["rd"] = row_data
            ind = tk.Frame(self._inner, bg=row_data["color"], height=3)
            state["ind"] = ind

        def on_motion(e: tk.Event) -> None:
            if not state.get("active"):
                return
            y_root = e.widget.winfo_rooty() + e.y
            pos = _get_insert_pos(y_root)
            state["pos"] = pos

            ind = state["ind"]
            frames = [rd["frame"] for rd in self._rows]
            if not frames:
                return
            ind.place_forget()
            if pos < len(frames):
                ref = frames[pos]
                iy = ref.winfo_y() - 3
            else:
                ref = frames[-1]
                iy = ref.winfo_y() + ref.winfo_height() + 1
            ind.place(in_=self._inner, x=2, y=iy, relwidth=1, width=-4, height=3)
            ind.lift()

        def on_release(e: tk.Event) -> None:
            if not state.get("active"):
                return
            state["active"] = False
            state["ind"].place_forget()
            state["ind"].destroy()

            rd = state["rd"]
            pos = state.get("pos", len(self._rows))
            cur = self._rows.index(rd)

            if pos != cur and pos != cur + 1:
                self._rows.pop(cur)
                if pos > cur:
                    pos -= 1
                self._rows.insert(pos, rd)

                for r in self._rows:
                    r["frame"].pack_forget()
                for r in self._rows:
                    r["frame"].pack(fill=tk.X, pady=4, padx=2)

                self._inner.update_idletasks()
                self._canvas.configure(scrollregion=self._canvas.bbox("all"))

        handle.bind("<ButtonPress-1>", on_press)
        handle.bind("<B1-Motion>", on_motion)
        handle.bind("<ButtonRelease-1>", on_release)

    def _remove_row(self, frame: tk.Frame, row_data: Dict[str, Any]) -> None:
        frame.destroy()
        self._rows.remove(row_data)

    def _collect_configs(self) -> List[ViewGroupConfig]:
        cfgs: List[ViewGroupConfig] = []
        for i, rd in enumerate(self._rows):
            name = rd["name_var"].get().strip() or f"Group {i + 1}"
            pats = [pr["var"].get() for pr in rd["pat_rows"]]
            ops = [pr["op_var"].get() for pr in rd["pat_rows"][1:]]
            color = rd["color"]
            cfgs.append(ViewGroupConfig(name=name, patterns=pats, operators=ops, color=color))
        return cfgs

    def _on_apply(self) -> None:
        self.unbind_all("<MouseWheel>")
        cfgs = self._collect_configs()
        layout = self._layout_var.get()
        self.controller.apply_view_groups(cfgs, layout)
        self.destroy()

    def _on_cancel(self) -> None:
        self.unbind_all("<MouseWheel>")
        self.destroy()
