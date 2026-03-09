import tkinter as tk
from tkinter import ttk
from typing import Any
from project.visualizer.constants import HALPE26_KEYPOINT_NAMES, KPT_BGR, W
from project.visualizer.models import ImageEntry


class InfoPanel(tk.Frame):
    def __init__(self, parent: tk.Frame, controller) -> None:
        super().__init__(parent, bg=W["panel"])
        self.controller = controller

        self.font = "Consolas"
        self._anns: list[tuple[str, tuple[int, int, int], dict[str, Any], int]] = []

        # Permanent selection: group name + annotation index within that group
        self._pinned_group: str = ""
        self._pinned_ann_idx: int = 0 # position within entry.annotations of this group
        self._build_ui()

    def _build_ui(self) -> None:
        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True)

        ann_frame = tk.Frame(notebook, bg=W["surface"])
        notebook.add(ann_frame, text="Annotations")
        self._build_ann(ann_frame)

        joints_frame = tk.Frame(notebook, bg=W["surface"])
        notebook.add(joints_frame, text="Joints")
        self._build_legend(joints_frame)

    def _build_ann(self, parent: tk.Frame) -> None:
        background = W["surface"]
        pw = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        pw.pack(fill=tk.BOTH, expand=True)

        # separate paned window on two sectors
        left_sector = tk.Frame(pw, bg=background)
        pw.add(left_sector, weight=1)

        label_header = tk.Frame(left_sector, bg=W["surf2"])
        label_header.pack(fill=tk.X)
        self._ann_hdr_lbl = tk.Label(label_header, text="Frame annotations", bg=W["surf2"], fg=W["dim"],
                                     font=(self.font, 7, "bold"), padx=6, pady=3)
        self._ann_hdr_lbl.pack(side=tk.LEFT)

        tree_frame = tk.Frame(left_sector, bg=background)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        self.tree = ttk.Treeview(
            tree_frame,
            columns=("idx", "grp", "cat", "nk"),
            show="headings", selectmode="browse"
        )
        self.tree.heading("idx", text="#")
        self.tree.heading("grp", text="Group")
        self.tree.heading("cat", text="Cat")
        self.tree.heading("nk", text="KP")
        self.tree.column("idx", width=25, minwidth=25, stretch=True, anchor="center")
        self.tree.column("grp", width=50, minwidth=45, stretch=True, anchor="center")
        self.tree.column("cat", width=25, minwidth=30, stretch=True, anchor="center")
        self.tree.column("nk", width=25, minwidth=30, stretch=True, anchor="center")

        sb = ttk.Scrollbar(tree_frame, command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        right_sector = tk.Frame(pw, bg=background)
        pw.add(right_sector, weight=5)
        rdr_hdr = tk.Frame(right_sector, bg=W["surf2"])
        rdr_hdr.pack(fill=tk.X)
        self._detail_hdr = tk.Label(rdr_hdr, text="Choose annotation", bg=W["surf2"], fg=W["dim"],
                                    font=(self.font, 7, "bold"), padx=6, pady=3)
        self._detail_hdr.pack(side=tk.LEFT)

        txt_frame = tk.Frame(right_sector, bg=background)
        txt_frame.pack(fill=tk.BOTH, expand=True)
        self._detail_txt = tk.Text(
            txt_frame, bg=background, fg=W["text"], font=(self.font, 8), wrap=tk.NONE, relief="flat", borderwidth=0,
            state="disabled", cursor="arrow", selectbackground=W["surf2"]
        )
        sb_x = ttk.Scrollbar(txt_frame, orient=tk.HORIZONTAL, command=self._detail_txt.xview)
        sb_y = ttk.Scrollbar(txt_frame, orient=tk.VERTICAL, command=self._detail_txt.yview)
        self._detail_txt.configure(xscrollcommand=sb_x.set, yscrollcommand=sb_y.set)
        sb_y.pack(side=tk.RIGHT, fill=tk.Y)
        sb_x.pack(side=tk.BOTTOM, fill=tk.X)
        self._detail_txt.pack(fill=tk.BOTH, expand=True)
        self._detail_txt.tag_configure("key", foreground=W["accent"], font=(self.font, 8, "bold"))
        self._detail_txt.tag_configure("val", foreground=W["text"], font=(self.font, 8))
        self._detail_txt.tag_configure("dim", foreground=W["dim"], font=(self.font, 8))
        self._detail_txt.tag_configure("grp", foreground=W["accent"], font=(self.font, 8, "bold"))

    def _on_select(self, _: tk.Event) -> None:
        selected_items = self.tree.selection()
        if not selected_items:
            return

        flat_idx = self.tree.item(selected_items[0])["values"][0]
        if 0 <= flat_idx < len(self._anns):
            grp_name, _, ann, ann_idx = self._anns[flat_idx]
            # save the group and the position of the annotation within it - stable between frames
            self._pinned_group = grp_name
            self._pinned_ann_idx = ann_idx
            self._show_detail(ann, flat_idx, grp_name)

    def _show_detail(self, ann: dict[str, Any], flat_idx: int, group_name: str = "") -> None:
        label = f"Annotation #{flat_idx}"
        if group_name:
            label += f"  [{group_name}]"
        self._detail_hdr.config(text=label)
        txt = self._detail_txt
        txt.config(state="normal")
        txt.delete("1.0", "end")
        txt.insert("end", "{\n", "dim")
        for key, val in ann.items():
            if key == "file_name":
                continue

            txt.insert("end", f"  {key}", "key")
            txt.insert("end", ": ", "dim")
            if key == "keypoints" and isinstance(val, list) and len(val) % 3 == 0:
                txt.insert("end", "[\n", "dim")
                nk = len(val) // 3
                for i in range(nk):
                    x, y, v = val[i * 3], val[i * 3 + 1], val[i * 3 + 2]
                    nm = HALPE26_KEYPOINT_NAMES[i] if i < len(HALPE26_KEYPOINT_NAMES) else str(i)
                    txt.insert("end", f"    # {i:>2} {nm:<16} ", "dim")
                    txt.insert("end", f"x={x:.1f}  y={y:.1f}  v={v}\n", "val")
                txt.insert("end", "  ]\n", "dim")
            elif isinstance(val, list):
                items = ", ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in val)
                txt.insert("end", f"[{items}]\n", "val")
            elif isinstance(val, float):
                txt.insert("end", f"{val:.4f}\n", "val")
            else:
                txt.insert("end", f"{val}\n", "val")
        txt.insert("end", "}", "dim")
        txt.config(state="disabled")

    def update_ann(self, entry: ImageEntry) -> None:
        self._ann_hdr_lbl.config(text="Frame annotations")
        self._anns = [("", W["dim"], ann, i) for i, ann in enumerate(entry.annotations)]
        self._rebuild_tree()

    def update_groups(self, groups: list) -> None:
        n = len(groups)
        self._ann_hdr_lbl.config(text=f"Annotations  ({n} group{'s' if n != 1 else ''})")
        self._anns: list[tuple] = []
        for grp in groups:
            entry = grp.current_entry()
            if entry is None:
                continue

            for i, ann in enumerate(entry.annotations):
                self._anns.append((grp.cfg.name, grp.cfg.color, ann, i))
        self._rebuild_tree()

    def _rebuild_tree(self) -> None:
        self.tree.delete(*self.tree.get_children())
        image_ids = []
        for flat_idx, (grp_name, grp_color, ann, ann_idx) in enumerate(self._anns):
            nk = len(ann.get("keypoints", [])) // 3
            cat = ann.get("category_id", "?")
            image_id = self.tree.insert("", "end", values=(flat_idx, grp_name or "—", cat, nk))
            image_ids.append(image_id)

        if not image_ids:
            self._detail_hdr.config(text="No annotations")
            self._detail_txt.config(state="normal")
            self._detail_txt.delete("1.0", "end")
            self._detail_txt.config(state="disabled")
            return

        # restore the selection: search for the same group and the same index within it.
        # if there is no exact match, take the closest available index in the same group.
        restore_idx = None
        fallback_idx = None

        for fi, (grp_name, _, ann, ann_idx) in enumerate(self._anns):
            if grp_name == self._pinned_group:
                if fallback_idx is None:
                    fallback_idx = fi
                if ann_idx == self._pinned_ann_idx:
                    restore_idx = fi
                    break

        if restore_idx is None:
            restore_idx = fallback_idx if fallback_idx is not None else 0

        restore_idx = min(restore_idx, len(image_ids) - 1)
        self.tree.selection_set(image_ids[restore_idx])
        self.tree.see(image_ids[restore_idx])
        grp_name, _, ann, _ = self._anns[restore_idx]
        self._show_detail(ann, restore_idx, grp_name)

    def _build_legend(self, parent: tk.Frame) -> None:
        canvas = tk.Canvas(parent, bg=W["surface"], highlightthickness=0)
        sb = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        inn = tk.Frame(canvas, bg=W["surface"])
        canvas.create_window((0, 0), window=inn, anchor="nw")
        inn.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1 * e.delta / 120),"units"))
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT,fill=tk.Y)

        grid = tk.Frame(inn, bg=W["surface"])
        grid.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        for i, name in enumerate(HALPE26_KEYPOINT_NAMES):
            bgr = KPT_BGR[i] if i < len(KPT_BGR) else (150, 150, 150)
            hx = "#%02x%02x%02x" % (bgr[2], bgr[1], bgr[0])
            row, col = divmod(i, 2)
            cell = tk.Frame(grid, bg=W["surface"])
            cell.grid(row=row, column=col, sticky="w", padx=3, pady=1)
            sw = tk.Canvas(cell, width=11, height=11, bg=W["surface"], highlightthickness=0)
            sw.pack(side=tk.LEFT, padx=(0, 4))
            sw.create_oval(1, 1, 10, 10, fill=hx, outline="")
            tk.Label(cell, text=f"{i:>2}:", bg=W["surface"], fg=W["dim"], font=(self.font, 8), width=3).pack(side=tk.LEFT)
            tk.Label(cell, text=name, bg=W["surface"], fg=W["text"], font=(self.font, 8), width=16, anchor="w").pack(side=tk.LEFT)
