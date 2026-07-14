import tkinter as tk
from typing import Optional
from PIL import ImageTk
from PIL import Image as PILImage, ImageDraw as PILDraw
from project.visualizers.surface.preprocessing import RenderSettings


class LegendPanel(tk.Toplevel):

    _BG = "#C9C9C9"
    _FG = "#080808"
    _CARD_BG = "#B8B4B4"
    _SEP = "#9A9696"
    _ACCENT = "#545252"

    def __init__(self, parent: tk.Tk, rs: RenderSettings) -> None:
        super().__init__(parent)
        self._rs = rs
        self._parent = parent
        self._dot_photos: list = []

        self.title("")
        self.configure(bg=self._BG)
        self.resizable(False, False)
        self.overrideredirect(True)  # frameless — looks attached
        self.withdraw()
        self.transient(parent)

        self._frame: Optional[tk.Frame] = None

        # close when main window gets focus
        parent.bind("<FocusIn>", self._on_parent_focus, add="+")
        self.bind("<FocusOut>",  self._on_self_focus_out, add="+")

        self._side_colors_list: list[dict[str, tuple[float, float, float]]] = []
        self._joint_colors: list[tuple[float, float, float]] = []
        self._legends: list[str] = []

    def populate(
        self,
        legends: list[str],
        joint_colors: list[tuple[float, float, float]],
        side_colors_list: list[dict[str, tuple[float, float, float]]],
    ) -> None:
        self._legends = legends
        self._joint_colors = joint_colors
        self._side_colors_list = side_colors_list

    def close(self) -> None:
        if self.winfo_viewable():
            self.withdraw()

    def toggle(self, x: int, y: int) -> None:
        if self.winfo_viewable():
            self.withdraw()
        else:
            self._rebuild()
            self.geometry(f"+{x}+{y}")
            self.deiconify()
            self.lift()

    def _make_dot(self, color_rgb_f: tuple[float, float, float], size: int = 14) -> tk.PhotoImage:
        img = PILImage.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = PILDraw.Draw(img)
        r, g, b = (int(c * 255) for c in color_rgb_f)
        draw.ellipse([0, 0, size - 1, size - 1], fill=(180, 176, 176, 220))
        draw.ellipse([2, 2, size - 3, size - 3], fill=(r, g, b, 255))
        hs = size // 4
        draw.ellipse([3, 3, 3 + hs, 3 + hs], fill=(255, 255, 255, 140))
        return ImageTk.PhotoImage(img)

    def _rebuild(self) -> None:
        if self._frame is not None:
            self._frame.destroy()
        self._dot_photos.clear()

        if hasattr(self, "_border_frame"):
            self._border_frame.destroy()

        border = tk.Frame(self, bg=self._ACCENT, bd=0)
        border.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        self._border_frame = border

        outer = tk.Frame(border, bg=self._BG)
        outer.pack(fill=tk.BOTH, expand=True)
        self._frame = outer

        use_side = self._rs.use_side_color

        card = tk.Frame(outer, bg=self._CARD_BG, padx=12, pady=8)
        card.pack(fill=tk.X)

        stripe = tk.Frame(card, bg=self._ACCENT, width=3)
        stripe.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10), pady=(0, 6))

        rows = tk.Frame(card, bg=self._CARD_BG)
        rows.pack(side=tk.LEFT, fill=tk.X, expand=True)

        for i, legend in enumerate(self._legends):
            row = tk.Frame(rows, bg=self._CARD_BG)
            row.pack(fill=tk.X, pady=4)

            if use_side and self._side_colors_list:
                sc = self._side_colors_list[i]
                for side_key in ("left", "center", "right"):
                    col = sc.get(side_key, self._joint_colors[i])
                    ph = self._make_dot(col, size=12)
                    self._dot_photos.append(ph)
                    tk.Label(row, image=ph, bg=self._CARD_BG).pack(side=tk.LEFT, padx=1)
            else:
                col = self._joint_colors[i]
                ph = self._make_dot(col, size=14)
                self._dot_photos.append(ph)
                tk.Label(row, image=ph, bg=self._CARD_BG).pack(side=tk.LEFT, padx=(0, 6))

            tk.Label(row, text=legend, bg=self._CARD_BG, fg=self._FG, font=("SegoeUI", 11, "bold")).pack(side=tk.LEFT)

    def _on_parent_focus(self, event: tk.Event) -> None:
        if event.widget is self._parent:
            self.close()

    def _on_self_focus_out(self, event: tk.Event) -> None:
        if event.widget is self:
            self.after(50, self._check_still_focused)

    def _check_still_focused(self) -> None:
        try:
            focused = self.focus_get()
        except (Exception,):
            focused = None

        if focused is None or str(focused) == str(self._parent):
            self.close()
