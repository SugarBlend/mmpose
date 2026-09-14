import tkinter as tk

from project.visualizers.space.constants import W
from project.visualizers.space.models import DrawParams, ImageEntry
from project.visualizers.space.widgets.view_group import ViewGroup
from project.visualizers.space.zoom import ZoomState, bind_zoom_pan


class GroupPanel(tk.Frame):
    BORDER_W = 2

    def __init__(self, parent: tk.Frame, group: ViewGroup, controller, **kw):
        super().__init__(parent, bg=W["canvas"], highlightthickness=self.BORDER_W,
                         highlightbackground=group.cfg.color, **kw)
        self.group = group
        self.controller = controller

        # Zoom/pan state, own to this panel, persists frame to frame.
        self.zoom = ZoomState()

        self.font = "Comic Sans MS"
        self.canvas = tk.Canvas(self, bg=W["canvas"], highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self._overlay = tk.Label(self.canvas, text="", bg=W["canvas"], fg=group.cfg.color, font=(self.font, 8, "bold"),
                                 anchor="ne", padx=4, pady=2)
        self._overlay.place(relx=1.0, rely=0.0, anchor="ne")

        self.canvas.bind("<Double-Button-1>", self._on_dbl_click)
        self.canvas.bind("<Button-3>", self._on_right_click)
        bind_zoom_pan(self.canvas, self.zoom, self._on_zoom_change)

    def _on_dbl_click(self, _) -> None:
        self.controller.expand_group(self.group)

    def _on_right_click(self, _) -> None:
        self.controller.collapse_group()

    def _on_zoom_change(self) -> None:
        # Redraw just this panel for responsiveness instead of re-laying out
        # and redrawing the whole grid on every wheel tick / drag step.
        self.draw(self.controller._make_dp())

    def draw(self, dp: DrawParams) -> None:
        cw = max(self.canvas.winfo_width(), 100)
        ch = max(self.canvas.winfo_height(), 100)
        self.canvas.delete("all")

        photo = self.group.render(cw, ch, dp, zoom_state=self.zoom)
        if photo is None:
            if not self.group.loader.directory:
                msg = "Folder with images doesn't chose"
            else:
                msg = f"File not found: {(self.group.current_entry() or ImageEntry(0, '?')).file_name}"

            self.canvas.create_text(cw // 2, ch // 2, text=msg, fill="#555", font=(self.font, 11), justify=tk.CENTER)
        else:
            self.canvas.create_image((cw - photo.width()) // 2, (ch - photo.height()) // 2, anchor=tk.NW,
                                     image=photo)

        self._overlay.config(text=f"{self.group._idx + 1}/{self.group.total()}" if self.group.total() else "0/0")
