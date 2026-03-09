import tkinter as tk
from tkinter import ttk
from typing import Callable

from project.visualizer.constants import W


def apply_style(root: tk.Tk) -> None:
    style = ttk.Style(root)
    style.theme_use("clam")

    font = "Comic Sans MS"
    background, fg = W["panel"], W["text"]
    surface, border = W["surface"], W["border"]

    style.configure(
        ".",
        background=background,
        foreground=fg,
        fieldbackground=surface,
        bordercolor=border,
        darkcolor=surface,
        lightcolor=surface,
        troughcolor=W["surf2"],
        insertcolor=fg,
        selectbackground=W["accent"],
        selectforeground="#fff",
        font=(font, 9),
    )

    style.configure("TFrame", background=background)
    style.configure("W.TFrame", background=surface)
    style.configure("P.TFrame", background=W["panel"])
    style.configure("T.TFrame", background=W["toolbar"])

    style.configure("TLabel", background=background, foreground=fg)
    style.configure("W.TLabel", background=surface, foreground=fg)
    style.configure("T.TLabel", background=W["toolbar"], foreground=fg)

    style.configure("Dim.TLabel", background=background, foreground=W["dim"], font=(font, 8))
    style.configure("Acc.TLabel", background=background, foreground=W["accent"], font=(font, 9, "bold"))
    style.configure("Grn.TLabel", background=background, foreground=W["green"], font=(font, 8))
    style.configure("Red.TLabel", background=background, foreground=W["red"], font=(font, 8, "bold"))

    style.configure("TButton",
        background=W["surf2"],
        foreground=fg,
        bordercolor=border,
        padding=(6, 3),
        relief="flat",
        focusthickness=0,
    )

    style.map(
        "TButton",
        background=[("active", W["border"]), ("pressed", W["accent"])],
        foreground=[("active", fg)],
    )

    style.configure(
        "Acc.TButton",
        background=W["accent"],
        foreground="#fff",
        bordercolor=W["accent"],
        padding=(8, 4),
    )

    style.map(
        "Acc.TButton",
        background=[("active", W["acc_h"]), ("pressed", W["acc_h"])],
    )

    style.configure("TCheckbutton", background=background, foreground=fg)

    style.map(
        "TCheckbutton",
        background=[("active", background)],
        indicatorcolor=[("selected", W["accent"]), ("!selected", W["surf2"])],
    )

    style.configure(
        "TEntry",
        fieldbackground=surface,
        foreground=fg,
        bordercolor=border,
        insertcolor=fg,
        padding=3,
    )

    style.configure(
        "TSpinbox",
        fieldbackground=surface,
        foreground=fg,
        bordercolor=border,
        arrowcolor=W["dim"],
        background=surface,
    )

    style.configure(
        "Horizontal.TScale",
        background=background,
        troughcolor=W["border"],
        slidercolor=W["accent"],
        sliderlength=12,
        bordercolor=border,
    )

    style.map(
        "Horizontal.TScale",
        slidercolor=[("active", W["acc_h"])],
    )

    style.configure(
        "Horizontal.TProgressbar",
        background=W["accent"],
        troughcolor=W["surf2"],
        bordercolor=border,
        lightcolor=W["accent"],
        darkcolor=W["accent"],
    )

    style.configure("TNotebook", background=background, bordercolor=border, tabmargins=0)

    style.configure(
        "TNotebook.Tab",
        background=W["surf2"],
        foreground=W["dim"],
        padding=(12, 4),
        bordercolor=border,
    )

    style.map(
        "TNotebook.Tab",
        background=[("selected", surface), ("active", W["tab_h"])],
        foreground=[("selected", fg), ("active", fg)],
    )

    style.configure("TPanedwindow", background=W["border"])
    style.configure("Sash", sashthickness=5, sashpad=0, background=W["border"], relief="flat")

    style.configure(
        "TScrollbar",
        background=W["surf2"],
        troughcolor=background,
        arrowcolor=W["dim"],
        bordercolor=background,
        relief="flat",
    )

    style.map("TScrollbar", background=[("active", W["border"])])

    style.configure(
        "Treeview",
        background=surface,
        foreground=fg,
        fieldbackground=surface,
        rowheight=20,
        bordercolor=border,
        font=(font, 8),
    )

    style.configure(
        "Treeview.Heading",
        background=W["surf2"],
        foreground=W["dim"],
        font=(font, 8, "bold"),
        relief="flat",
        padding=(4, 3),
    )

    style.map(
        "Treeview",
        background=[("selected", W["accent"])],
        foreground=[("selected", "#fff")],
    )


class Toggle(tk.Canvas):

    W_SIZE = 32
    H_SIZE = 16

    def __init__(
        self,
        parent: tk.Widget,
        variable: tk.BooleanVar,
        command: Callable[[], None] | None = None,
        bg: str | None = None,
        **kw,
    ) -> None:
        bg = bg or W["panel"]
        super().__init__(
            parent,
            width=self.W_SIZE,
            height=self.H_SIZE,
            bg=bg,
            highlightthickness=0,
            cursor="hand2",
            **kw,
        )

        self._v = variable
        self._cmd = command
        self.bind("<Button-1>", self._click)
        self.bind("<Destroy>", self._on_destroy)
        self._trace_id = variable.trace_add("write", lambda *_: self._draw())
        self._draw()

    def _on_destroy(self, _event: tk.Event) -> None:
        try:
            self._v.trace_remove("write", self._trace_id)
        except (Exception, ):
            pass

    def _click(self, _event: tk.Event) -> None:
        self._v.set(not self._v.get())
        if self._cmd:
            self._cmd()

    def _draw(self) -> None:
        try:
            self.delete("all")
        except (Exception, ):
            return

        on = self._v.get()
        fill = W["accent"] if on else W["border"]
        r = self.H_SIZE // 2 - 1
        self.create_arc(1, 1, self.H_SIZE - 1, self.H_SIZE - 1, start=90, extent=180, fill=fill, outline="")
        self.create_arc(self.W_SIZE - self.H_SIZE + 1, 1, self.W_SIZE - 1, self.H_SIZE - 1, start=270,
                        extent=180, fill=fill, outline="")

        self.create_rectangle(self.H_SIZE // 2, 1, self.W_SIZE - self.H_SIZE // 2, self.H_SIZE - 1, fill=fill,
                              outline="")
        cx = self.W_SIZE - r - 2 if on else r + 2
        self.create_oval(cx - r + 1, 2, cx + r - 1, self.H_SIZE - 2, fill="white", outline="")


class SliderRow(tk.Frame):
    def __init__(
        self,
        parent: tk.Widget,
        label: str,
        var: tk.DoubleVar,
        from_: float,
        to: float,
        fmt: str = "{:.1f}",
        command: Callable[[], None] | None = None,
        bg: str | None = None,
    ) -> None:

        bg = bg or W["panel"]
        self.font = "Comic Sans MS"

        super().__init__(parent, bg=bg)

        self._v = var
        self._fmt = fmt
        self._cmd = command

        row = tk.Frame(self, bg=bg)
        row.pack(fill=tk.X)

        tk.Label(row, text=label, bg=bg, fg=W["dim"], font=(self.font, 8)).pack(side=tk.LEFT)
        self._lbl = tk.Label(row, text=fmt.format(var.get()), bg=bg, fg=W["accent"], font=(self.font, 8, "bold"),
                             width=5)
        self._lbl.pack(side=tk.RIGHT)
        ttk.Scale(self, from_=from_, to=to, variable=var, orient=tk.HORIZONTAL, command=self._on).pack(fill=tk.X)

    def _on(self, _=None) -> None:
        self._lbl.config(text=self._fmt.format(self._v.get()))
        if self._cmd:
            self._cmd()


class TogRow(tk.Frame):
    def __init__(
        self,
        parent: tk.Frame,
        label: str,
        var: tk.BooleanVar,
        command: Callable[[], None] | None = None,
        bg: str | None = None,
    ) -> None:
        bg = bg or W["panel"]
        self.font = "Comic Sans MS"
        super().__init__(parent, bg=bg)
        tk.Label(self, text=label, bg=bg, fg=W["text"], font=(self.font, 9),
                 anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)
        Toggle(self, var, command=command, bg=bg).pack(side=tk.RIGHT)


def hsep(parent: tk.Widget, bg: str | None = None) -> tk.Frame:
    return tk.Frame(parent, height=1, bg=bg or W["sep"])


def vsep(parent: tk.Widget, bg: str | None = None) -> tk.Frame:
    return tk.Frame(parent, width=1, bg=bg or W["sep"])
