import tkinter as tk
from tkinter import ttk
from typing import Callable

from project.visualizers.space.constants import W
from project.visualizers.space.style import Toggle, vsep


class PlayerBar(tk.Frame):
    H = 38

    def __init__(self, parent: tk.Frame, controller) -> None:
        super().__init__(parent, bg=W["panel"], height=self.H)
        self.controller = controller

        self.font = "Comic Sans MS"
        self.pack_propagate(False)
        self._build()

    def _build(self) -> None:
        top = tk.Frame(self, bg=W["panel"])
        top.pack(fill=tk.X, padx=4, pady=(3, 0))
        self.progress = ttk.Scale(top, from_=0, to=100, orient=tk.HORIZONTAL, command=self.controller.on_nav_slider)
        self.progress.pack(fill=tk.X, expand=True)

        # Lock the arrows on the slider—otherwise, they'll trigger twice
        # (once via root.bind, and once via the standard Scale handler)
        for key in ("<Left>", "<Right>", "<Up>", "<Down>"):
            self.progress.bind(key, lambda e: "break")

        bottom_frame = tk.Frame(self, bg=W["panel"])
        bottom_frame.pack(fill=tk.X, padx=4)

        def label_generator(txt: str, callback: Callable, width: int = 22) -> tk.Label:
            label = tk.Label(bottom_frame, text=txt, bg=W["surf2"], fg=W["text"], font=(self.font, 9), cursor="hand2",
                             width=width // 8, padx=3, pady=0)
            label.pack(side=tk.LEFT, padx=1)
            label.bind("<Button-1>", lambda e: callback())
            label.bind("<Enter>", lambda e, b = label: b.config(bg=W["border"]))
            label.bind("<Leave>", lambda e, b = label: b.config(bg=W["surf2"]))
            return label

        label_generator("⏮", self.controller.first_image)
        label_generator("⏪", lambda: self.controller.previous_image(10))
        label_generator("←", self.controller.previous_image)

        self.play_btn = tk.Label(bottom_frame, text="▶", bg=W["accent"], fg="white", font=(self.font, 9, "bold"), padx=8,
                                 pady=0, cursor="hand2")
        self.play_btn.pack(side=tk.LEFT, padx=3)
        self.play_btn.bind("<Button-1>", lambda e: self.controller.toggle_play())
        self.play_btn.bind("<Enter>", lambda e: self.play_btn.config(bg=W["acc_h"]))
        self.play_btn.bind("<Leave>", lambda e: self.play_btn.config(bg=W["accent"]))

        label_generator("→", self.controller.next_image)
        label_generator("⏩", lambda: self.controller.next_image(10))
        label_generator("⏭", self.controller.last_image)

        vsep(bottom_frame, W["border"]).pack(side=tk.LEFT, fill=tk.Y, pady=2, padx=4)
        self._direction_buttons = {}
        for txt, val in [("▶▶", "forward"), ("◀◀", "backward")]:
            lb = tk.Label(bottom_frame, text=txt, bg=W["panel"], fg=W["dim"], font=(self.font, 8), cursor="hand2", padx=4)
            lb.pack(side=tk.LEFT)
            lb.bind("<Button-1>", lambda e, v = val: self._set_direction(v))
            self._direction_buttons[val] = lb
        self._set_direction("forward")

        vsep(bottom_frame, W["border"]).pack(side=tk.LEFT, fill=tk.Y, pady=2, padx=4)
        tk.Label(bottom_frame, text="FPS:", bg=W["panel"], fg=W["dim"], font=(self.font, 7)).pack(side=tk.LEFT)
        self.fps_slider = ttk.Scale(bottom_frame, from_=1, to=6, orient=tk.HORIZONTAL,
                                    command=self.controller.on_fps_move, length=70)
        self.fps_slider.set(4)
        self.fps_slider.pack(side=tk.LEFT, padx=(2, 0))
        self.fps_slider.bind("<ButtonRelease-1>", self.controller.on_fps_release)

        self.fps_label = tk.Label(bottom_frame, text="30", bg=W["panel"], fg=W["accent"], font=(self.font, 8, "bold"), width=4)
        self.fps_label.pack(side=tk.LEFT, padx=(2, 4))

        Toggle(bottom_frame, self.controller.loop_var, bg=W["panel"]).pack(side=tk.LEFT)
        tk.Label(bottom_frame, text="loop", bg=W["panel"], fg=W["dim"], font=(self.font, 7)).pack(side=tk.LEFT, padx=(2, 8))
        vsep(bottom_frame, W["border"]).pack(side=tk.LEFT, fill=tk.Y, pady=2, padx=4)

        self.play_status = tk.Label(bottom_frame, text="⏸", bg=W["panel"], fg=W["dim"], font=(self.font, 9, "bold"))
        self.play_status.pack(side=tk.RIGHT, padx=4)

        self.frame_label = tk.Label(bottom_frame, text="0 / 0", bg=W["panel"], fg=W["text"], font=(self.font, 8, "bold"))
        self.frame_label.pack(side=tk.RIGHT, padx=8)

        self.id_label = tk.Label(bottom_frame, text="ID:—  CAM:—", bg=W["panel"], fg=W["dim"], font=(self.font, 8))
        self.id_label.pack(side=tk.RIGHT, padx=4)

    def _set_direction(self, val: str) -> None:
        self.controller.play_direction_var.set(val)
        for v, label in self._direction_buttons.items():
            label.config(fg=W["accent"] if v == val else W["dim"])
