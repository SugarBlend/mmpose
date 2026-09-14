from typing import Tuple


class ZoomState:
    MIN_ZOOM = 1.0
    MAX_ZOOM = 20.0
    ZOOM_STEP = 1.2

    def __init__(self) -> None:
        self.zoom: float = 1.0
        self.cx: float = 0.5
        self.cy: float = 0.5

        # Where the (possibly cropped) image was last placed on its canvas,
        # in canvas pixels. Kept up to date by resize_for_canvas() every
        # render so wheel/drag handlers can map mouse position -> image
        # fraction.
        self.disp_x: int = 0
        self.disp_y: int = 0
        self.disp_w: int = 1
        self.disp_h: int = 1

    @property
    def is_default(self) -> bool:
        return self.zoom <= self.MIN_ZOOM + 1e-6

    def set_display_rect(self, x: int, y: int, w: int, h: int) -> None:
        self.disp_x, self.disp_y = x, y
        self.disp_w, self.disp_h = max(1, w), max(1, h)

    def _half(self, zoom: float) -> float:
        return 0.5 / zoom

    def canvas_to_frac(self, mx: float, my: float) -> Tuple[float, float]:
        u = (mx - self.disp_x) / self.disp_w
        v = (my - self.disp_y) / self.disp_h
        u = min(max(u, 0.0), 1.0)
        v = min(max(v, 0.0), 1.0)
        hw, hh = self._half(self.zoom), self._half(self.zoom)
        nx = (self.cx - hw) + u * 2 * hw
        ny = (self.cy - hh) + v * 2 * hh
        return nx, ny

    def zoom_at(self, mx: float, my: float, factor: float) -> bool:
        nx, ny = self.canvas_to_frac(mx, my)
        old_hw, old_hh = self._half(self.zoom), self._half(self.zoom)
        u = 0.5 if old_hw <= 0 else (nx - (self.cx - old_hw)) / (2 * old_hw)
        v = 0.5 if old_hh <= 0 else (ny - (self.cy - old_hh)) / (2 * old_hh)

        new_zoom = min(max(self.zoom * factor, self.MIN_ZOOM), self.MAX_ZOOM)
        if abs(new_zoom - self.zoom) < 1e-9:
            return False

        new_hw, new_hh = self._half(new_zoom), self._half(new_zoom)
        new_cx = nx - (u - 0.5) * 2 * new_hw
        new_cy = ny - (v - 0.5) * 2 * new_hh

        self.zoom = new_zoom
        self.cx = 0.5 if new_hw >= 0.5 else min(max(new_cx, new_hw), 1 - new_hw)
        self.cy = 0.5 if new_hh >= 0.5 else min(max(new_cy, new_hh), 1 - new_hh)
        return True

    def pan(self, dx_px: float, dy_px: float) -> bool:
        if self.is_default:
            return False
        hw, hh = self._half(self.zoom), self._half(self.zoom)
        new_cx = self.cx - dx_px / self.disp_w * 2 * hw
        new_cy = self.cy - dy_px / self.disp_h * 2 * hh
        self.cx = 0.5 if hw >= 0.5 else min(max(new_cx, hw), 1 - hw)
        self.cy = 0.5 if hh >= 0.5 else min(max(new_cy, hh), 1 - hh)
        return True

    def reset(self) -> None:
        self.zoom = 1.0
        self.cx = 0.5
        self.cy = 0.5

    def crop_rect(self, iw: int, ih: int) -> Tuple[int, int, int, int]:
        hw, hh = self._half(self.zoom), self._half(self.zoom)
        x0 = int(round((self.cx - hw) * iw))
        x1 = int(round((self.cx + hw) * iw))
        y0 = int(round((self.cy - hh) * ih))
        y1 = int(round((self.cy + hh) * ih))
        x0 = max(0, min(x0, iw - 1))
        y0 = max(0, min(y0, ih - 1))
        x1 = max(x0 + 1, min(x1, iw))
        y1 = max(y0 + 1, min(y1, ih))
        return x0, y0, x1, y1


def bind_zoom_pan(canvas, state: ZoomState, on_change) -> None:
    pan = {"active": False, "lx": 0, "ly": 0}

    def _ctrl_held(event) -> bool:
        return bool(event.state & 0x0004)

    def _on_wheel(event) -> None:
        # Windows / macOS style: <MouseWheel> with event.delta, gated on Ctrl.
        if not _ctrl_held(event):
            return
        factor = ZoomState.ZOOM_STEP if event.delta > 0 else (1.0 / ZoomState.ZOOM_STEP)
        if state.zoom_at(event.x, event.y, factor):
            on_change()

    def _on_wheel_up(event) -> None:
        # X11 style: Button-4 == wheel up.
        if state.zoom_at(event.x, event.y, ZoomState.ZOOM_STEP):
            on_change()

    def _on_wheel_down(event) -> None:
        # X11 style: Button-5 == wheel down.
        if state.zoom_at(event.x, event.y, 1.0 / ZoomState.ZOOM_STEP):
            on_change()

    def _on_mid_press(event) -> None:
        pan["active"] = True
        pan["lx"], pan["ly"] = event.x, event.y

    def _on_mid_motion(event) -> None:
        if not pan["active"]:
            return
        dx, dy = event.x - pan["lx"], event.y - pan["ly"]
        pan["lx"], pan["ly"] = event.x, event.y
        if (dx or dy) and state.pan(dx, dy):
            on_change()

    def _on_mid_release(_event) -> None:
        pan["active"] = False

    canvas.bind("<MouseWheel>", _on_wheel, add="+")
    canvas.bind("<Control-Button-4>", _on_wheel_up, add="+")
    canvas.bind("<Control-Button-5>", _on_wheel_down, add="+")
    canvas.bind("<Button-2>", _on_mid_press, add="+")
    canvas.bind("<B2-Motion>", _on_mid_motion, add="+")
    canvas.bind("<ButtonRelease-2>", _on_mid_release, add="+")
