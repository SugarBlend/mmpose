"""
overlay.py — skeleton rendering + HUD helpers for pose comparison player.

Key additions vs original:
- Side-based joint/limb colouring  (left / right / center), 3 colours per model.
- BodyPartFilter: decides which skeleton edges and joints to draw based on
  enabled body-part flags and the active keypoint format.
- draw_skeleton_pretty now accepts optional joint_r / limb_w overrides so the
  player UI can pass slider values at render time.
"""
from __future__ import annotations

import colorsys
from pathlib import Path
from typing import Literal

import cairo
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ──────────────────────────────────────────────────────────────────────────────
# Colour palettes
# ──────────────────────────────────────────────────────────────────────────────

def _hsl(h: float, s: float, l: float) -> tuple[float, float, float]:
    return colorsys.hls_to_rgb(h, l, s)


# Each entry: (hue 0-1,  joint_sat, joint_lit,  limb_sat, limb_lit)
_SPECS = [
    (0.02,  0.90, 0.55,   0.40, 0.62),   # red-orange
    (0.35,  0.82, 0.48,   0.35, 0.58),   # green
    (0.58,  0.88, 0.58,   0.38, 0.65),   # blue
    (0.75,  0.85, 0.58,   0.36, 0.65),   # violet
    (0.12,  0.88, 0.55,   0.38, 0.62),   # amber
    (0.50,  0.82, 0.52,   0.35, 0.60),   # cyan
]

MODEL_JOINT_COLORS: list[tuple[float, float, float]] = [
    _hsl(h, js, jl) for h, js, jl, _, _ in _SPECS
]
MODEL_LIMB_COLORS: list[tuple[float, float, float]] = [
    _hsl(h, ls, ll) for h, _, _, ls, ll in _SPECS
]


def _to_bgr(c: tuple[float, float, float]) -> tuple[int, int, int]:
    r, g, b = c
    return int(b * 255), int(g * 255), int(r * 255)


MODEL_COLORS = [_to_bgr(c) for c in MODEL_JOINT_COLORS]
MODEL_LIMB_COLORS_BGR = [_to_bgr(c) for c in MODEL_LIMB_COLORS]


# ──────────────────────────────────────────────────────────────────────────────
# Side-based colour triplets  (left / right / center)
# Each model gets a triplet derived from its base hue.
# ──────────────────────────────────────────────────────────────────────────────

# (hue, sat, lit) overrides for the three roles
_SIDE_LEFT_OFFSET   = (+0.00,  0.88, 0.55)   # same hue, vivid
_SIDE_RIGHT_OFFSET  = (+0.50,  0.85, 0.55)   # complementary hue
_SIDE_CENTER_OFFSET = (+0.17,  0.55, 0.65)   # warm neutral


def _side_colors_for_model(base_h: float) -> dict[str, tuple[float, float, float]]:
    dh_l, s_l, l_l = _SIDE_LEFT_OFFSET
    dh_r, s_r, l_r = _SIDE_RIGHT_OFFSET
    dh_c, s_c, l_c = _SIDE_CENTER_OFFSET
    return {
        "left":   _hsl((base_h + dh_l) % 1.0, s_l, l_l),
        "right":  _hsl((base_h + dh_r) % 1.0, s_r, l_r),
        "center": _hsl((base_h + dh_c) % 1.0, s_c, l_c),
    }


# Pre-built for each model slot
MODEL_SIDE_COLORS: list[dict[str, tuple[float, float, float]]] = [
    _side_colors_for_model(h) for h, *_ in _SPECS
]


# ──────────────────────────────────────────────────────────────────────────────
# Keypoint format metadata
# ──────────────────────────────────────────────────────────────────────────────

Side = Literal["left", "right", "center"]

# fmt: off
# Joint index → which body-part category it belongs to, and its side.
# Supported formats: COCO-17, HALPE-26, HALPE-136, COCO-Wholebody-133

# COCO-17: все 17 точек — это стандартный body.
# Нос/глаза/уши входят в body (они часть скелета из 17 точек, не face-маска).
# Запястья — body (не hands), колени/лодыжки — body (не feet).
# hands = только детализированные точки ладоней (пальцы и т.д.)
# feet  = только детализированные точки ступней (пальцы, пятка)
# face  = только face-меш (контур + внутренние точки лица)
_COCO17_JOINT_META: dict[int, tuple[str, Side]] = {
    0:  ("body", "center"),  # nose
    1:  ("body", "left"),    # left eye
    2:  ("body", "right"),   # right eye
    3:  ("body", "left"),    # left ear
    4:  ("body", "right"),   # right ear
    5:  ("body", "left"),    # left shoulder
    6:  ("body", "right"),   # right shoulder
    7:  ("body", "left"),    # left elbow
    8:  ("body", "right"),   # right elbow
    9:  ("body", "left"),    # left wrist
    10: ("body", "right"),   # right wrist
    11: ("body", "left"),    # left hip
    12: ("body", "right"),   # right hip
    13: ("body", "left"),    # left knee
    14: ("body", "right"),   # right knee
    15: ("body", "left"),    # left ankle
    16: ("body", "right"),   # right ankle
}

# HALPE-26: COCO-17 body (0-16) + head/neck/pelvis (body) + toe/heel points (feet).
_HALPE26_EXTRA: dict[int, tuple[str, Side]] = {
    17: ("body", "center"),  # head top
    18: ("body", "center"),  # neck
    19: ("body", "center"),  # pelvis
    20: ("feet", "left"),    # left big toe
    21: ("feet", "left"),    # left small toe
    22: ("feet", "left"),    # left heel
    23: ("feet", "right"),   # right big toe
    24: ("feet", "right"),   # right small toe
    25: ("feet", "right"),   # right heel
}

def _halpe26_meta() -> dict[int, tuple[str, Side]]:
    m = dict(_COCO17_JOINT_META)
    m.update(_HALPE26_EXTRA)
    return m

_HALPE26_JOINT_META = _halpe26_meta()


def _coco_wholebody133_meta() -> dict[int, tuple[str, Side]]:
    """
    COCO-Wholebody 133 (mmpose layout):
      0-16   body keypoints (COCO-17)
      17-22  feet: left foot detailed (big toe, small toe, heel × 2 sides — varies)
             actual mmpose layout: 17-22 = left/right foot details
      23-90  face mesh (68 points: contour + inner landmarks)
      91-111 left hand (21 points: wrist + 4 fingers × 4 + thumb × 4 + tip)
     112-132 right hand (21 points)
    """
    m: dict[int, tuple[str, Side]] = {}

    # Body: 0-16
    for idx, val in _COCO17_JOINT_META.items():
        m[idx] = val

    # Feet detailed: 17-22
    # mmpose wholebody: 17=left_big_toe, 18=left_small_toe, 19=left_heel,
    #                   20=right_big_toe, 21=right_small_toe, 22=right_heel
    feet_extra = {
        17: ("feet", "left"),
        18: ("feet", "left"),
        19: ("feet", "left"),
        20: ("feet", "right"),
        21: ("feet", "right"),
        22: ("feet", "right"),
    }
    m.update(feet_extra)

    # Face mesh: 23-90 (68 points)
    for idx in range(23, 91):
        m[idx] = ("face", "center")

    # Left hand: 91-111 (21 points)
    for idx in range(91, 112):
        m[idx] = ("hands", "left")

    # Right hand: 112-132 (21 points)
    for idx in range(112, 133):
        m[idx] = ("hands", "right")

    return m

_COCO133_JOINT_META = _coco_wholebody133_meta()


def _halpe136_meta() -> dict[int, tuple[str, Side]]:
    """
    HALPE-136: HALPE-26 body (0-25) + left hand (21) + right hand (21) + face mesh (68).
      0-25   body (HALPE-26: COCO-17 + head/neck/pelvis + toe/heel)
     26-46   left hand (21 points)
     47-67   right hand (21 points)
     68-135  face mesh (68 points)
    """
    m: dict[int, tuple[str, Side]] = dict(_HALPE26_JOINT_META)
    for idx in range(26, 47):
        m[idx] = ("hands", "left")
    for idx in range(47, 68):
        m[idx] = ("hands", "right")
    for idx in range(68, 136):
        m[idx] = ("face", "center")
    return m

_HALPE136_JOINT_META = _halpe136_meta()


FORMAT_JOINT_META: dict[int, dict[int, tuple[str, Side]]] = {
    17:  _COCO17_JOINT_META,
    26:  _HALPE26_JOINT_META,
    133: _COCO133_JOINT_META,
    136: _HALPE136_JOINT_META,
}
# fmt: on


# ──────────────────────────────────────────────────────────────────────────────
# BodyPartFilter
# ──────────────────────────────────────────────────────────────────────────────

class BodyPartFilter:
    """
    Decides which joints and skeleton edges are visible based on the
    current enable-flags for body / face / hands / feet.
    """

    PARTS = ("body", "face", "hands", "feet")

    def __init__(self) -> None:
        self._enabled: dict[str, bool] = {p: True for p in self.PARTS}

    def set(self, part: str, value: bool) -> None:
        if part not in self._enabled:
            raise ValueError(f"Unknown part {part!r}. Valid: {self.PARTS}")
        self._enabled[part] = value

    def is_enabled(self, part: str) -> bool:
        return self._enabled.get(part, True)

    def joint_visible(self, joint_idx: int, num_kpts: int) -> bool:
        meta = FORMAT_JOINT_META.get(num_kpts, {})
        info = meta.get(joint_idx)
        if info is None:
            # Unknown format / extra joints → show by default
            return True
        part, _ = info
        return self._enabled.get(part, True)

    def edge_visible(self, a: int, b: int, num_kpts: int) -> bool:
        return self.joint_visible(a, num_kpts) and self.joint_visible(b, num_kpts)

    def joint_side(self, joint_idx: int, num_kpts: int) -> Side:
        meta = FORMAT_JOINT_META.get(num_kpts, {})
        info = meta.get(joint_idx)
        if info is None:
            return "center"
        _, side = info
        return side

    def edge_side(self, a: int, b: int, num_kpts: int) -> Side:
        sa = self.joint_side(a, num_kpts)
        sb = self.joint_side(b, num_kpts)
        if sa == sb:
            return sa
        if "center" in (sa, sb):
            # one end is center → use the non-center side
            return sb if sa == "center" else sa
        # crosses sides → center colour
        return "center"


# Shared singleton — the player writes to this, draw_skeleton_pretty reads it
body_filter = BodyPartFilter()


# ──────────────────────────────────────────────────────────────────────────────
# Font helpers
# ──────────────────────────────────────────────────────────────────────────────

def _find_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _find_font_bold(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return _find_font(size)


# ──────────────────────────────────────────────────────────────────────────────
# Cairo helpers
# ──────────────────────────────────────────────────────────────────────────────

def _cairo_surface_from_bgr(img: np.ndarray) -> tuple[cairo.ImageSurface, np.ndarray]:
    h, w = img.shape[:2]
    bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    buf = bgra.flatten()
    surface = cairo.ImageSurface.create_for_data(buf, cairo.FORMAT_ARGB32, w, h, w * 4)
    return surface, buf


def _bgra_to_bgr(buf: np.ndarray, h: int, w: int) -> np.ndarray:
    bgra = buf.reshape(h, w, 4)
    return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)


# ──────────────────────────────────────────────────────────────────────────────
# Core drawing
# ──────────────────────────────────────────────────────────────────────────────

def draw_skeleton_pretty(
    img: np.ndarray,
    keypoints: np.ndarray,
    skeleton: list[tuple[int, int]],
    # Legacy single-colour path still accepted; side_colors takes priority if given
    joint_color: tuple[float, float, float],
    limb_color:  tuple[float, float, float],
    *,
    side_colors: dict[str, tuple[float, float, float]] | None = None,
    kpt_thr: float = 0.3,
    joint_r: float = 5.0,
    limb_w: float = 3.0,
    shadow_a: float = 0.35,
    bpart_filter: BodyPartFilter | None = None,
) -> np.ndarray:
    """
    Render skeleton on *img* (in-place + returned).

    Parameters
    ----------
    side_colors : optional dict with keys "left", "right", "center" → (r,g,b) 0-1.
                  When provided, joints / limbs are coloured by body side.
    bpart_filter: optional BodyPartFilter. When None, uses the module-level singleton.
    """
    h, w = img.shape[:2]
    surface, buf = _cairo_surface_from_bgr(img)
    ctx = cairo.Context(surface)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)

    kp = np.asarray(keypoints, dtype=np.float64)
    has_score = kp.ndim == 2 and kp.shape[1] >= 3
    vis = (kp[:, 2] >= kpt_thr) if has_score else np.ones(len(kp), bool)
    xy = kp[:, :2]
    K = len(xy)

    flt = bpart_filter if bpart_filter is not None else body_filter
    use_side = side_colors is not None

    def _jc(idx: int) -> tuple[float, float, float]:
        if use_side:
            side = flt.joint_side(idx, K)
            return side_colors[side]
        return joint_color

    def _lc_edge(a: int, b: int) -> tuple[float, float, float]:
        if use_side:
            side = flt.edge_side(a, b, K)
            return side_colors[side]
        return limb_color

    # ── limbs ────────────────────────────────────────────────────────────────
    for a, b in skeleton:
        if a >= K or b >= K:
            continue
        if not (vis[a] and vis[b]):
            continue
        if not flt.edge_visible(a, b, K):
            continue

        x0, y0 = xy[a]
        x1, y1 = xy[b]
        lc = _lc_edge(a, b)
        jc_a = _jc(a)
        jc_b = _jc(b)

        # Drop shadow
        ctx.set_line_width(limb_w + 4)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        shadow = cairo.LinearGradient(x0, y0, x1, y1)
        shadow.add_color_stop_rgba(0,   0, 0, 0, shadow_a * 0.6)
        shadow.add_color_stop_rgba(0.5, 0, 0, 0, shadow_a)
        shadow.add_color_stop_rgba(1,   0, 0, 0, shadow_a * 0.6)
        ctx.set_source(shadow)
        ctx.move_to(x0 + 1.5, y0 + 1.5)
        ctx.line_to(x1 + 1.5, y1 + 1.5)
        ctx.stroke()

        # Gradient limb: joint_a_color → limb_color → joint_b_color
        ctx.set_line_width(limb_w)
        grad = cairo.LinearGradient(x0, y0, x1, y1)
        grad.add_color_stop_rgba(0,   *jc_a, 0.92)
        grad.add_color_stop_rgba(0.5, *lc,   0.85)
        grad.add_color_stop_rgba(1,   *jc_b, 0.92)
        ctx.set_source(grad)
        ctx.move_to(x0, y0)
        ctx.line_to(x1, y1)
        ctx.stroke()

    # ── joints ───────────────────────────────────────────────────────────────
    for i, (x, y) in enumerate(xy):
        if not vis[i]:
            continue
        if not flt.joint_visible(i, K):
            continue

        jr = _jc(i)

        # Soft shadow
        shadow_pat = cairo.RadialGradient(x + 1, y + 2, 0, x + 1, y + 2, joint_r + 4)
        shadow_pat.add_color_stop_rgba(0, 0, 0, 0, 0.30)
        shadow_pat.add_color_stop_rgba(1, 0, 0, 0, 0.0)
        ctx.set_source(shadow_pat)
        ctx.arc(x + 1, y + 2, joint_r + 4, 0, 2 * np.pi)
        ctx.fill()

        # White outline
        ctx.set_source_rgba(1, 1, 1, 0.95)
        ctx.arc(x, y, joint_r + 2, 0, 2 * np.pi)
        ctx.fill()

        # Filled joint with radial gradient
        radial = cairo.RadialGradient(x - joint_r * 0.3, y - joint_r * 0.3, 0,
                                      x, y, joint_r)
        light = tuple(min(1.0, c + 0.3) for c in jr)
        radial.add_color_stop_rgba(0, *light, 1.0)
        radial.add_color_stop_rgba(1, *jr, 1.0)
        ctx.set_source(radial)
        ctx.arc(x, y, joint_r, 0, 2 * np.pi)
        ctx.fill()

        # Specular highlight
        ctx.set_source_rgba(1, 1, 1, 0.55)
        ctx.arc(x - joint_r * 0.32, y - joint_r * 0.32, joint_r * 0.28, 0, 2 * np.pi)
        ctx.fill()

    surface.flush()
    result = _bgra_to_bgr(buf, h, w)
    np.copyto(img, result)
    return img


# ──────────────────────────────────────────────────────────────────────────────
# HUD
# ──────────────────────────────────────────────────────────────────────────────

class _HUDRenderer:
    def __init__(self) -> None:
        self._fonts: dict = {}

    def _font(self, size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
        key = (size, bold)
        if key not in self._fonts:
            self._fonts[key] = _find_font_bold(size) if bold else _find_font(size)
        return self._fonts[key]

    def _tsz(self, draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
        bb = draw.textbbox((0, 0), text, font=font)
        return bb[2] - bb[0], bb[3] - bb[1]

    def _rounded_rect(
        self,
        draw: ImageDraw.ImageDraw,
        x0: int, y0: int, x1: int, y1: int,
        radius: int,
        fill: tuple,
        outline: tuple | None = None,
        outline_width: int = 1,
    ) -> None:
        draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=fill,
                                outline=outline, width=outline_width)

    def legend(
        self,
        layer: Image.Image,
        legends: list[str],
        joint_colors_f: list[tuple[float, float, float]],
        limb_colors_f:  list[tuple[float, float, float]],
        *,
        mg: int = 10,
    ) -> None:
        draw = ImageDraw.Draw(layer)
        font_title = self._font(14)
        font_name  = self._font(18, bold=True)

        dot_r  = 10
        limb_sw = 36
        limb_th = 6
        gap = 10
        pad = 20
        row_h = 52

        max_tw = max(self._tsz(draw, lg, font_name)[0] for lg in legends)
        swatch_w = dot_r * 2 + gap + limb_sw + gap
        pw = pad + swatch_w + max_tw + pad
        title_h = self._tsz(draw, "MODELS", font_title)[1]
        ph = pad + title_h + 10 + len(legends) * row_h + pad

        x0, y0, x1, y1 = mg, mg, mg + pw, mg + ph

        self._rounded_rect(draw, x0, y0, x1, y1, 10,
                           fill=(18, 20, 28, 210),
                           outline=(60, 65, 92, 200), outline_width=1)
        draw.rounded_rectangle([x0, y0, x0 + 4, y1], radius=2,
                                fill=(80, 140, 255, 255))
        draw.text((x0 + pad + 2, y0 + pad), "MODELS", font=font_title,
                  fill=(120, 128, 160, 200))

        for i, (lg, jc, lc) in enumerate(zip(legends, joint_colors_f, limb_colors_f)):
            cy = y0 + pad + title_h + 10 + i * row_h + row_h // 2
            jx = x0 + pad + dot_r + 2
            jc8 = tuple(int(c * 255) for c in jc) + (255,)
            draw.ellipse([jx - dot_r - 3, cy - dot_r - 3,
                          jx + dot_r + 3, cy + dot_r + 3],
                         fill=(240, 242, 250, 255))
            draw.ellipse([jx - dot_r, cy - dot_r, jx + dot_r, cy + dot_r],
                         fill=jc8)
            draw.ellipse([jx - dot_r // 2 - 1, cy - dot_r // 2 - 1,
                          jx - dot_r // 4,      cy - dot_r // 4],
                         fill=(255, 255, 255, 160))

            lx0 = jx + dot_r + gap
            lx1 = lx0 + limb_sw
            lc8 = tuple(int(c * 255) for c in lc) + (220,)
            draw.rounded_rectangle(
                [lx0, cy - limb_th // 2, lx1, cy + limb_th // 2],
                radius=limb_th // 2, fill=lc8,
            )

            tx = lx1 + gap
            th = self._tsz(draw, lg, font_name)[1]
            draw.text((tx, cy - th // 2), lg, font=font_name,
                      fill=(232, 235, 248, 255))


_hud = _HUDRenderer()


def render_hud(
    img: np.ndarray,
    legends: list[str],
    joint_colors: list[tuple[float, float, float]],
    limb_colors:  list[tuple[float, float, float]],
    current_idx: int,
    total: int,
) -> np.ndarray:
    h, w = img.shape[:2]
    base_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGBA))
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    _hud.legend(layer, legends, joint_colors, limb_colors)
    out_pil = Image.alpha_composite(base_pil, layer)
    return cv2.cvtColor(np.array(out_pil), cv2.COLOR_RGBA2BGR)