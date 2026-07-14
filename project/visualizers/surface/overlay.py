from __future__ import annotations

import colorsys
from typing import Literal

import cairo
import cv2
import numpy as np


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


Side = Literal["left", "right", "center"]

# fmt: off
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
    m: dict[int, tuple[str, Side]] = {}

    # Body: 0-16
    for idx, val in _COCO17_JOINT_META.items():
        m[idx] = val

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
    m: dict[int, tuple[str, Side]] = dict(_HALPE26_JOINT_META)
    for idx in range(26, 47):
        m[idx] = ("hands", "left")
    for idx in range(47, 68):
        m[idx] = ("hands", "right")
    for idx in range(68, 136):
        m[idx] = ("face", "center")
    return m

_HALPE136_JOINT_META = _halpe136_meta()


def _goliath308_meta() -> dict[int, tuple[str, Side]]:
    m: dict[int, tuple[str, Side]] = {}

    # 0-4: nose, left_eye, right_eye, left_ear, right_ear
    m[0]  = ("body", "center")  # nose
    m[1]  = ("body", "left")    # left_eye
    m[2]  = ("body", "right")   # right_eye
    m[3]  = ("body", "left")    # left_ear
    m[4]  = ("body", "right")   # right_ear
    # 5-6: shoulders
    m[5]  = ("body", "left")    # left_shoulder
    m[6]  = ("body", "right")   # right_shoulder
    # 7-8: elbows
    m[7]  = ("body", "left")    # left_elbow
    m[8]  = ("body", "right")   # right_elbow
    # 9-10: hips
    m[9]  = ("body", "left")    # left_hip
    m[10] = ("body", "right")   # right_hip
    # 11-12: knees
    m[11] = ("body", "left")    # left_knee
    m[12] = ("body", "right")   # right_knee
    # 13-14: ankles
    m[13] = ("body", "left")    # left_ankle
    m[14] = ("body", "right")   # right_ankle

    # 15-20: feet
    m[15] = ("feet", "left")    # left_big_toe
    m[16] = ("feet", "left")    # left_small_toe
    m[17] = ("feet", "left")    # left_heel
    m[18] = ("feet", "right")   # right_big_toe
    m[19] = ("feet", "right")   # right_small_toe
    m[20] = ("feet", "right")   # right_heel

    # 21-41: right hand
    for idx in range(21, 42):
        m[idx] = ("hands", "right")

    # 42-62: left hand
    for idx in range(42, 63):
        m[idx] = ("hands", "left")

    # 63-68: extra body landmarks
    m[63] = ("body", "left")    # left_olecranon
    m[64] = ("body", "right")   # right_olecranon
    m[65] = ("body", "left")    # left_cubital_fossa
    m[66] = ("body", "right")   # right_cubital_fossa
    m[67] = ("body", "left")    # left_acromion
    m[68] = ("body", "right")   # right_acromion

    # 69: neck
    m[69] = ("body", "center")

    _face_70_219_sides: list[Side] = [
        # 70-77: center (glabella, nose_root, nose_bridge x4, labiomental, chin)
        "center", "center", "center", "center", "center", "center", "center", "center",
        # 78-86: right eyebrow (9 pts)
        "right", "right", "right", "right", "right", "right", "right", "right", "right",
        # 87-95: left eyebrow (9 pts)
        "left", "left", "left", "left", "left", "left", "left", "left", "left",
        # 96-119: left eyelid upper (24 pts: lash_line x9 + eyelid_line x8 + crease_line x7)
        "left", "left", "left", "left", "left", "left", "left", "left", "left",
        "left", "left", "left", "left", "left", "left", "left", "left",
        "left", "left", "left", "left", "left", "left", "left",
        # 120-143: right eyelid upper (24 pts)
        "right", "right", "right", "right", "right", "right", "right", "right", "right",
        "right", "right", "right", "right", "right", "right", "right", "right",
        "right", "right", "right", "right", "right", "right", "right",
        # 144-160: left eyelid lower (17 pts: lash_line x9 + eyelid_line x8)
        "left", "left", "left", "left", "left", "left", "left", "left", "left",
        "left", "left", "left", "left", "left", "left", "left", "left",
        # 161-177: right eyelid lower (17 pts)
        "right", "right", "right", "right", "right", "right", "right", "right", "right",
        "right", "right", "right", "right", "right", "right", "right", "right",
        # 178-187: nose detail (10 pts)
        # 178=tip_of_nose, 179=bottom_center, 180=r_outer_corner, 181=l_outer_corner
        # 182-184=r_nostril, 185-187=l_nostril
        "center", "center", "right", "left", "right", "right", "right", "left", "left", "left",
        # 188-219: mouth (32 pts)
        # 188=r_outer_corner, 189=l_outer_corner, 190=cupid_bow(center), 191=lower_center
        # 192-203: outer lip points (mix left/right/center)
        # 204=r_inner, 205=l_inner, 206-219: inner lip
        "right", "left", "center", "center",
        "right", "left", "right", "left", "right", "right", "right", "right",
        "left", "left", "left", "left",
        "right", "left", "center", "center",
        "right", "left", "right", "right", "right", "right",
        "left", "left", "left", "left",
    ]
    for i, side in enumerate(_face_70_219_sides):
        m[70 + i] = ("face", side)

    # 220-245: left ear (26 pts, previous 256-281)
    for idx in range(220, 246):
        m[idx] = ("face", "left")

    # 246-271: right ear (26 pts, previous 282-307)
    for idx in range(246, 272):
        m[idx] = ("face", "right")

    # 272-280: left iris (9 pts, previous 308-316)
    for idx in range(272, 281):
        m[idx] = ("face", "left")

    # 281-289: right iris (9 pts, previous 317-325)
    for idx in range(281, 290):
        m[idx] = ("face", "right")

    # 290-298: left pupil (9 pts, previous 326-334)
    for idx in range(290, 299):
        m[idx] = ("face", "left")

    # 299-307: right pupil (9 pts, previous 335-343)
    for idx in range(299, 308):
        m[idx] = ("face", "right")

    return m


_SAPIENS308_JOINT_META = _goliath308_meta()


FORMAT_JOINT_META: dict[int, dict[int, tuple[str, Side]]] = {
    17:  _COCO17_JOINT_META,
    26:  _HALPE26_JOINT_META,
    133: _COCO133_JOINT_META,
    136: _HALPE136_JOINT_META,
    308: _SAPIENS308_JOINT_META,
}
# fmt: on


class BodyPartFilter:
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


def _cairo_surface_from_bgr(img: np.ndarray) -> tuple[cairo.ImageSurface, np.ndarray]:
    h, w = img.shape[:2]
    bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    buf = bgra.flatten()
    surface = cairo.ImageSurface.create_for_data(buf, cairo.FORMAT_ARGB32, w, h, w * 4)
    return surface, buf


def _bgra_to_bgr(buf: np.ndarray, h: int, w: int) -> np.ndarray:
    bgra = buf.reshape(h, w, 4)
    return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)


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
    h, w = img.shape[:2]
    surface, buf = _cairo_surface_from_bgr(img)
    ctx = cairo.Context(surface)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)

    kp = np.asarray(keypoints, dtype=np.float64)
    has_score = kp.ndim == 2 and kp.shape[1] >= 3
    kp[:, 2] /= max(1., max(kp[:, 2]))
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

