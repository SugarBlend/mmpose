import colorsys
from pathlib import Path
import cairo
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

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

# Public: per-model (r,g,b) tuples in 0-1 range
MODEL_JOINT_COLORS: list[tuple[float, float, float]] = [
    _hsl(h, js, jl) for h, js, jl, _, _ in _SPECS
]
MODEL_LIMB_COLORS: list[tuple[float, float, float]] = [
    _hsl(h, ls, ll) for h, _, _, ls, ll in _SPECS
]

# OpenCV BGR uint8 versions (for anything still needing cv2)
def _to_bgr(c: tuple[float, float, float]) -> tuple[int, int, int]:
    r, g, b = c
    return int(b * 255), int(g * 255), int(r * 255)

MODEL_COLORS = [_to_bgr(c) for c in MODEL_JOINT_COLORS]
MODEL_LIMB_COLORS_BGR = [_to_bgr(c) for c in MODEL_LIMB_COLORS]


def _find_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        # macOS
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        # Windows
        "C:/Windows/Fonts/Comic Sans MS.ttf",
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


def _cairo_surface_from_bgr(img: np.ndarray) -> tuple[cairo.ImageSurface, np.ndarray]:
    h, w = img.shape[:2]
    # Cairo wants BGRA
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
    joint_color: tuple[float, float, float], # (r,g,b) 0-1
    limb_color:  tuple[float, float, float], # (r,g,b) 0-1
    *,
    kpt_thr: float = 0.3,
    joint_r: float = 5.0,
    limb_w: float = 3.0,
    shadow_a: float = 0.35,
) -> np.ndarray:
    h, w = img.shape[:2]
    surface, buf = _cairo_surface_from_bgr(img)
    ctx = cairo.Context(surface)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)

    kp = np.asarray(keypoints, dtype=np.float64)
    has_score = kp.ndim == 2 and kp.shape[1] >= 3
    vis = (kp[:, 2] >= kpt_thr) if has_score else np.ones(len(kp), bool)
    xy = kp[:, :2]
    K = len(xy)

    jr, lc = joint_color, limb_color

    for a, b in skeleton:
        if a >= K or b >= K or not (vis[a] and vis[b]):
            continue
        x0, y0 = xy[a]
        x1, y1 = xy[b]

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

        # Gradient limb: joint_color → limb_color → joint_color
        ctx.set_line_width(limb_w)
        grad = cairo.LinearGradient(x0, y0, x1, y1)
        grad.add_color_stop_rgba(0,   *jr, 0.92)
        grad.add_color_stop_rgba(0.5, *lc, 0.85)
        grad.add_color_stop_rgba(1,   *jr, 0.92)
        ctx.set_source(grad)
        ctx.move_to(x0, y0)
        ctx.line_to(x1, y1)
        ctx.stroke()

    for i, (x, y) in enumerate(xy):
        if not vis[i]:
            continue

        # Soft shadow under joint
        shadow_pat = cairo.RadialGradient(x+1, y+2, 0, x+1, y+2, joint_r + 4)
        shadow_pat.add_color_stop_rgba(0,   0, 0, 0, 0.30)
        shadow_pat.add_color_stop_rgba(1,   0, 0, 0, 0.0)
        ctx.set_source(shadow_pat)
        ctx.arc(x + 1, y + 2, joint_r + 4, 0, 2 * np.pi)
        ctx.fill()

        # White outline
        ctx.set_source_rgba(1, 1, 1, 0.95)
        ctx.arc(x, y, joint_r + 2, 0, 2 * np.pi)
        ctx.fill()

        # Filled joint with subtle radial gradient (light center → full color edge)
        radial = cairo.RadialGradient(x - joint_r * 0.3, y - joint_r * 0.3, 0,
                                      x, y, joint_r)
        light = tuple(min(1.0, c + 0.3) for c in jr)
        radial.add_color_stop_rgba(0,   *light, 1.0)
        radial.add_color_stop_rgba(1,   *jr,    1.0)
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


class _HUDRenderer:
    def __init__(self):
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
        fill: tuple[int, int, int, int],
        outline: tuple[int, int, int, int] | None = None,
        outline_width: int = 1,
    ):
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

        dot_r = 10
        limb_sw = 36 # limb swatch width
        limb_th = 6 # limb swatch thickness
        gap = 10
        pad = 20
        row_h = 52

        # measure widest name
        max_tw = max(self._tsz(draw, lg, font_name)[0] for lg in legends)
        swatch_w = dot_r * 2 + gap + limb_sw + gap
        pw = pad + swatch_w + max_tw + pad
        title_h = self._tsz(draw, "MODELS", font_title)[1]
        ph = pad + title_h + 10 + len(legends) * row_h + pad

        x0, y0, x1, y1 = mg, mg, mg + pw, mg + ph

        self._rounded_rect(draw, x0, y0, x1, y1, 10,
                           fill=(18, 20, 28, 210),
                           outline=(60, 65, 92, 200), outline_width=1)

        # Left accent stripe
        draw.rounded_rectangle([x0, y0, x0 + 4, y1], radius=2,
                               fill=(80, 140, 255, 255))

        # Title
        draw.text((x0 + pad + 2, y0 + pad), "MODELS", font=font_title,
                  fill=(120, 128, 160, 200))

        for i, (lg, jc, lc) in enumerate(zip(legends, joint_colors_f, limb_colors_f)):
            cy = y0 + pad + title_h + 10 + i * row_h + row_h // 2

            # Joint swatch — filled circle with white ring
            jx = x0 + pad + dot_r + 2
            jc8 = tuple(int(c * 255) for c in jc) + (255,)
            draw.ellipse([jx - dot_r - 3, cy - dot_r - 3,
                          jx + dot_r + 3, cy + dot_r + 3],
                         fill=(240, 242, 250, 255))
            draw.ellipse([jx - dot_r, cy - dot_r, jx + dot_r, cy + dot_r],
                         fill=jc8)
            # specular
            draw.ellipse([jx - dot_r // 2 - 1, cy - dot_r // 2 - 1,
                          jx - dot_r // 4,      cy - dot_r // 4],
                         fill=(255, 255, 255, 160))

            # Limb swatch — rounded line
            lx0 = jx + dot_r + gap
            lx1 = lx0 + limb_sw
            lc8 = tuple(int(c * 255) for c in lc) + (220,)
            draw.rounded_rectangle(
                [lx0, cy - limb_th // 2, lx1, cy + limb_th // 2],
                radius=limb_th // 2, fill=lc8,
            )

            # Model name
            tx = lx1 + gap
            th = self._tsz(draw, lg, font_name)[1]
            draw.text((tx, cy - th // 2), lg, font=font_name,
                      fill=(232, 235, 248, 255))

    def counter(
        self,
        layer: Image.Image,
        cur: int,
        total: int,
        *,
        mg: int = 18,
    ) -> None:
        draw = ImageDraw.Draw(layer)
        font = self._font(26, bold=True)
        h, w = layer.height, layer.width
        txt = f"{cur} / {total}"
        tw, th = self._tsz(draw, txt, font)
        pad = 14
        bw, bh = tw + pad * 2, th + pad * 2
        x0 = w - bw - mg
        y0 = h - bh - mg
        self._rounded_rect(draw, x0, y0, x0 + bw, y0 + bh, 10,
                           fill=(18, 20, 28, 210),
                           outline=(60, 65, 92, 190))
        draw.text((x0 + pad, y0 + pad), txt, font=font,
                  fill=(180, 185, 210, 240))

    def progress_bar(
        self,
        layer: Image.Image,
        cur: int,
        total: int,
        *,
        height: int = 6,
    ) -> None:
        draw = ImageDraw.Draw(layer)
        w = layer.width
        h = layer.height
        filled = int(w * cur / max(total, 1))

        # Track
        draw.rectangle([0, h - height, w, h], fill=(22, 24, 34, 200))

        # Gradient fill
        if filled > 1:
            A = np.array([80, 140, 255], dtype=np.float32) # accent blue
            B = np.array([55, 200, 130], dtype=np.float32) # teal
            for x in range(filled):
                t = x / max(filled - 1, 1)
                col = ((1 - t) * A + t * B).astype(np.uint8)
                draw.line([(x, h - height), (x, h - 1)],
                          fill=(int(col[0]), int(col[1]), int(col[2]), 230))

    def hints(self, layer: Image.Image, *, mg: int = 18) -> None:
        draw = ImageDraw.Draw(layer)
        font_key  = self._font(20, bold=True)
        font_desc = self._font(18)
        h = layer.height
        items = [("Q", "Quit"), ("ANY KEY", "Next")]
        x = mg
        y = h - mg - 28

        for key, desc in items:
            kw, kh = self._tsz(draw, key, font_key)
            self._rounded_rect(draw, x - 8, y - 4, x + kw + 8, y + kh + 4, 5,
                               fill=(48, 52, 72, 200),
                               outline=(80, 85, 115, 180))
            draw.text((x, y), key, font=font_key, fill=(228, 232, 248, 255))
            x += kw + 18
            dw, dh = self._tsz(draw, desc, font_desc)
            draw.text((x, y + (kh - dh) // 2), desc, font=font_desc,
                      fill=(115, 122, 155, 220))
            x += dw + 28


_hud = _HUDRenderer()


def render_hud(
    img: np.ndarray,
    legends: list[str],
    joint_colors: list[tuple[float, float, float]],
    limb_colors: list[tuple[float, float, float]],
    current_idx: int,
    total: int,
) -> np.ndarray:
    h, w = img.shape[:2]

    # Convert frame to RGBA Pillow image
    base_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGBA))

    # Transparent overlay layer
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))

    _hud.legend(layer, legends, joint_colors, limb_colors)
    # _hud.counter(layer, current_idx, total)
    # _hud.progress_bar(layer, current_idx, total)
    # _hud.hints(layer)

    # Composite
    out_pil = Image.alpha_composite(base_pil, layer)
    return cv2.cvtColor(np.array(out_pil), cv2.COLOR_RGBA2BGR)
