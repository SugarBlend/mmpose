import cv2
import numpy as np
from PIL import Image, ImageTk
from typing import Any

from project.visualizer.models import DrawParams
from project.visualizer.constants import KPT_BGR, SKELETON, BBOX_PALETTE_BGR, USE_CUDA


def outlined(img: np.ndarray, text: str, pos: tuple[int, int], font: int, scale: float, color: tuple[int, int, int],
             thickness: int = 1) -> None:
    x, y = pos
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx or dy:
                cv2.putText(img, text, (x + dx, y + dy), font, scale, (0, 0, 0), thickness + 1)
    cv2.putText(img, text, pos, font, scale, color, thickness)

def kpt_color_bgr(i: int) -> tuple[int, int, int]:
    return KPT_BGR[i] if i < len(KPT_BGR) else (180, 180, 180)

def blend_rect(
    img: np.ndarray, x1: int, y1: int, x2: int, y2: int, color_bgr: tuple[int, int, int], alpha: float
) -> None:
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1: y2, x1: x2]
    ov = np.empty_like(roi)
    ov[:] = color_bgr
    cv2.addWeighted(ov, alpha, roi, 1 - alpha, 0, roi)
    img[y1: y2, x1: x2] = roi

def draw_tracks(img: np.ndarray, track_window: dict[int, Any], params: DrawParams) -> None:
    for _ai, kd in track_window.items():
        for ki, pts_list in kd.items():
            base = kpt_color_bgr(ki)
            pts = [(x, y) for _, x, y in pts_list][-max(1, params.track_length):]
            if len(pts) < 2:
                continue
            n = len(pts)
            for i in range(1, n):
                t = i / n
                alpha = t * params.track_alpha
                c = tuple(int(ch * alpha) for ch in base)
                thick = max(1, int(params.skel_thickness * t * 0.8))
                cv2.line(img, pts[i - 1], pts[i], c, thick)

def draw_frame(img: np.ndarray, annotations: list[dict[str, Any]], params: DrawParams, track_window = None) -> None:
    if params.show_tracks and track_window:
        draw_tracks(img, track_window, params)

    for idx, ann in enumerate(annotations):
        if params.show_bbox and ann.get("bbox"):
            bbox = ann["bbox"]
            if len(bbox) == 4:
                x, y, w, h = (int(item) for item in bbox)
                bc = BBOX_PALETTE_BGR[idx % len(BBOX_PALETTE_BGR)]
                if params.show_bbox_fill:
                    blend_rect(img, x, y, x + w, y + h, bc, params.bbox_fill_alpha)

                cv2.rectangle(img, (x, y), (x + w, y + h), bc, 2)
                if params.show_ann_ids:
                    outlined(img, f"#{idx}", (x + 4, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bc, 1)
        joints = ann.get("keypoints", [])
        if not joints or len(joints) % 3 != 0:
            continue

        num_joints = len(joints) // 3
        if params.show_skeleton:
            for i1, i2, sc in SKELETON:
                if i1 >= num_joints or i2 >= num_joints:
                    continue
                x1, y1, v1 = int(joints[i1 * 3]), int(joints[i1 * 3 + 1]), joints[i1 * 3 + 2]
                x2, y2, v2 = int(joints[i2 * 3]), int(joints[i2 * 3 + 1]), joints[i2 * 3 + 2]
                if v1 > 0 and v2 > 0:
                    cv2.line(img,(x1, y1),(x2, y2), sc, params.skel_thickness)

        if params.show_keypoints:
            r = max(1, int(params.point_radius))
            for i in range(num_joints):
                x, y, v = int(joints[i * 3]), int(joints[i * 3 + 1]), joints[i * 3 + 2]
                if v > 0:
                    c = kpt_color_bgr(i)
                    cv2.circle(img,(x, y), r + 1,(0, 0, 0),1)
                    cv2.circle(img,(x, y), r, c, -1)
                    if params.show_joint_ids:
                        outlined(img, str(i),(x + r + 2, y - r - 2), cv2.FONT_HERSHEY_SIMPLEX,
                                 params.font_scale, c, 1)

def fit_image(iw: int, ih: int, cw: int, ch: int) -> tuple[int, int]:
    if iw / ih > cw / ch:
        return cw, max(1, int(cw * ih / iw))
    return max(1, int(ch * iw / ih)), ch


def resize(img_rgb: np.ndarray, nw: int, nh: int) -> np.ndarray:
    if USE_CUDA and nw * nh < img_rgb.shape[0] * img_rgb.shape[1]:
        try:
            gpu = cv2.cuda_GpuMat()
            gpu.upload(img_rgb)
            gpu_resized = cv2.cuda.resize(gpu, (nw, nh), interpolation=cv2.INTER_LINEAR)
            return gpu_resized.download()
        except Exception:
            pass
    interp = cv2.INTER_AREA if (nw < img_rgb.shape[1]) else cv2.INTER_LINEAR
    return cv2.resize(img_rgb, (nw, nh), interpolation=interp)


def rgb_to_photoimage(img_rgb: np.ndarray) -> ImageTk.PhotoImage:
    h, w = img_rgb.shape[:2]
    pil_image = Image.frombuffer("RGB", (w, h), img_rgb, "raw", "RGB", 0, 1)
    return ImageTk.PhotoImage(pil_image)
