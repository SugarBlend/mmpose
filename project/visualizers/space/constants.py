import cv2

def _check_cuda():
    try:
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            return True
    except Exception:
        pass
    return False

USE_CUDA = _check_cuda()

FPS_PRESETS = {
    1: ("1 FPS", 1000), 2: ("5 FPS", 200), 3: ("10 FPS", 100),
    4: ("30 FPS", 33), 5: ("60 FPS", 17), 6: ("120 FPS", 8)
}

BBOX_PALETTE_BGR = [
    (0, 180, 255), (0, 220, 100), (0, 100, 255), (180, 0, 255),
    (0, 220, 220), (0, 180, 255), (80, 220, 0), (220, 0, 180),
]

PANEL_BORDER_COLORS = [
    "#e06c4a", "#4a9ee0", "#4acd7a", "#c47ae0",
    "#e0c54a", "#4ae0d9", "#e04a8b", "#8be04a",
]

W = {
    "bg":       "#f0f0f0",
    "panel":    "#e8e8e8",
    "toolbar":  "#f5f5f5",
    "surface":  "#ffffff",
    "surf2":    "#ebebeb",
    "canvas":   "#1a1a1a",
    "row_alt":  "#f7f7f7",
    "text":     "#1a1a1a",
    "dim":      "#707070",
    "label":    "#444444",
    "accent":   "#0066cc",
    "acc_h":    "#0050aa",
    "green":    "#1a7a3a",
    "red":      "#cc2222",
    "border":   "#c8c8c8",
    "sep":      "#d0d0d0",
    "tab_act":  "#ffffff",
    "tab_bg":   "#f5f5f5",
    "tab_h":    "#e0e0e0",
}
