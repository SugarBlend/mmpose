from dataclasses import dataclass, field
from typing import List


@dataclass
class FilterParams:
    patterns:  List[str] = field(default_factory=lambda: [".*"])
    operators: List[str] = field(default_factory=list)


@dataclass
class DrawParams:
    show_bbox: bool = True
    show_bbox_fill: bool = True
    bbox_fill_alpha: float = 0.15
    show_keypoints: bool = True
    show_skeleton: bool = True
    show_joint_ids: bool = False
    show_ann_ids: bool = False
    show_frame_label: bool = True
    show_tracks: bool = False
    track_length: int = 30
    track_alpha: float = 0.7
    point_radius: float = 5.0
    font_scale: float = 0.4
    skel_thickness: int = 2


@dataclass
class ImageEntry:
    image_id: int
    file_name: str
    annotations: list = field(default_factory=list)


@dataclass
class ViewGroupConfig:
    name: str
    patterns: List[str] = field(default_factory=lambda: [".*"])
    operators: List[str] = field(default_factory=list)
    color: str = "#4a9ee0"
