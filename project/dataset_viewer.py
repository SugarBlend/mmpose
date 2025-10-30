from pycocotools.coco import COCO
import cv2
import numpy as np
from pathlib import Path
from mmpose.visualization.fast_visualizer import FastVisualizer
from deploy2serve.deployment.projects.sapiens.utils.adapters import visualizer_adapter
from deploy2serve.deployment.projects.sapiens.utils.palettes import (COCO_WHOLEBODY_SKELETON_INFO, COCO_WHOLEBODY_KPTS_COLORS,
                                                                     COCO_HALPE26_SKELETON_INFO, COCO_HALPE26_KPTS_COLORS)
from mmengine import Config
from typing import Optional


def visualize_annotations(
    annotation_file: str,
    images_folder: str,
    num_joints: int,
    vis_delay: int = 100,
    offset: Optional[int] = None
) -> None:
    coco = COCO(annotation_file)
    images_folder = Path(images_folder)

    if num_joints == 26:
        meta_info = visualizer_adapter(COCO_HALPE26_SKELETON_INFO, COCO_HALPE26_KPTS_COLORS)
    elif num_joints == 133:
        meta_info = visualizer_adapter(COCO_WHOLEBODY_SKELETON_INFO, COCO_WHOLEBODY_KPTS_COLORS)
    else:
        raise Exception(f"Undefined meta information for visualization with such number of joints: {num_joints}.")

    visualizer = FastVisualizer(meta_info, radius=3, line_width=1, kpt_thr=0.3)

    cv2.namedWindow("Mapping", cv2.WINDOW_GUI_EXPANDED)
    for idx, anns in coco.imgToAnns.items():
        filename = coco.imgs[idx]["file_name"]
        if not Path(filename).is_absolute():
            path = images_folder.joinpath(filename)
        else:
            path = Path(filename)
            path = Path(images_folder).joinpath(path.parents[0].stem, path.name)

        if offset is not None:
            if (int(path.stem) - offset) % 4:
                continue

        image = cv2.imread(path.as_posix())
        for ann in anns:
            if not ann.get("keypoints"):
                continue

            if ann.get("righthand_kpts"):
                joints = [
                    *ann["keypoints"], *ann["foot_kpts"], *ann["face_kpts"], *ann["lefthand_kpts"],
                    *ann["righthand_kpts"]
                ]
                joints = np.asarray(joints).reshape(1, -1, 3)

            elif num_joints == 26:
                joints = np.asarray(ann["keypoints"]).reshape(1, -1, 3)[:, :num_joints]

            mask = joints[:, :, 2] > 0.3
            visualizer.draw_pose(
                image, Config({"keypoints": joints[:, :, :2], "keypoint_scores": mask})
            )
            cv2.putText(image, text=f"{ann['image_id'] + 1}", org=(10, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=2, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA, bottomLeftOrigin=False)
        cv2.imshow("Mapping", image)
        cv2.waitKey(vis_delay)


if __name__ == "__main__":
    num_joints = 26
    ann_file = r"C:\Users\Alexander\Downloads\project-3-at-2025-10-30-11-29-5fd18df9\result.json"
    # ann_file = "../data/coco/annotations/coco_wholebody_val_v1.0.json"
    # ann_file = "../data/coco/annotations/halpe_val_v1.json"
    # images_folder = "../data/detection/coco/val2017"
    images_folder = "D:/NewPoseCustom"
    visualize_annotations(ann_file, images_folder, num_joints=26, offset=-2)
