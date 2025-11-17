import cv2
from deploy2serve.deployment.projects.sapiens.utils.adapters import visualizer_adapter
from pycocotools.coco import COCO
from pathlib import Path
from mmengine.config import Config
from mmpose.structures import PoseDataSample
from mmpose.visualization.fast_visualizer import FastVisualizer
from tqdm import tqdm
import torch.cuda
from typing import List
from project.pipeline import MMPipeline


colors = [
    [0, 0, 255],
    [0, 255, 0],
    [255, 0, 0],
    [0, 255, 255],
    [255, 0, 255],
    [255, 255, 0],
]


def visual_difference(params: Config) -> None:
    pipelines: List[MMPipeline] = []
    streams: List[torch.cuda.Stream] = []
    for model in params.models:
        pipelines.append(
            MMPipeline(pose_checkpoint=model.pose_checkpoints, pose_config=model.pose_config)
        )
        streams.append(torch.cuda.Stream())

    coco = COCO(params.ann_file)
    catIds = coco.getCatIds(catNms=["person"])
    imgIds = coco.getImgIds(catIds=catIds)

    num_joints = pipelines[0].model_cfg.num_keypoints
    if num_joints == 26:
        from project.dataset_viewer import (COCO_HALPE26_SKELETON_INFO as SKELETON_INFO,
                                            COCO_HALPE26_KPTS_COLORS as KPTS_COLORS)
    elif num_joints == 133:
        from deploy2serve.deployment.projects.sapiens.utils.palettes import (
            COCO_WHOLEBODY_SKELETON_INFO as SKELETON_INFO, COCO_WHOLEBODY_KPTS_COLORS as KPTS_COLORS
        )
    else:
        raise Exception(f"Undefined meta information for visualization with such number of joints: {num_joints}.")
    meta_info = visualizer_adapter(SKELETON_INFO, KPTS_COLORS)

    visualizer = FastVisualizer(meta_info, radius=3, line_width=2, kpt_thr=0.)
    cv2.namedWindow("Mapping", cv2.WINDOW_GUI_EXPANDED)

    for image_id in tqdm(imgIds):
        img_instances = coco.loadImgs(image_id)
        for instance in img_instances:
            image_path = f'{params.dataset_path}/{instance["file_name"]}'
            image = cv2.imread(image_path)
            bboxes = [item["bbox"] for item in coco.imgToAnns[instance["id"]]]

            pose_data_samples: List[List[PoseDataSample]] = []
            for idx in range(len(pipelines)):
                streams[idx].wait_stream(torch.cuda.default_stream())
                with torch.cuda.stream(streams[idx]):
                    pose_data_samples.append(pipelines[idx](image_path, bboxes))
                torch.cuda.default_stream().wait_stream(streams[idx])

            for idx, pose_data_sample in enumerate(pose_data_samples):

                link_fixed_colors = SKELETON_INFO.copy()
                for item in link_fixed_colors.values():
                    item["color"] = colors[idx]
                meta_info = visualizer_adapter(link_fixed_colors,[colors[idx]] * len(KPTS_COLORS))
                visualizer.keypoint_colors = meta_info["keypoint_colors"]
                visualizer.skeleton_link_colors = meta_info["skeleton_link_colors"]

                for data_sample in pose_data_sample:
                    visualizer.draw_pose(image, data_sample.pred_instances)
                cv2.putText(image, Path(params.models[idx].pose_checkpoints).name, (20, 25 + 25 * idx),
                            cv2.FONT_HERSHEY_COMPLEX, 1, colors[idx], thickness=2)
            cv2.imshow("Mapping", image)
            cv2.waitKey(0)


if __name__ == "__main__":
    params = Config(
            dict(
                dataset_path=r"D:/NewPoseCustom/2025-10-20 13-00-09",
                ann_file=r"E:/Projects/mmpose/project/annotations/detection/anns/coco/2025-10-20 13-00-09.json",
                models=[
                    dict(
                        pose_checkpoints="../checkpoints/rtmpose/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.pth",
                        pose_config="../../configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb512-700e_body8-halpe26-256x192.py"
                    ),
                    dict(
                        pose_checkpoints="../../work_dirs/rtmpose-m_8xb512-700e_body8-halpe26-256x192/best_AUC_epoch_29.pth",
                        pose_config="../../configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb512-700e_body8-halpe26-256x192.py"
                    ),
                ]
            )
        )

    visual_difference(params)
