import argparse
import cv2
from tqdm import tqdm
from pathlib import Path
from pycocotools.coco import COCO
from dotenv import load_dotenv
from project.label_studio.pipelines.pipeline import MMPipeline
from pipeline import HookedPipeline
from reader import S3Reader
from visualizers import GroupedSimCCVisualizer, GroupedHeatmapVisualizer
import mmpose.codecs


def visualize(pose_checkpoint: str, pose_config: str, ann_path: str, indices: list[int] = None) -> None:
    coco = COCO(ann_path)
    reader = S3Reader()
    pipeline = HookedPipeline(MMPipeline(pose_checkpoint, pose_config))

    if pipeline.pipeline.codec.type == mmpose.codecs.simcc_label.SimCCLabel.__name__:
        visualizer = GroupedSimCCVisualizer()
    elif pipeline.pipeline.codec.type == mmpose.codecs.udp_heatmap.UDPHeatmap.__name__:
        visualizer = GroupedHeatmapVisualizer()
    else:
        raise RuntimeError

    pbar = tqdm(total=len(coco.imgToAnns.items()), desc="Progress")

    for fn, desc in coco.imgToAnns.items():
        path = coco.imgs[fn]["file_name"]
        image = reader.fetch(path, bucket_name="datasets")

        for person_id, ann in enumerate(desc):
            pipeline(image, [ann["bbox"]])

            out_img = visualizer.draw_instance_heatmap(
                pipeline.heatmap, pipeline.overlay, indices=indices, mix=True, weight=0.5
            )

            if args.save:
                filename = Path(f'finetune_vitpose_/{Path(path).stem}_{pbar.n}_p{person_id}.png')
                if not filename.exists():
                    filename.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(filename.as_posix(), out_img)
            if args.show:
                cv2.imshow("", out_img)
                cv2.waitKey(0)
        pbar.update()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Heatmap visualizer")
    parser.add_argument("--ann_path",
                        default="../../annotations/labelstudio/Hands + Body Pose Annotation (2026-05-04 16-50-13).json",
                        help="The base path relative to which paths to the data described in the annotation are constructed.")
    parser.add_argument("--pose_checkpoint",
                        default="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606.pth",
                        help="Path to weights.")
    parser.add_argument("--pose_config",
                        default="../../../configs/body_2d_keypoint/rtmpose/body8/rtmpose-x_8xb256-700e_body8-halpe26-384x288.py",
                        help="Path to model configuration.")
    parser.add_argument("--indices", default=None, help="Indices of joints for collaborative visualization heatmaps.")
    parser.add_argument("--save", action="store_true",
                        help="Save results in report folder?")
    parser.add_argument("--show", action="store_false", help="Show visualization window?")
    parser.add_argument("--delay", default=0, help="Delay before visualize new frame in ms.")
    return parser.parse_args()


if __name__ == "__main__":
    load_dotenv("../../../tools/.env")
    args = parse_args()
    visualize(args.pose_checkpoint, args.pose_config, args.ann_path, args.indices)
