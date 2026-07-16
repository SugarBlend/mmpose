import project # for torch patching
import argparse
from pathlib import Path
import cv2
import numpy as np
from dotenv import load_dotenv
from pycocotools.coco import COCO
from mmpose.apis import init_model, inference_topdown
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples
from tqdm import tqdm
from reader import S3Reader


def build_visualizer(model):
    visualizer_cfg = model.cfg.visualizer.copy()
    visualizer_cfg.setdefault('type', 'PoseLocalVisualizer')
    visualizer_cfg['vis_backends'] = []
    visualizer = VISUALIZERS.build(visualizer_cfg)
    visualizer.set_dataset_meta(
        model.dataset_meta,
        skeleton_style=visualizer_cfg.get('skeleton_style', 'mmpose'),
    )
    return visualizer


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Validation heatmap visualizer (official mmpose path)")
    parser.add_argument("--pose_config", default="E:/Projects/mmpose/experiments_/delicate-chimp-466/config.py", help="Path to model config.")
    parser.add_argument("--pose_checkpoint", default="E:/Projects/mmpose/experiments_/delicate-chimp-466/model.pth", help="Path/URL to weights.")
    parser.add_argument("--ann_path",
                        default="E:/Projects/mmpose/project/annotations/labelstudio_preds/Hands + Body Pose Annotation (2026-05-04 16-54-24).json",
                        help="COCO-annotation with bbox.")
    parser.add_argument("--bucket", default="datasets", help="S3 bucket for images fetching.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out_dir", default="val_heatmaps")
    parser.add_argument("--kpt_thr", type=float, default=0.3)
    parser.add_argument("--draw_bbox", action="store_true")
    parser.add_argument("--flip_test", action="store_true",
                        help="Enable flip_test (if not specified, it is taken from the model config).")
    parser.add_argument("--save", default=False, action="store_true", help="Save results to disk.")
    parser.add_argument("--show", default=True, action="store_true", help="Show windows cv2.imshow.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv("../../../tools/.env")

    model = init_model(args.pose_config, args.pose_checkpoint, device=args.device)

    if getattr(model, "test_cfg", None) is not None:
        model.test_cfg["output_heatmaps"] = True
        if args.flip_test:
            model.test_cfg["flip_test"] = True

    visualizer = build_visualizer(model)

    coco = COCO(args.ann_path)
    reader = S3Reader()

    out_dir = Path(args.out_dir)
    if args.save:
        out_dir.mkdir(parents=True, exist_ok=True)

    img_items = list(coco.imgToAnns.items())

    cameras = range(4) # four sequentially cameras
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_folder = Path("recordings")
    output_folder.mkdir(parents=True, exist_ok=True)
    writers: list[cv2.VideoWriter] | None = None

    for img_id, anns in tqdm(img_items):
        file_name = coco.imgs[img_id]["file_name"]
        image_bgr = reader.fetch(file_name, bucket_name=args.bucket)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        ind = img_id % len(cameras)

        for person_id, ann in enumerate(anns):
            bbox = np.array([ann["bbox"]], dtype=np.float32)

            data_samples = inference_topdown(model, image_bgr, bboxes=bbox, bbox_format="xywh")
            data_sample = merge_data_samples(data_samples)

            drawn = visualizer.add_datasample(
                name=f"{Path(file_name).stem}_p{person_id}",
                image=image_rgb,
                data_sample=data_sample,
                draw_gt=False,
                draw_pred=True,
                draw_heatmap=True,
                draw_bbox=args.draw_bbox,
                show=args.show,
                kpt_thr=args.kpt_thr,
                wait_time=1
            )

            if args.save:
                out_path = out_dir / f"{Path(file_name).stem}_p{person_id}.png"
                out_path_full = out_dir / f"full/{Path(file_name).stem}_p{person_id}.png"
                out_path_full.parent.mkdir(parents=True, exist_ok=True)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                h, w, c = drawn.shape
                cv2.imwrite(out_path_full.as_posix(), drawn[..., ::-1])
                cv2.imwrite(out_path.as_posix(), drawn[h//2:, :, ::-1])

            if not writers:
                h, w, c = drawn.shape
                writers = [cv2.VideoWriter(output_folder.joinpath(f'output_{cam}.avi').as_posix(), fourcc, 3, (w, h))
                           for cam in cameras]
            writers[ind].write(drawn[:,:,::-1])


if __name__ == "__main__":
    main()
