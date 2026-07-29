import numpy as np
from mmengine import Config

from mmpose.registry import METRICS
from mmpose.evaluation.metrics.coco_wholebody_metric import CocoWholeBodyMetric
from mmpose.datasets.datasets.utils import parse_pose_metainfo


_coco133_halpe136 = (
    [(i, i) for i in range(17)] +
    [(17, 20), (18, 22), (19, 24), (20, 21), (21, 23), (22, 25)] +
    [(23 + i, 26 + i) for i in range(68)] +
    [(91 + i, 94 + i) for i in range(21)] +
    [(112 + i, 115 + i) for i in range(21)]
)


@METRICS.register_module()
class Halpe136ToCocoWholeBodyMetric(CocoWholeBodyMetric):
    def __init__(self, *args,
                 coco_wholebody_meta_file='configs/_base_/datasets/coco_wholebody.py',
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._halpe_idx = [h for c, h in sorted(_coco133_halpe136)]
        cfg = Config.fromfile(coco_wholebody_meta_file)
        self._coco133_meta = parse_pose_metainfo(cfg['dataset_info'])

    def process(self, data_batch, data_samples):
        remapped = []
        for ds in data_samples:
            ds = ds.copy()
            pred = dict(ds['pred_instances'])
            kpts = np.asarray(pred['keypoints'])[:, self._halpe_idx, :]
            scores = np.asarray(pred['keypoint_scores'])[:, self._halpe_idx]
            pred['keypoints'] = kpts
            pred['keypoint_scores'] = scores
            ds['pred_instances'] = pred
            remapped.append(ds)

        self.dataset_meta = self._coco133_meta
        super().process(data_batch, remapped)
