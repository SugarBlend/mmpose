import argparse
from dotenv import load_dotenv
import sys
from pathlib import Path
from pycocotools.coco import COCO
import json
import torch

from project.benchmarks.accuracy.surrogate_wrapper import AnnFormats

sys.path.insert(0, Path(__file__).parents[3].as_posix())

from config import EvalConfig, ModelConfig
from surrogate_wrapper import SurrogateEstimatorWrapper
from metrics_wrapper import BaseMetricConfigurator, correspondence
from project.label_studio.pipelines.pipeline import MMPipeline
from tools import logger, generate_radar_plot, save_metrics_xlsx, generate_radar_plot_html


class EvaluationOrchestrator(object):
    def __init__(
        self,
        pipeline: MMPipeline,
        evaluators: list[BaseMetricConfigurator],
        surrogate_wrapper: SurrogateEstimatorWrapper
    ) -> None:
        self.pipeline = pipeline
        self.evaluators: list[BaseMetricConfigurator] = evaluators
        self.surrogate_estimator: SurrogateEstimatorWrapper = surrogate_wrapper

        self.coco: COCO | None = None

    def load_annotations(self, ann_file: str, num_samples: int) -> None:
        self.coco = COCO(ann_file)
        self.coco.imgs = dict(list(self.coco.imgs.items())[: num_samples])  # restrict for testing samples
        self.coco.anno_file = [ann_file]

        for item in self.evaluators:
            item.coco = self.coco
        self.surrogate_estimator.coco = self.coco

    @torch.no_grad()
    def evaluate(
        self,
        config: ModelConfig,
        num_samples: int | None = None,
    ) -> dict[str, float]:
        self.load_annotations(config.ann_file, num_samples)
        self.surrogate_estimator.pipeline = self.pipeline
        self.surrogate_estimator.predict_converters = config.pred_converters
        results = self.surrogate_estimator(config.dataset_folder, config.ann_format)

        return {k: v for evaluator in self.evaluators for k, v in evaluator.calculate_results(results).items()}


def launch_evaluation(config: EvalConfig) -> dict[str, dict[str, float]] | None:
    exp_metrics: dict[str, dict[str, float]] = {}
    is_whole_body = False

    for desc in config.models:
        logger.info(f"Evaluating: {desc.legend}")

        if "sapiens2" in desc.config_path:
            from project.label_studio.pipelines.sapiens2 import Sapiens2
            pipeline = Sapiens2(desc.model_path, desc.config_path)
        else:
            pipeline = MMPipeline(desc.model_path, desc.config_path)

        metric_evaluators: [BaseMetricConfigurator] = []
        for name, params in config.metrics.__dict__.items():
            params.update({
                'gt_converter': desc.gt_converter,
                'ann_file': desc.ann_file
            })
            metric_evaluator = correspondence[name](metapath=desc.meta_file, params=params.copy())
            metric_evaluators.append(metric_evaluator)

        if not metric_evaluators:
            logger.warning("Empty configurations for metrics")
            return None

        surrogate_wrapper = SurrogateEstimatorWrapper()
        orchestrator = EvaluationOrchestrator(pipeline, metric_evaluators, surrogate_wrapper)
        metrics = orchestrator.evaluate(desc)
        exp_metrics[desc.legend] = dict(sorted(metrics.items()))

        is_gt_whole_body = desc.ann_format == AnnFormats.CocoWholeBody

        any_whole_body_out = any(converter["num_keypoints"] == 133 for converter in desc.pred_converters) if desc.pred_converters else False
        is_whole_body = any_whole_body_out or is_gt_whole_body
        logger.info(f"Results: {metrics}")

    save_dir = config.visualization.save_dir
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = save_dir.joinpath("metrics.json")
        metrics_path.write_text(json.dumps(exp_metrics, indent=2))
        logger.info(f"Metrics JSON saved: '{metrics_path}'")

        xlsx_path = save_dir.joinpath("metrics.xlsx")
        save_metrics_xlsx(exp_metrics, xlsx_path.as_posix(), is_wholebody=is_whole_body)

        plot_path = save_dir.joinpath("radar.png").as_posix()
        generate_radar_plot(exp_metrics, is_whole_body, config.visualization.radar_xticks, plot_path,
                            config.visualization.show_plot, title="Pose model comparison")

        generate_radar_plot_html(exp_metrics, is_whole_body,
                                 save_dir.joinpath("radar.html").as_posix(), title="Coco Wholebody dataset")

    return exp_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pose model evaluation with flexible keypoint subsets")
    parser.add_argument("--config", "-c", type=str, default="eval-config_wholebody.yaml",
                        help="Path to eval-config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    load_dotenv(Path(__file__).parent.joinpath("../../../tools/.env"))
    cfg = EvalConfig.load(parse_args().config)
    launch_evaluation(cfg)
