from deploy2serve.deployment.projects.sapiens.utils.palettes import (COCO_SKELETON_INFO, COCO_WHOLEBODY_SKELETON_INFO,
                                                                     GOLIATH_SKELETON_INFO, GOLIATH_KPTS_COLORS,
                                                                     GOLIATH_CLASSES, COCO_KPTS_COLORS,
                                                                     COCO_WHOLEBODY_KPTS_COLORS)
from project.services.ml_backend.utils import (_HANDS_KEYPOINTS, HALPE26_KEYPOINTS, COCO133_KEYPOINTS,
                                               COCO17_KEYPOINTS, HALPE136_KEYPOINTS, _FEETS_KEYPOINTS)
from project.label_studio.palettes import (COCO_HALPE26_KPTS_COLORS, COCO_HALPE26_SKELETON_INFO, HANDS21_SKELETON_INFO,
                                           HANDS21_KPTS_COLORS)
from dataclasses import dataclass, field

Color = tuple[int, int, int]  # BGR


@dataclass(frozen=True)
class SkeletonLink:
    id: int
    link: tuple[int, int]
    color: Color

    def __getitem__(self, item):
        return getattr(self, item)

@dataclass
class KeypointCategory:
    num_keypoints: int
    keypoints: list[str]
    kpt_colors: list[Color]
    skeleton: dict[int, SkeletonLink]  # id -> link
    name: str = "person"
    supercategory: str = "person"
    id: int = 1

    def to_coco(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "supercategory": self.supercategory,
            "num_keypoints": self.num_keypoints,
            "keypoints": self.keypoints,
        }

    def get_links(self) -> list[SkeletonLink]:
        return list(self.skeleton.values())


CATEGORY_HAND21 = KeypointCategory(
    num_keypoints=21,
    keypoints=_HANDS_KEYPOINTS,
    name="hand",
    kpt_colors=HANDS21_KPTS_COLORS,
    skeleton={
        k: SkeletonLink(id=v["id"], link=v["link"], color=v["color"])
        for k, v in HANDS21_SKELETON_INFO.items()
    },
)

CATEGORY_HALPE26 = KeypointCategory(
    num_keypoints=26,
    keypoints=HALPE26_KEYPOINTS,
    name="person",
    kpt_colors=COCO_HALPE26_KPTS_COLORS,
    skeleton={
        k: SkeletonLink(id=v["id"], link=v["link"], color=v["color"])
        for k, v in COCO_HALPE26_SKELETON_INFO.items()
    },
)

CATEGORY_COCO17 = KeypointCategory(
    num_keypoints=17,
    keypoints=COCO17_KEYPOINTS,
    name="person",
    kpt_colors=COCO_KPTS_COLORS,
    skeleton={
        k: SkeletonLink(id=v["id"], link=v["link"], color=v["color"])
        for k, v in COCO_SKELETON_INFO.items()
    },
)
CATEGORY_COCO133 = KeypointCategory(
    num_keypoints=133,
    keypoints=COCO133_KEYPOINTS,
    name="person",
    kpt_colors=COCO_WHOLEBODY_KPTS_COLORS,
    skeleton={
        k: SkeletonLink(id=v["id"], link=v["link"], color=v["color"])
        for k, v in COCO_WHOLEBODY_SKELETON_INFO.items()
    },
)

CATEGORY_GOLIATH308 = KeypointCategory(
    num_keypoints=344,
    keypoints=GOLIATH_CLASSES,
    name="person",
    kpt_colors=GOLIATH_KPTS_COLORS,
    skeleton={
        k: SkeletonLink(id=v["id"], link=v["link"], color=v["color"])
        for k, v in GOLIATH_SKELETON_INFO.items()
    },
)

structs = {
    17: CATEGORY_COCO17,
    21: CATEGORY_HAND21,
    26: CATEGORY_HALPE26,
    133: CATEGORY_COCO133,
    344: CATEGORY_GOLIATH308
}
