from __future__ import annotations


SKELETON_SUBSETS: dict[str, dict[str, list[int]]] = {
    "coco_wholebody": {
        "body": list(range(0, 17)),
        "foot": list(range(17, 23)),
        "face": list(range(23, 91)),
        "left_hand": list(range(91, 112)),
        "right_hand": list(range(112, 133)),
        "hands": list(range(91, 133)),
        "all": list(range(133)),
    },
    "halpe136": {
        "body": list(range(0, 26)),
        "face": list(range(26, 94)),
        "left_hand": list(range(94, 115)),
        "right_hand": list(range(115, 136)),
        "hands": list(range(94, 136)),
        "all": list(range(136)),
    },
    "halpe26": {
        "body": list(range(0, 26)),
        "all": list(range(26)),
    },
    "coco17": {
        "body": list(range(0, 17)),
        "all": list(range(17)),
    },
    "hand21": {
        "all": list(range(21)),
    },
}



coco_halpe26 = [(i, i) for i in range(17)] + [(17, 20), (18, 22), (19, 24), (20, 21), (21, 23), (22, 25)]

crowdpose_halpe26 = [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9), (5, 10), (6, 11), (7, 12), (8, 13), (9, 14), (10, 15),
                     (11, 16), (12, 17), (13, 18)]

halpe_halpe26 = [(i, i) for i in range(26)]

ochuman_halpe26 = [(i, i) for i in range(17)]
