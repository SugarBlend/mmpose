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



coco133_halpe26 = [(i, i) for i in range(17)] + [(17, 20), (18, 22), (19, 24), (20, 21), (21, 23), (22, 25)]

crowdpose_halpe26 = [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9), (5, 10), (6, 11), (7, 12), (8, 13), (9, 14), (10, 15),
                     (11, 16), (12, 17), (13, 18)]

# source halpe - union (26 and 136 points)
halpe_halpe26 = [(i, i) for i in range(26)]

ochuman_halpe26 = [(i, i) for i in range(17)]

goliath308_halpe26 = [
    # Halpe26: 0-16 COCO body, 17-22 foot, 23-25 extra (head_top, neck, hip/pelvis)
    (0,  0),   # Nose
    (1,  1),   # LEye
    (2,  2),   # REye
    (3,  3),   # LEar
    (4,  4),   # REar
    (5,  5),   # LShoulder
    (6,  6),   # RShoulder
    (7,  7),   # LElbow
    (8,  8),   # RElbow
    (62, 9),   # LWrist (goliath: 62)
    (41, 10),  # RWrist (goliath: 41)
    (9,  11),  # LHip
    (10, 12),  # RHip
    (11, 13),  # LKnee
    (12, 14),  # RKnee
    (13, 15),  # LAnkle
    (14, 16),  # RAnkle
    # extra (17-19)
    (70, 17),  # Head -> center_of_glabella (goliath: 70)
    (69, 18),  # Neck (goliath: 69)
    # Hip/pelvis is the average of left+right hip. There is no direct analog; we use goliath, which doesn't have a
    # pelvis. We'll leave it as 0.0 or skip it for evaluation; we'll just take the average: but KeypointConverter
    # doesn't support the average, so we'll map it to left_hip as an approximation.
    (9,  19),  # Hip (zooming in via left_hip)
    # foot (20-25)
    (15, 20),  # LBigToe
    (18, 21),  # RBigToe
    (16, 22),  # LSmallToe
    (19, 23),  # RSmallToe
    (17, 24),  # LHeel
    (20, 25),  # RHeel
]

halpe2coco_wholebody: dict[str, tuple[list[int], list[int]]] = {
    "body": (list(range(17)), list(range(17))),
    "foot": ([20, 22, 24, 21, 23, 25], [17, 18, 19, 20, 21, 22]),
    "left_hand": (list(range(94, 115)), list(range(91, 112))),
    "right_hand": (list(range(115, 136)), list(range(112, 133))),
}
coco2coco_wholebody: dict[str, tuple[list[int], list[int]]] = {
    "body": (list(range(17)), list(range(17))),
}