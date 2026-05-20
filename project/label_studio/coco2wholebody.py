import json
from copy import deepcopy
from pathlib import Path

BODY_KPTS = 17
FOOT_KPTS = 6
FACE_KPTS = 68
HAND_KPTS = 21


def split_joints(joints: list[float]) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
    idx = 0

    def take(n: int) -> list[float]:
        nonlocal idx

        size = n * 3
        part = joints[idx: idx + size]

        if len(part) < size:
            part += [0.0] * (size - len(part))

        idx += size
        return part

    body = take(BODY_KPTS)
    foot = take(FOOT_KPTS)
    face = take(FACE_KPTS)
    left_hand = take(HAND_KPTS)
    right_hand = take(HAND_KPTS)

    return body, foot, face, left_hand, right_hand


def count_visible(joints: list[float]) -> int:
    cnt = 0

    for i in range(2, len(joints), 3):
        if joints[i] > 0:
            cnt += 1

    return cnt


def convert_to_wholebody(source_path: str, dest_path: str) -> None:
    source_path = Path(source_path)
    if not source_path.exists():
        raise FileExistsError

    with source_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    result = deepcopy(data)
    result["categories"] = [
        {
            "id": 1,
            "name": "person",
            "supercategory": "person",
            "keypoints": [
                "nose",
                "left_eye",
                "right_eye",
                "left_ear",
                "right_ear",
                "left_shoulder",
                "right_shoulder",
                "left_elbow",
                "right_elbow",
                "left_wrist",
                "right_wrist",
                "left_hip",
                "right_hip",
                "left_knee",
                "right_knee",
                "left_ankle",
                "right_ankle"
            ],
            "skeleton": [
                [16, 14], [14, 12], [17, 15], [15, 13],
                [12, 13], [6, 12], [7, 13], [6, 7],
                [6, 8], [7, 9], [8, 10], [9, 11],
                [2, 3], [1, 2], [1, 3], [2, 4],
                [3, 5], [4, 6], [5, 7]
            ]
        }
    ]

    for ann in result["annotations"]:
        joints = ann["keypoints"]

        body, foot, face, left_hand, right_hand = split_joints(joints)
        ann.update(
            {
                "keypoints": body,
                "foot_kpts": foot,
                "face_kpts": face,
                "lefthand_kpts": left_hand,
                "righthand_kpts": right_hand,
                "num_keypoints": count_visible(joints),
                "foot_valid": int(count_visible(foot) > 0),
                "face_valid": int(count_visible(face) > 0),
                "lefthand_valid": int(count_visible(left_hand) > 0),
                "righthand_valid": int(count_visible(right_hand) > 0)
            }
        )

    dest_path = Path(dest_path)
    with dest_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert from COCO style to COCO-Wholebody annotation")
    parser.add_argument("--source_ann", default="configs/config-wholebody.xml", help="Path to input coco-style annotation")
    parser.add_argument("--dest_ann", default="../annotations", help="Path to output coco-wholebody style annotation")
    args = parser.parse_args()

    convert_to_wholebody(args.source_ann, args.dest_ann)
