"""把一次运行的结果拼成接触印相表，用肉眼判断身份聚类对不对。

干什么：读 <out_dir>/tracks.json，按 character_id 分组，每个簇一行，
行内是该簇若干条轨迹的代表裁剪图。同一行长得不像 = 合过头；
不同行是同一个角色 = 拆过头。没有真值的素材（完整片源）只能这么看。

怎么跑（项目根目录）：

    $py backend/core/montage.py output_raw/魔女之旅
    $py backend/core/montage.py output_raw/魔女之旅 --clusters 30 --per-cluster 10

产出：<out_dir>/montage.jpg（每次重跑覆盖，幂等）。
"""

import argparse
import collections
import json
import os
from typing import List, Optional

import cv2
import numpy as np

from embedders import imread_unicode

CELL = 96          # 每格边长（像素）
LABEL_W = 64       # 行首标签栏宽度


def build_montage(out_dir: str, max_clusters: int, per_cluster: int,
                  tracks_name: str = "tracks.json") -> Optional[np.ndarray]:
    """按簇大小降序取前 max_clusters 个簇，每簇最多 per_cluster 张图，拼成一张图。"""
    with open(os.path.join(out_dir, tracks_name), encoding="utf-8") as f:
        tracks = json.load(f)

    groups = collections.defaultdict(list)
    for track in tracks:
        if track["character_id"] is not None and track["representative_crops"]:
            groups[track["character_id"]].append(track)

    ranked = sorted(groups.items(), key=lambda kv: -len(kv[1]))[:max_clusters]
    if not ranked:
        return None

    rows = []
    for cid, members in ranked:
        # 簇内按轨迹时间均匀取样，而不是取前 K 条——前 K 条往往来自同一场戏，
        # 看不出这个簇在全片尺度上有没有混进别的角色。
        members.sort(key=lambda t: t["start_time"])
        step = max(1, len(members) / per_cluster)
        picked = [members[min(len(members) - 1, int(i * step))] for i in range(per_cluster)]

        row = np.full((CELL, LABEL_W + per_cluster * CELL, 3), 40, dtype=np.uint8)
        cv2.putText(row, f"#{cid}", (4, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(row, f"n={len(members)}", (4, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 220, 160), 1)
        for i, track in enumerate(picked):
            path = os.path.join(out_dir, track["representative_crops"][0])
            if not os.path.isfile(path):
                continue
            cell = cv2.resize(imread_unicode(path), (CELL, CELL))
            row[:, LABEL_W + i * CELL:LABEL_W + (i + 1) * CELL] = cell
        rows.append(row)
        rows.append(np.full((2, rows[0].shape[1], 3), 0, dtype=np.uint8))  # 行间分隔线
    return np.vstack(rows)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="按 character_id 拼接代表裁剪图，肉眼核对聚类。")
    parser.add_argument("out_dir", help="某个视频的输出目录（含 tracks.json 与 crops/）。")
    parser.add_argument("--clusters", type=int, default=24, help="最多展示几个簇（按簇内轨迹数降序）。")
    parser.add_argument("--per-cluster", type=int, default=8, help="每个簇展示几张代表图。")
    parser.add_argument("--tracks", default="tracks.json", help="轨迹文件名（换聚类参数重跑时可指向另一份）。")
    parser.add_argument("--out", help="输出图片路径，默认 <out_dir>/montage.jpg。")
    args = parser.parse_args(argv)

    canvas = build_montage(args.out_dir, args.clusters, args.per_cluster, args.tracks)
    if canvas is None:
        print("没有任何带 character_id 的轨迹。")
        return 1
    path = args.out or os.path.join(args.out_dir, "montage.jpg")
    cv2.imencode(".jpg", canvas)[1].tofile(path)  # 中文路径下 cv2.imwrite 会静默失败
    print(f"写入 {path}（{canvas.shape[1]}x{canvas.shape[0]}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
