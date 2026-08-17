"""从素材文件名里解析人工标注的真值。

公开数据集不可用、自建数据集成本过高，所以本项目的真值只有一个来源：
`data/` 下每段素材的文件名。格式固定为

    魔女之旅_760s（人脸数期望：2+5+1+0+3+1）（跨镜头身份去重后期望：9）.mp4
    ↑标题   ↑源片位置  ↑逐镜头主体数（段数 = 镜头数）      ↑全片不同角色数

标注口径：正脸、出镜 >1s、表情清晰、人脸框宽度占画面宽度达标。标注时存在
大量模棱两可的脸，真值本身带噪声——评估结果要按"数量级对不对"来读，
不要当成小数点后两位可信的测量。

本模块只做解析，不做 I/O、不碰视频。
"""

import dataclasses
import os
import re
from typing import List, Optional

# 逐镜头主体数，形如 "（人脸数期望：2+5+1+0+3+1）"
_SHOT_FACES_RE = re.compile(r"人脸数期望[：:]\s*([\d+\s]+?)\s*[）)]")
# 跨镜头去重后的角色总数，形如 "（跨镜头身份去重后期望：9）"
_CHARACTERS_RE = re.compile(r"跨镜头身份去重后期望[：:]\s*(\d+)")
# 标题与源片位置，形如 "魔女之旅_760s"
_TITLE_RE = re.compile(r"^(.+?)_(\d+)s")

# 画风真值。用于验证自动风格路由判得对不对，不参与流水线本身。
# 键是标题（文件名首个下划线之前的部分）。
STYLE_BY_TITLE = {
    "爱情神话": "real",
    "凡人修仙传": "3d",
    "JOJO的奇妙冒险": "2d",
    "擅长捉弄的高木同学": "2d",
    "新世纪福音战士": "2d",
    "魔女之旅": "2d",
}


@dataclasses.dataclass(frozen=True)
class GroundTruth:
    """一段素材的人工标注。

    属性：
        stem: 不含扩展名的文件名（也是输出目录名）。
        title: 作品名。
        source_offset: 该片段在源片中的起始秒数（仅作溯源）。
        shot_faces: 逐镜头的主体数，顺序与片中镜头顺序一致。
        num_characters: 该 30 秒内不同角色总数（跨镜头去重后）。
        style: 画风真值 real / 2d / 3d。
    """

    stem: str
    title: str
    source_offset: int
    shot_faces: List[int]
    num_characters: int
    style: str

    @property
    def num_shots(self) -> int:
        """真值镜头数 = 逐镜头主体数的段数。"""
        return len(self.shot_faces)

    @property
    def total_faces(self) -> int:
        """所有镜头的主体数之和（同一角色跨镜头会被重复计入）。"""
        return sum(self.shot_faces)


def parse_ground_truth(video_path: str) -> Optional[GroundTruth]:
    """从文件名解析真值；文件名不含标注时返回 None。

    fail fast 的例外：无标注素材是合法输入（用户可以拿任意视频跑流水线），
    只是不能参与召回评估，所以这里返回 None 而不是抛异常。但标注**写了却
    写坏了**（两段标注只有一段）属于笔误，会直接抛出来。
    """
    stem = os.path.splitext(os.path.basename(video_path))[0]
    faces_match = _SHOT_FACES_RE.search(stem)
    chars_match = _CHARACTERS_RE.search(stem)
    if not faces_match and not chars_match:
        return None
    if not faces_match or not chars_match:
        raise ValueError(
            f"文件名标注不完整（需要同时有「人脸数期望」和「跨镜头身份去重后期望」）：{stem}"
        )

    shot_faces = [int(v) for v in faces_match.group(1).split("+")]
    title_match = _TITLE_RE.match(stem)
    title = title_match.group(1) if title_match else stem
    source_offset = int(title_match.group(2)) if title_match else 0

    return GroundTruth(
        stem=stem,
        title=title,
        source_offset=source_offset,
        shot_faces=shot_faces,
        num_characters=int(chars_match.group(1)),
        style=STYLE_BY_TITLE.get(title, "unknown"),
    )
