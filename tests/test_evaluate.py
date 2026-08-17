"""真值解析与召回计算的单元测试。

全部是纯函数：不解码视频、不跑模型、不碰磁盘。真值解析是整个评估的入口，
解析错了后面每个数字都是错的，所以这里覆盖得比别处细。

运行方式（Windows PowerShell，项目根目录下）：
    D:\\Programs\\DevEnvironments\\Anaconda\\anaconda3\\envs\\myenv\\python.exe -m pytest tests/test_evaluate.py -v
"""

import pytest

from config import Config
from detectors import Detection
from evaluate import characters_per_shot, recall
from groundtruth import parse_ground_truth
from main import Track, passes_frontal

REAL_NAME = "魔女之旅_760s（人脸数期望：2+5+1+0+3+1）（跨镜头身份去重后期望：9）.mp4"


class TestParseGroundTruth:
    def test_real_filename(self):
        gt = parse_ground_truth("data/" + REAL_NAME)
        assert gt.title == "魔女之旅"
        assert gt.source_offset == 760
        assert gt.shot_faces == [2, 5, 1, 0, 3, 1]
        assert gt.num_shots == 6
        assert gt.total_faces == 12
        assert gt.num_characters == 9
        assert gt.style == "2d"

    def test_single_shot(self):
        gt = parse_ground_truth("爱情神话_1525s（人脸数期望：4）（跨镜头身份去重后期望：4）.mp4")
        assert gt.shot_faces == [4]
        assert gt.num_shots == 1
        assert gt.style == "real"

    def test_unknown_title_style(self):
        gt = parse_ground_truth("某新番_10s（人脸数期望：1）（跨镜头身份去重后期望：1）.mp4")
        assert gt.style == "unknown"

    def test_no_annotation_returns_none(self):
        assert parse_ground_truth("data/1.mp4") is None

    def test_half_annotation_is_an_error(self):
        # 写了一半的标注是笔误，不能当成"无标注"静默跳过。
        with pytest.raises(ValueError):
            parse_ground_truth("片子_1s（人脸数期望：1+2）.mp4")


class TestRecall:
    def test_partial(self):
        assert recall(3, 9) == pytest.approx(1 / 3)

    def test_exact(self):
        assert recall(9, 9) == 1.0

    def test_over_detection_is_capped(self):
        # 检出多于真值不算超额完成；上限就是 1.0，过检要看 ratio 列。
        assert recall(20, 9) == 1.0

    def test_zero_ground_truth(self):
        assert recall(0, 0) == 1.0


def make_track(track_id, start_time, character_id):
    return Track(
        track_id=track_id,
        label="anime_face",
        start_time=start_time,
        end_time=start_time + 0.5,
        detections=[],
        character_id=character_id,
    )


class TestCharactersPerShot:
    def test_splits_on_cuts(self):
        cuts = [10.0, 20.0]
        tracks = [
            make_track(1, 1.0, 0), make_track(2, 2.0, 1),
            make_track(3, 12.0, 0),
            make_track(4, 25.0, 2), make_track(5, 26.0, 3),
        ]
        assert characters_per_shot(tracks, cuts) == [2, 1, 2]

    def test_same_character_twice_in_one_shot_counts_once(self):
        tracks = [make_track(1, 1.0, 0), make_track(2, 3.0, 0)]
        assert characters_per_shot(tracks, []) == [1]

    def test_unidentified_tracks_are_ignored(self):
        # character_id=None 不参与计数，与选段阶段口径一致。
        tracks = [make_track(1, 1.0, None), make_track(2, 2.0, 0)]
        assert characters_per_shot(tracks, []) == [1]

    def test_no_cuts_gives_one_shot(self):
        assert characters_per_shot([], []) == [0]

    def test_track_starting_on_a_cut_belongs_to_the_new_shot(self):
        # bisect_right：start_time == 切镜时刻的轨迹归到切镜之后那个镜头。
        # 这个方向是对的——轨迹不会跨切镜，恰好起于切镜说明它属于新镜头。
        assert characters_per_shot([make_track(1, 10.0, 0)], [10.0]) == [0, 1]


class TestPassesFrontal:
    def make_det(self, num_eyes):
        return Detection(
            frame_index=0, time=0.0, bbox=(0, 0, 10, 10),
            confidence=0.9, label="anime_face", num_eyes=num_eyes,
        )

    def test_two_eyes_passes(self):
        assert passes_frontal(self.make_det(2), Config(require_eyes=2))

    def test_one_eye_fails(self):
        assert not passes_frontal(self.make_det(1), Config(require_eyes=2))

    def test_disabled_lets_everything_through(self):
        # require_eyes=0 时不跑推理，num_eyes 保持 None 也要放行。
        assert passes_frontal(self.make_det(None), Config(require_eyes=0))
