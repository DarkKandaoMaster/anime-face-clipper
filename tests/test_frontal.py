"""正脸评分的纯逻辑单测（合成数据，不需要 GPU / 模型 / 网络）。"""

import numpy as np
import pytest

from config import Config, set_frontal_weight
from detectors import Detection
from frontal import frontal_from_eyes, frontal_from_landmarks
from main import FaceTracker

# ArcFace 的标准 112×112 五点模板就是「正脸」的定义本身。
FRONTAL_LANDMARKS = [
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041],
]


class TestFrontalFromLandmarks:
    def test_arcface_template_is_frontal(self):
        assert frontal_from_landmarks(FRONTAL_LANDMARKS) > 0.95

    def test_profile_scores_low(self):
        # 鼻尖与嘴心都滑到近侧那只眼上 = 强侧脸。
        profile = [[38.3, 51.7], [55.0, 51.5], [55.0, 71.7], [53.0, 92.4], [58.0, 92.2]]
        assert frontal_from_landmarks(profile) < 0.2

    def test_rotation_invariant(self):
        """画面内旋转（人歪着头）不该改变正脸分——判据是偏航，不是滚转。"""
        points = np.array(FRONTAL_LANDMARKS)
        angle = np.deg2rad(35)
        rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        rotated = (points - points.mean(0)) @ rot.T + points.mean(0)
        assert frontal_from_landmarks(rotated) == pytest.approx(
            frontal_from_landmarks(points), abs=0.02
        )

    def test_scale_invariant(self):
        scaled = (np.array(FRONTAL_LANDMARKS) * 4.0).tolist()
        assert frontal_from_landmarks(scaled) == pytest.approx(
            frontal_from_landmarks(FRONTAL_LANDMARKS), abs=1e-6
        )

    def test_missing_landmarks_is_neutral(self):
        # 未知不惩罚：给不出关键点的检测器不该被排到所有人后面。
        assert frontal_from_landmarks(None) == 1.0
        assert frontal_from_landmarks([[0.0, 0.0]]) == 1.0

    def test_degenerate_eyes(self):
        collapsed = [[50.0, 50.0], [50.0, 50.0], [56.0, 71.7], [41.5, 92.4], [70.7, 92.2]]
        assert frontal_from_landmarks(collapsed) == 0.0


def _eye(cx, score=0.9):
    return ((cx - 6, 40, cx + 6, 54), "eye", score)


class TestFrontalFromEyes:
    def test_two_symmetric_eyes_is_frontal(self):
        assert frontal_from_eyes([_eye(28), _eye(72)], 100) > 0.95

    def test_narrow_span_scores_lower(self):
        """两眼间距被投影压缩 = 侧过头。"""
        wide = frontal_from_eyes([_eye(28), _eye(72)], 100)
        narrow = frontal_from_eyes([_eye(46), _eye(62)], 100)
        assert narrow < wide / 2

    def test_offset_pair_scores_lower(self):
        """两眼一起滑到一侧 = 整张脸转过去了。"""
        centered = frontal_from_eyes([_eye(30), _eye(70)], 100)
        shifted = frontal_from_eyes([_eye(60), _eye(100)], 100)
        assert shifted < centered

    def test_one_eye_beats_none(self):
        assert 0 < frontal_from_eyes([], 100) < frontal_from_eyes([_eye(30)], 100) < 0.5

    def test_uses_two_highest_confidence(self):
        # 低分的第三个框（误检）不该把间距算歪。
        pair = [_eye(28), _eye(72)]
        assert frontal_from_eyes(pair + [_eye(50, score=0.1)], 100) == pytest.approx(
            frontal_from_eyes(pair, 100)
        )

    def test_zero_width_crop(self):
        assert frontal_from_eyes([_eye(30), _eye(70)], 0) > 0


def _det(index, blur, frontal=None):
    return Detection(
        frame_index=index, time=index * 0.3, bbox=(0, 0, 10, 10),
        confidence=1.0, label="anime_face", blur_var=blur, frontal=frontal,
        crop=np.zeros((4, 4, 3), dtype=np.uint8),
    )


class TestFrontalWeightInTournament:
    """权重只改「送哪张脸去聚类」，不改轨迹的存亡。"""

    def _represent(self, weight, dets):
        config = Config(frontal_weight=weight, crops_per_track=1)
        tracker = FaceTracker(config)
        for i, det in enumerate(dets):
            tracker.update(i, False, [det])
        tracks = tracker.finish()
        assert len(tracks) == 1  # 同一位置的框会串成一条轨迹
        return tracks[0].representative_frame

    def test_weight_zero_picks_sharpest(self):
        # 关掉权重时行为必须和加这个功能之前逐字节一致。
        assert self._represent(0.0, [_det(0, 100.0, 0.1), _det(1, 90.0, 1.0)]) == 0

    def test_weight_one_prefers_frontal(self):
        assert self._represent(1.0, [_det(0, 100.0, 0.1), _det(1, 90.0, 1.0)]) == 1

    def test_frontal_cannot_beat_a_much_sharper_face(self):
        # 加权不是「只看正脸」：糊得多的正脸仍然输给清晰的准正脸。
        assert self._represent(1.0, [_det(0, 500.0, 0.6), _det(1, 50.0, 1.0)]) == 0

    def test_none_frontal_is_ignored(self):
        assert self._represent(1.0, [_det(0, 100.0, None), _det(1, 90.0, None)]) == 0


class TestSetFrontalWeight:
    def test_overrides_every_style_profile(self):
        """只改全局字段会被画风路由的 profile 静默盖掉，所以必须两边一起改。"""
        config = Config()
        set_frontal_weight(config, 0.7)
        assert config.frontal_weight == 0.7
        assert all(p.frontal_weight == 0.7 for p in config.style_profiles.values())

    def test_keeps_other_profile_fields(self):
        config = Config()
        before = config.style_profiles["real"].min_face_height_ratio
        set_frontal_weight(config, 0.5)
        assert config.style_profiles["real"].min_face_height_ratio == before


class TestCropTimeSpread:
    """`crop_min_gap_frames`：让 K 张代表图来自不同时间，而不是相邻几帧。"""

    def _kept(self, gap, blur_by_frame, last=2):
        """喂一条连续轨迹（帧号必须连续，否则跟踪器会断轨），返回留下几张代表图。

        只有 blur_by_frame 里列出的帧带裁剪图，其余帧不参与擂台。
        """
        config = Config(crops_per_track=3, crop_min_gap_frames=gap)
        tracker = FaceTracker(config)
        for i in range(last + 1):
            det = _det(i, blur_by_frame.get(i, 1.0))
            if i not in blur_by_frame:
                det.crop = None
            tracker.update(i, False, [det])
        tracks = tracker.finish()
        assert len(tracks) == 1
        return len(tracks[0].representative_images)

    def test_off_keeps_adjacent_frames(self):
        # 默认 0：前三名可以全来自相邻帧（这正是 K=3 实测无收益的原因）。
        assert self._kept(0, {0: 100.0, 1: 99.0, 2: 98.0}) == 3

    def test_gap_collapses_a_burst_into_one(self):
        assert self._kept(10, {0: 100.0, 1: 99.0, 2: 98.0}) == 1

    def test_gap_admits_temporally_separated_faces(self):
        assert self._kept(10, {0: 100.0, 1: 99.0, 20: 98.0}, last=20) == 2

    def test_better_face_wins_within_a_burst(self):
        config = Config(crops_per_track=3, crop_min_gap_frames=10)
        tracker = FaceTracker(config)
        for det in [_det(0, 50.0), _det(1, 200.0)]:
            tracker.update(det.frame_index, False, [det])
        track = tracker.finish()[0]
        assert len(track.representative_images) == 1
        assert track.representative_frame == 1  # 同一段时间里留下更清晰的那张
