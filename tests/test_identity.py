"""身份识别侧的纯逻辑单测：裁剪几何、人脸对齐、画风 profile、多图聚合。

不需要 GPU / 模型 / 网络：真人脸对齐只用到闭式几何解，身份特征用测试内
临时注册的假 embedder 顶掉。

    & $py -m pytest tests/test_identity.py -q
"""

import numpy as np
import pytest

from config import Config, StyleProfile
from detectors import Detection, Detector, crop_bbox, expand_bbox
from detectors import _ARCFACE_TEMPLATE, _umeyama
from embedders import Embedder, register
from main import Track, compute_differences
from style import apply_style


# === 裁剪几何 ===

class TestCropGeometry:
    def test_expand_bbox_scales_by_own_size(self):
        # 100×50 的框、margin=0.2 -> 左右各扩 20、上下各扩 10。
        assert expand_bbox((100, 100, 200, 150), 0.2) == (80, 90, 220, 160)

    def test_expand_bbox_zero_margin_is_identity(self):
        assert expand_bbox((10, 20, 30, 40), 0.0) == (10, 20, 30, 40)

    def test_crop_bbox_clamps_to_frame(self):
        # 外扩后越界的框要裁回画面内，而不是返回空。
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        crop = crop_bbox(frame, expand_bbox((0, 0, 20, 20), 1.0))
        assert crop.shape[:2] == (40, 40)  # (-20,-20,40,40) -> (0,0,40,40)

    def test_crop_bbox_rejects_degenerate(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        assert crop_bbox(frame, (50, 50, 50, 60)) is None
        assert crop_bbox(frame, (200, 200, 300, 300)) is None

    def test_crop_is_a_copy_not_a_view(self):
        # 切片视图会拖住整帧不放，流式处理的内存优势全靠这一条。
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        crop = crop_bbox(frame, (10, 10, 20, 20))
        crop[0, 0] = 255
        assert frame[10, 10, 0] == 0

    def test_default_make_crop_uses_config_margin(self):
        class Dummy(Detector):
            def detect(self, image, frame_index, time):
                return []

        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        det = Detection(0, 0.0, (50, 50, 100, 100), 0.9, "x")
        assert Dummy(Config(crop_margin=0.0)).make_crop(frame, det).shape[:2] == (50, 50)
        assert Dummy(Config(crop_margin=0.5)).make_crop(frame, det).shape[:2] == (100, 100)


# === 人脸对齐（ArcFace 五点仿射）===

class TestUmeyama:
    def test_recovers_known_similarity(self):
        # 把模板旋转 30°、放大 2 倍、平移，再求解，应当解回同一个变换。
        angle, scale = np.deg2rad(30.0), 2.0
        rotation = np.array([[np.cos(angle), -np.sin(angle)],
                             [np.sin(angle), np.cos(angle)]])
        shift = np.array([17.0, -9.0])
        src = _ARCFACE_TEMPLATE @ (scale * rotation).T + shift

        matrix = _umeyama(src, _ARCFACE_TEMPLATE)
        recovered = src @ matrix[:, :2].T + matrix[:, 2]
        assert np.allclose(recovered, _ARCFACE_TEMPLATE, atol=1e-6)

    def test_identity_when_already_aligned(self):
        matrix = _umeyama(_ARCFACE_TEMPLATE, _ARCFACE_TEMPLATE)
        assert np.allclose(matrix[:, :2], np.eye(2), atol=1e-9)
        assert np.allclose(matrix[:, 2], 0.0, atol=1e-9)


# === 画风 profile ===

class TestApplyStyle:
    def test_swaps_the_whole_set(self):
        config = Config(style_profiles={
            "real": StyleProfile("real_face_scrfd", "arcface", 1.05, 0.0, 0.07),
        })
        routed = apply_style(config, "real")
        assert (routed.detector, routed.embedder) == ("real_face_scrfd", "arcface")
        assert routed.identity_threshold == 1.05
        assert routed.crop_margin == 0.0
        assert routed.min_face_height_ratio == 0.07

    def test_unknown_style_leaves_config_untouched(self):
        config = Config()
        assert apply_style(config, "水墨画") is config

    def test_does_not_mutate_the_original(self):
        config = Config(identity_threshold=0.178, embedder="ccip")
        apply_style(config, "real")
        assert config.identity_threshold == 0.178
        assert config.embedder == "ccip"


# === 多图聚合 ===

@register("_test_pixel_mean")
class _PixelMeanEmbedder(Embedder):
    """假 embedder：差异 = 两张图平均像素值之差的绝对值 / 255。

    只为了让 compute_differences 的聚合逻辑可测，不碰任何模型。
    """

    def differences(self, crop_paths):
        import cv2

        values = np.array([
            float(cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR).mean())
            for p in crop_paths
        ])
        return np.abs(values[:, None] - values[None, :]) / 255.0

    def default_threshold(self):
        return 0.5


def _write_track(tmp_path, track_id, values):
    """造一条轨迹：每个灰度值一张纯色小图，写进 <tmp>/crops/。"""
    import cv2

    crops_dir = tmp_path / "crops"
    crops_dir.mkdir(exist_ok=True)
    rels = []
    for rank, value in enumerate(values):
        name = f"track_{track_id}_{rank}.jpg"
        image = np.full((16, 16, 3), value, dtype=np.uint8)
        cv2.imencode(".jpg", image)[1].tofile(str(crops_dir / name))
        rels.append(f"crops/{name}")
    return Track(
        track_id=track_id, label="x", start_time=0.0, end_time=1.0, detections=[],
        representative_crops=rels, representative_crop=rels[0],
    )


class TestComputeDifferences:
    @pytest.fixture
    def config(self):
        return Config(embedder="_test_pixel_mean")

    def test_single_crop_per_track_passes_through(self, tmp_path, config):
        tracks = [_write_track(tmp_path, 1, [0]), _write_track(tmp_path, 2, [255])]
        candidates, diff = compute_differences(tracks, str(tmp_path), config)
        assert len(candidates) == 2
        assert diff.shape == (2, 2)
        assert diff[0, 1] == pytest.approx(1.0, abs=0.02)

    def test_multi_crop_takes_the_median_not_the_extreme(self, tmp_path, config):
        # A = {0, 0, 200}，B = {200}。最小差异是 0（A 的第三张正好撞上 B），
        # 中位数是 200/255——聚合必须取后者，否则一张偶然相似的脸就能粘合两条轨迹。
        tracks = [_write_track(tmp_path, 1, [0, 0, 200]), _write_track(tmp_path, 2, [200])]
        _candidates, diff = compute_differences(tracks, str(tmp_path), config)
        assert diff[0, 1] == pytest.approx(200 / 255, abs=0.02)

    def test_diagonal_is_zero_and_matrix_symmetric(self, tmp_path, config):
        tracks = [_write_track(tmp_path, i, [i * 40, i * 40 + 10]) for i in range(1, 4)]
        _candidates, diff = compute_differences(tracks, str(tmp_path), config)
        assert np.allclose(np.diag(diff), 0.0)
        assert np.allclose(diff, diff.T)

    def test_tracks_without_crops_are_dropped(self, tmp_path, config):
        with_crop = _write_track(tmp_path, 1, [0])
        without = Track(track_id=2, label="x", start_time=0.0, end_time=1.0, detections=[])
        candidates, diff = compute_differences([with_crop, without], str(tmp_path), config)
        assert [t.track_id for t in candidates] == [1]
        assert diff.shape == (1, 1)

    def test_no_candidates_returns_none(self, tmp_path, config):
        without = Track(track_id=1, label="x", start_time=0.0, end_time=1.0, detections=[])
        candidates, diff = compute_differences([without], str(tmp_path), config)
        assert candidates == []
        assert diff is None

    def test_missing_file_is_skipped(self, tmp_path, config):
        track = _write_track(tmp_path, 1, [0])
        track.representative_crops.append("crops/track_1_missing.jpg")
        _candidates, diff = compute_differences([track], str(tmp_path), config)
        assert diff.shape == (1, 1)
