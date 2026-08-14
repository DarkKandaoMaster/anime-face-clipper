r"""流式改造相关的单元测试。

覆盖 iter_frames（解码采样与时间戳）、crop_bbox / laplacian_variance
（当帧裁剪与清晰度）、imwrite_unicode（中文路径写图）、以及 FaceTracker
在线挑选代表裁剪图的行为。跟踪的连接/断轨语义由 test_main.py 经
track_faces 覆盖，此处只测批处理版没有的那部分——代表裁剪图。

测试视频由 cv2.VideoWriter 现场合成，不依赖 data/ 下的素材。

运行方式（Windows PowerShell，项目根目录下）：
    D:\Programs\DevEnvironments\Anaconda\anaconda3\envs\myenv\python.exe -m pytest tests -v
"""

import os

import cv2
import numpy as np
import pytest

from config import Config
from detectors import Detection
from main import (
    FaceTracker,
    crop_bbox,
    imwrite_unicode,
    iter_frames,
    laplacian_variance,
    to_pil,
)


FPS = 10.0
NUM_FRAMES = 50  # 5 秒


@pytest.fixture
def video(tmp_path):
    """合成一段 5 秒 / 10fps 的视频，每帧亮度递增以便区分。"""
    path = str(tmp_path / "clip.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (64, 48))
    assert writer.isOpened(), "无法创建测试视频，缺少 mp4v 编码器？"
    for i in range(NUM_FRAMES):
        writer.write(np.full((48, 64, 3), i * 5 % 256, dtype=np.uint8))
    writer.release()
    return path


def make_det(frame_index, time, bbox, confidence=0.9, blur_var=0.0, crop=None):
    return Detection(
        frame_index=frame_index,
        time=time,
        bbox=bbox,
        confidence=confidence,
        label="anime_face",
        blur_var=blur_var,
        crop=crop,
    )


# === iter_frames ===

class TestIterFrames:
    def test_samples_at_interval(self, video):
        # 5 秒视频、0.5 秒间隔 -> 时间戳 0.0, 0.5, ..., 4.5 共 10 帧。
        frames = list(iter_frames(Config(frame_interval=0.5), video))
        assert len(frames) == 10
        times = [t for _i, t, _f in frames]
        assert times == pytest.approx([i * 0.5 for i in range(10)], abs=1 / FPS)

    def test_sample_index_is_contiguous(self, video):
        frames = list(iter_frames(Config(frame_interval=0.5), video))
        assert [i for i, _t, _f in frames] == list(range(len(frames)))

    def test_yields_decoded_bgr_frames(self, video):
        _index, _time, frame = next(iter(iter_frames(Config(frame_interval=1.0), video)))
        assert frame.shape == (48, 64, 3)
        assert frame.dtype == np.uint8

    def test_interval_smaller_than_frame_period_yields_every_frame(self, video):
        # 间隔比帧周期还小时，不应重复产出同一帧，上限是视频总帧数。
        frames = list(iter_frames(Config(frame_interval=0.01), video))
        assert len(frames) == NUM_FRAMES

    def test_limit_seconds_stops_early(self, video):
        frames = list(iter_frames(Config(frame_interval=0.5), video, limit_seconds=2.0))
        assert [t for _i, t, _f in frames] == pytest.approx([0.0, 0.5, 1.0, 1.5], abs=1 / FPS)

    def test_missing_file_fails_loudly(self, tmp_path):
        # fail fast：打不开就抛，不静默产出空序列。
        with pytest.raises(RuntimeError):
            list(iter_frames(Config(), str(tmp_path / "nope.mp4")))

    def test_unicode_path(self, tmp_path):
        cn_dir = tmp_path / "测试目录"
        cn_dir.mkdir()
        path = str(cn_dir / "片段.mp4")
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (64, 48))
        for i in range(20):
            writer.write(np.full((48, 64, 3), i * 5, dtype=np.uint8))
        writer.release()
        assert len(list(iter_frames(Config(frame_interval=0.5), path))) == 4


# === crop_bbox / laplacian_variance / to_pil ===

class TestCropBbox:
    @pytest.fixture
    def image(self):
        return np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)

    def test_basic_crop(self, image):
        assert crop_bbox(image, (10, 5, 30, 25)).shape == (20, 20, 3)

    def test_clamps_to_frame_bounds(self, image):
        # 框越界时裁回画面内，而不是报错或返回空。
        assert crop_bbox(image, (-10, -10, 1000, 1000)).shape == (48, 64, 3)

    def test_degenerate_box_returns_none(self, image):
        assert crop_bbox(image, (5, 5, 5, 5)) is None
        assert crop_bbox(image, (30, 30, 10, 10)) is None

    def test_fully_outside_returns_none(self, image):
        assert crop_bbox(image, (100, 100, 200, 200)) is None

    def test_crop_is_a_copy_not_a_view(self, image):
        # 关键：切片视图会拖住整帧不放，流式的内存优势就没了。
        crop = crop_bbox(image, (0, 0, 10, 10))
        assert crop.base is None
        crop[0, 0, 0] = 255
        assert image[0, 0, 0] != 255


class TestLaplacianVariance:
    def test_none_crop_is_zero(self):
        assert laplacian_variance(None) == 0.0

    def test_empty_crop_is_zero(self):
        assert laplacian_variance(np.zeros((0, 0, 3), dtype=np.uint8)) == 0.0

    def test_flat_image_is_zero(self):
        assert laplacian_variance(np.full((20, 20, 3), 128, dtype=np.uint8)) == 0.0

    def test_sharp_edges_score_higher_than_blurred(self):
        sharp = np.zeros((40, 40, 3), dtype=np.uint8)
        sharp[:, 20:] = 255
        blurred = cv2.GaussianBlur(sharp, (9, 9), 5)
        assert laplacian_variance(sharp) > laplacian_variance(blurred)


class TestToPil:
    def test_converts_bgr_to_rgb(self):
        bgr = np.zeros((4, 4, 3), dtype=np.uint8)
        bgr[:, :, 0] = 255  # 纯蓝（BGR）
        assert to_pil(bgr).getpixel((0, 0)) == (0, 0, 255)  # 纯蓝（RGB）


# === imwrite_unicode ===

class TestImwriteUnicode:
    def test_writes_ascii_path(self, tmp_path):
        path = str(tmp_path / "a.jpg")
        assert imwrite_unicode(path, np.zeros((10, 10, 3), dtype=np.uint8))
        assert os.path.getsize(path) > 0

    def test_writes_unicode_path(self, tmp_path):
        # cv2.imwrite 在这条路径上会静默返回 False，这正是本函数存在的理由。
        cn_dir = tmp_path / "李超超-人生路不熟"
        cn_dir.mkdir()
        path = str(cn_dir / "轨迹.jpg")
        assert imwrite_unicode(path, np.zeros((10, 10, 3), dtype=np.uint8))
        assert os.path.getsize(path) > 0

    def test_written_image_reads_back(self, tmp_path):
        cn_dir = tmp_path / "中文"
        cn_dir.mkdir()
        path = str(cn_dir / "图.jpg")
        imwrite_unicode(path, np.full((16, 16, 3), 200, dtype=np.uint8))
        decoded = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert decoded.shape == (16, 16, 3)


# === FaceTracker 的在线代表选择 ===

class TestFaceTrackerRepresentative:
    @pytest.fixture
    def config(self):
        return Config(iou_threshold=0.3, track_gap_tolerance=1)

    def test_picks_highest_blur_times_confidence(self, config):
        # 代表 = 该轨迹内 blur_var * confidence 最大的那个检测结果。
        best_crop = np.full((4, 4, 3), 7, dtype=np.uint8)
        tracker = FaceTracker(config)
        tracker.update(0, False, [make_det(0, 0.0, (0, 0, 100, 100), 0.9, 10.0,
                                           np.zeros((4, 4, 3), dtype=np.uint8))])
        tracker.update(1, False, [make_det(1, 0.3, (0, 0, 100, 100), 0.9, 99.0, best_crop)])
        tracker.update(2, False, [make_det(2, 0.6, (0, 0, 100, 100), 0.9, 20.0,
                                           np.zeros((4, 4, 3), dtype=np.uint8))])
        tracks = tracker.finish()
        assert len(tracks) == 1
        assert tracks[0].representative_frame == 1
        assert tracks[0].representative_time == pytest.approx(0.3)
        assert np.array_equal(tracks[0].representative_image, best_crop)

    def test_losing_crops_are_released(self, config):
        # 落选裁剪图必须立刻解引用，否则会随 dets 列表一路累积到轨迹结束。
        dets = [
            make_det(0, 0.0, (0, 0, 100, 100), 0.9, 99.0, np.zeros((4, 4, 3), dtype=np.uint8)),
            make_det(1, 0.3, (0, 0, 100, 100), 0.9, 1.0, np.zeros((4, 4, 3), dtype=np.uint8)),
        ]
        tracker = FaceTracker(config)
        tracker.update(0, False, [dets[0]])
        tracker.update(1, False, [dets[1]])
        tracker.finish()
        assert dets[0].crop is None  # 胜出者的图已转移到轨迹上
        assert dets[1].crop is None  # 落选者的图已丢弃

    def test_each_track_keeps_its_own_crop(self, config):
        left = np.full((4, 4, 3), 1, dtype=np.uint8)
        right = np.full((4, 4, 3), 2, dtype=np.uint8)
        tracker = FaceTracker(config)
        tracker.update(0, False, [
            make_det(0, 0.0, (0, 0, 100, 100), 0.9, 10.0, left),
            make_det(0, 0.0, (500, 0, 600, 100), 0.9, 10.0, right),
        ])
        tracks = tracker.finish()
        assert len(tracks) == 2
        images = [t.representative_image for t in tracks]
        assert any(np.array_equal(im, left) for im in images)
        assert any(np.array_equal(im, right) for im in images)

    def test_scene_cut_splits_representatives(self, config):
        # 切换断轨后是两条独立轨迹，各自选各自的代表。
        first = np.full((4, 4, 3), 1, dtype=np.uint8)
        second = np.full((4, 4, 3), 2, dtype=np.uint8)
        tracker = FaceTracker(config)
        tracker.update(0, False, [make_det(0, 0.0, (0, 0, 100, 100), 0.9, 10.0, first)])
        tracker.update(1, True, [make_det(1, 0.3, (0, 0, 100, 100), 0.9, 99.0, second)])
        tracks = tracker.finish()
        assert len(tracks) == 2
        assert np.array_equal(tracks[0].representative_image, first)
        assert np.array_equal(tracks[1].representative_image, second)

    def test_no_crop_still_records_representative_position(self, config):
        # 裁剪图为 None（退化框）时轨迹仍然成立，只是没有图可聚类。
        tracker = FaceTracker(config)
        tracker.update(0, False, [make_det(0, 0.0, (0, 0, 100, 100), 0.9, 5.0, None)])
        tracks = tracker.finish()
        assert tracks[0].representative_image is None
        assert tracks[0].representative_frame == 0

    def test_finish_is_idempotent(self, config):
        tracker = FaceTracker(config)
        tracker.update(0, False, [make_det(0, 0.0, (0, 0, 100, 100))])
        assert len(tracker.finish()) == 1
        assert len(tracker.finish()) == 1  # 不会把同一条轨迹再封存一次
