"""iou / track_faces / select_segments 的单元测试。

三个函数都是纯计算，不触碰 ffmpeg、检测器模型或磁盘，
因此直接构造 Detection / Track / Config 即可覆盖核心分支。

运行方式（Windows PowerShell，项目根目录下）：
    D:\Programs\DevEnvironments\Anaconda\anaconda3\envs\myenv\python.exe -m pytest tests -v
"""

import pytest

from config import Config
from detectors import Detection
from main import Track, iou, select_segments, track_faces


def make_det(frame_index, time, bbox, label="anime_face", confidence=0.9):
    """构造一个最小可用的 Detection。"""
    return Detection(
        frame_index=frame_index,
        time=time,
        bbox=bbox,
        confidence=confidence,
        label=label,
    )


def make_track(track_id, start_time, end_time=None, label="anime_face"):
    """构造一个只关心起点时间的 Track（select_segments 只用这两个字段）。"""
    return Track(
        track_id=track_id,
        label=label,
        start_time=start_time,
        end_time=end_time if end_time is not None else start_time,
        detections=[],
    )


# === iou ===

class TestIou:
    def test_identical_boxes(self):
        assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0

    def test_edge_touching_counts_as_zero(self):
        # 仅边缘相接，交集面积为 0。
        assert iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0

    def test_partial_overlap(self):
        # 交集 5x10=50，并集 100+100-50=150。
        assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)

    def test_containment(self):
        # b 完全在 a 内：交集 36，并集为大框面积 100。
        assert iou((0, 0, 10, 10), (2, 2, 8, 8)) == pytest.approx(0.36)

    def test_degenerate_box_returns_zero(self):
        # 零面积框与任何框的交集都是 0。
        assert iou((5, 5, 5, 5), (0, 0, 10, 10)) == 0.0

    def test_symmetry(self):
        a, b = (0, 0, 10, 10), (3, 4, 12, 14)
        assert iou(a, b) == pytest.approx(iou(b, a))


# === track_faces ===

class TestTrackFaces:
    @pytest.fixture
    def config(self):
        return Config(iou_threshold=0.3, track_gap_tolerance=1)

    def test_empty_input(self, config):
        assert track_faces([], [], config) == []

    def test_consecutive_overlap_joins_one_track(self, config):
        frames = [
            [make_det(0, 0.0, (0, 0, 100, 100))],
            [make_det(1, 0.3, (5, 5, 105, 105))],
        ]
        tracks = track_faces(frames, [False, False], config)
        assert len(tracks) == 1
        assert len(tracks[0].detections) == 2
        assert tracks[0].start_time == pytest.approx(0.0)
        assert tracks[0].end_time == pytest.approx(0.3)

    def test_low_iou_starts_new_track(self, config):
        frames = [
            [make_det(0, 0.0, (0, 0, 10, 10))],
            [make_det(1, 0.3, (200, 200, 210, 210))],
        ]
        tracks = track_faces(frames, [False, False], config)
        assert len(tracks) == 2

    def test_gap_within_tolerance_reconnects(self, config):
        # 帧 1 丢失，gap=1 <= track_gap_tolerance=1，仍然接回同一条轨迹。
        frames = [
            [make_det(0, 0.0, (0, 0, 100, 100))],
            [],
            [make_det(2, 0.6, (0, 0, 100, 100))],
        ]
        tracks = track_faces(frames, [False, False, False], config)
        assert len(tracks) == 1
        assert len(tracks[0].detections) == 2

    def test_gap_beyond_tolerance_splits(self, config):
        # 连续丢 2 帧，超过容忍值，拆成两条轨迹。
        frames = [
            [make_det(0, 0.0, (0, 0, 100, 100))],
            [],
            [],
            [make_det(3, 0.9, (0, 0, 100, 100))],
        ]
        tracks = track_faces(frames, [False] * 4, config)
        assert len(tracks) == 2

    def test_scene_cut_breaks_track_despite_high_iou(self, config):
        # 同一位置、IoU=1，但中间有镜头切换，必须断开。
        frames = [
            [make_det(0, 0.0, (0, 0, 100, 100))],
            [make_det(1, 0.3, (0, 0, 100, 100))],
        ]
        tracks = track_faces(frames, [False, True], config)
        assert len(tracks) == 2

    def test_cut_blocks_gap_reconnection(self, config):
        # 丢帧在容忍范围内，但切换发生在丢失帧与恢复帧之间，禁止重连。
        frames = [
            [make_det(0, 0.0, (0, 0, 100, 100))],
            [],
            [make_det(2, 0.6, (0, 0, 100, 100))],
        ]
        tracks = track_faces(frames, [False, False, True], config)
        assert len(tracks) == 2

    def test_different_labels_never_join(self, config):
        frames = [
            [make_det(0, 0.0, (0, 0, 100, 100), label="cat")],
            [make_det(1, 0.3, (0, 0, 100, 100), label="dog")],
        ]
        tracks = track_faces(frames, [False, False], config)
        assert len(tracks) == 2

    def test_two_parallel_tracks(self, config):
        frames = [
            [make_det(0, 0.0, (0, 0, 100, 100)), make_det(0, 0.0, (500, 0, 600, 100))],
            [make_det(1, 0.3, (2, 2, 102, 102)), make_det(1, 0.3, (502, 0, 602, 100))],
        ]
        tracks = track_faces(frames, [False, False], config)
        assert len(tracks) == 2
        assert all(len(t.detections) == 2 for t in tracks)

    def test_tracks_sorted_by_start_time(self, config):
        frames = [
            [make_det(0, 0.0, (0, 0, 100, 100))],
            [make_det(1, 0.3, (0, 0, 100, 100)), make_det(1, 0.3, (500, 0, 600, 100))],
        ]
        tracks = track_faces(frames, [False, False], config)
        starts = [t.start_time for t in tracks]
        assert starts == sorted(starts)

    def test_track_ids_unique(self, config):
        frames = [
            [make_det(0, 0.0, (0, 0, 10, 10)), make_det(0, 0.0, (100, 100, 110, 110))],
            [make_det(1, 0.3, (300, 300, 310, 310))],
        ]
        tracks = track_faces(frames, [False, False], config)
        ids = [t.track_id for t in tracks]
        assert len(ids) == len(set(ids))


# === select_segments ===

class TestSelectSegments:
    @pytest.fixture
    def config(self):
        # 小参数便于手算：窗口 5 秒、步进 1 秒、窗口内至少 2 个轨迹起点。
        return Config(window_seconds=5.0, frame_interval=1.0, min_events_per_window=2)

    def test_no_tracks_no_segments(self, config):
        segments, num_qualified = select_segments([], 100.0, config)
        assert segments == []
        assert num_qualified == 0

    def test_single_qualified_window(self, config):
        tracks = [make_track(1, 0.0), make_track(2, 1.0)]
        segments, num_qualified = select_segments(tracks, 10.0, config)
        assert num_qualified == 1
        assert len(segments) == 1
        seg = segments[0]
        assert seg["start"] == pytest.approx(0.0)
        assert seg["end"] == pytest.approx(5.0)
        assert seg["event_count"] == 2
        assert seg["track_ids"] == [1, 2]

    def test_not_enough_events(self, config):
        tracks = [make_track(1, 0.0)]
        segments, num_qualified = select_segments(tracks, 10.0, config)
        assert segments == []
        assert num_qualified == 0

    def test_duration_shorter_than_window(self, config):
        # 视频总长不足一个窗口时，没有任何候选窗口。
        tracks = [make_track(1, 0.0), make_track(2, 1.0)]
        segments, num_qualified = select_segments(tracks, 4.9, config)
        assert segments == []
        assert num_qualified == 0

    def test_window_end_is_exclusive(self, config):
        # 起点恰好等于 t+window 的轨迹不计入 [t, t+W) 窗口：
        # 两个起点相距整整一个窗口长度，任何 5 秒窗口都无法同时框住两者。
        tracks = [make_track(1, 0.0), make_track(2, 5.0)]
        segments, num_qualified = select_segments(tracks, 10.0, config)
        assert segments == []
        assert num_qualified == 0

    def test_segments_do_not_overlap(self, config):
        # 前 4 个事件都落在第一个窗口内，贪心选中后应跳过其余重叠的合格窗口。
        tracks = [make_track(i, float(i)) for i in range(1, 5)]  # 起点 1,2,3,4
        segments, num_qualified = select_segments(tracks, 30.0, config)
        assert num_qualified == 1
        assert len(segments) == 1
        assert segments[0]["start"] == pytest.approx(0.0)

    def test_two_disjoint_segments(self, config):
        tracks = [
            make_track(1, 0.0),
            make_track(2, 1.0),
            make_track(3, 6.0),
            make_track(4, 7.0),
        ]
        segments, num_qualified = select_segments(tracks, 12.0, config)
        assert num_qualified == 2
        assert len(segments) == 2
        # 第二个窗口从上一个窗口结束处(>=5.0)开始，互不重叠。
        assert segments[0]["end"] <= segments[1]["start"]
        assert segments[1]["track_ids"] == [3, 4]

    def test_unsorted_track_input(self, config):
        # select_segments 内部会对起点排序，输入顺序不应影响结果。
        tracks = [make_track(2, 1.0), make_track(1, 0.0)]
        segments, num_qualified = select_segments(tracks, 10.0, config)
        assert num_qualified == 1
        assert segments[0]["event_count"] == 2
