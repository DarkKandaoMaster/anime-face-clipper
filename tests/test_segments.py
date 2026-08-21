r"""windows_scan.json 的扫描表与其上的选段重放的单元测试。

关注点只有两个：
  1. scan_windows 落下的表能不能让 select_from_scan 重放出与流水线**逐字节一致**
     的片段（Web 端拖 X 滑块走的就是这条路径，走偏了两边数字就对不上）；
  2. 吸附到切镜点时用的角色数确实来自 cut_windows，而不是拿网格点凑合。

全部是纯计算，不碰模型、磁盘和 ffmpeg。

运行方式（Windows PowerShell，项目根目录下）：
    D:\Programs\DevEnvironments\Anaconda\anaconda3\envs\myenv\python.exe -m pytest tests/test_segments.py -v
"""

import dataclasses

import pytest

from config import Config
from main import Track, scan_windows, select_segments
from segments import nearest_cut, select_from_scan


def make_track(track_id, start_time, end_time=None, character_id=None):
    return Track(
        track_id=track_id,
        label="anime_face",
        start_time=start_time,
        end_time=end_time if end_time is not None else start_time,
        detections=[],
        character_id=character_id,
    )


@pytest.fixture
def config():
    # 与 TestSelectSegments 同款小参数：窗口 5 秒、步进 1 秒、X = 2。
    return Config(window_seconds=5.0, frame_interval=1.0, min_events_per_window=2)


class TestScanWindows:
    def test_grid_index_matches_start_time(self, config):
        tracks = [make_track(1, 0.0, character_id=0), make_track(2, 1.0, character_id=1)]
        scan = scan_windows(tracks, 10.0, config)
        # 下标 k 必须对应 t = k × frame_interval，select_from_scan 直接按下标取值。
        assert [w["t"] for w in scan["windows"]] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        assert scan["duration"] == 10.0
        assert scan["window_seconds"] == 5.0
        assert scan["frame_interval"] == 1.0

    def test_counts_are_independent_of_x(self, config):
        tracks = [make_track(1, 0.0, character_id=0), make_track(2, 1.0, character_id=1)]
        loose = scan_windows(tracks, 10.0, config)
        strict = scan_windows(tracks, 10.0, dataclasses.replace(config, min_events_per_window=9))
        # X 只在选段那一步做比较，扫描表本身与它无关——整个方案就建立在这一条上。
        assert loose["windows"] == strict["windows"]

    def test_window_shorter_than_duration_yields_no_windows(self, config):
        scan = scan_windows([make_track(1, 0.0, character_id=0)], 4.0, config)
        assert scan["windows"] == []

    def test_cut_windows_cover_每个可用切镜点(self, config):
        tracks = [make_track(1, 0.0, end_time=9.0, character_id=0),
                  make_track(2, 0.0, end_time=9.0, character_id=1)]
        # 8.0 起的窗口越过 10 秒时长，不该出现在 cut_windows 里。
        scan = scan_windows(tracks, 10.0, config, cuts=[1.5, 8.0])
        assert [w["t"] for w in scan["cut_windows"]] == [1.5]
        assert scan["cuts"] == [1.5, 8.0]


class TestReplayMatchesPipeline:
    """重放结果必须与 select_segments 完全一致——这是本次改动的全部理由。"""

    def _assert_same(self, tracks, duration, config, cuts=None):
        segments, num_qualified = select_segments(tracks, duration, config, cuts)
        scan = scan_windows(tracks, duration, config, cuts)
        picks, replay_qualified = select_from_scan(
            scan, config.min_events_per_window, config.clip_snap_max_shift
        )
        assert replay_qualified == num_qualified
        assert [(p["start"], p["end"], p["character_count"]) for p in picks] == [
            (s["start"], s["end"], s["character_count"]) for s in segments
        ]
        return picks

    def test_no_tracks(self, config):
        assert self._assert_same([], 100.0, config) == []

    def test_single_window(self, config):
        tracks = [make_track(1, 0.0, character_id=0), make_track(2, 1.0, character_id=1)]
        picks = self._assert_same(tracks, 10.0, config)
        assert len(picks) == 1

    def test_non_overlapping_greedy_jump(self, config):
        tracks = [
            make_track(1, 0.0, character_id=0),
            make_track(2, 0.5, character_id=1),
            make_track(3, 6.0, character_id=2),
            make_track(4, 6.5, character_id=3),
        ]
        picks = self._assert_same(tracks, 20.0, config)
        assert len(picks) == 2
        assert picks[1]["start"] >= picks[0]["end"]

    def test_snap_to_cut(self, config):
        # 两条轨迹一直在画面里，任何起点都合格 → 起点必被吸附到切镜点。
        tracks = [make_track(1, 0.0, end_time=20.0, character_id=0),
                  make_track(2, 0.0, end_time=20.0, character_id=1)]
        picks = self._assert_same(tracks, 20.0, config, cuts=[1.5, 12.0])
        assert picks[0]["start"] == pytest.approx(1.5)

    def test_snap_rejected_when_count_drops(self, config):
        # 吸附点 1.5 之后第二个角色已经离场，复核不达标 → 保持原起点 0.0。
        tracks = [make_track(1, 0.0, end_time=6.0, character_id=0),
                  make_track(2, 0.0, end_time=1.0, character_id=1)]
        picks = self._assert_same(tracks, 10.0, config, cuts=[1.5])
        assert picks[0]["start"] == pytest.approx(0.0)

    def test_x_changes_segment_count_on_same_scan(self, config):
        """同一张扫描表上换 X 就换答案——Web 端拖滑块要的就是这个。"""
        tracks = [
            make_track(1, 0.0, end_time=20.0, character_id=0),
            make_track(2, 0.0, end_time=20.0, character_id=1),
            make_track(3, 0.0, end_time=2.0, character_id=2),
        ]
        scan = scan_windows(tracks, 20.0, config)
        assert len(select_from_scan(scan, 2, 0.0)[0]) == 4
        assert len(select_from_scan(scan, 3, 0.0)[0]) == 1
        assert select_from_scan(scan, 9, 0.0)[0] == []


class TestNearestCut:
    def test_picks_closest_within_shift(self):
        assert nearest_cut(5.0, [1.0, 4.6, 9.0], 2.0) == pytest.approx(4.6)

    def test_returns_none_beyond_shift(self):
        assert nearest_cut(5.0, [1.0, 9.0], 2.0) is None

    def test_disabled_by_zero_shift(self):
        assert nearest_cut(5.0, [4.9], 0.0) is None

    def test_no_cuts(self):
        assert nearest_cut(5.0, [], 2.0) is None
