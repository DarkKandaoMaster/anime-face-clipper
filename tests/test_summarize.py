"""`summarize.collect` 的纯逻辑单测（tmp_path 下现造 windows.json）。"""

import json

from summarize import collect


def _write(root, stem, duration, segments, window=30.0, style="2d"):
    out = root / stem
    out.mkdir(parents=True)
    (out / "windows.json").write_text(json.dumps({
        "duration": duration,
        "style": style,
        "num_tracks": 10,
        "num_characters": 5,
        "params": {"window_seconds": window},
        "segments": [{"character_count": c} for c in segments],
    }), encoding="utf-8")


class TestCollect:
    def test_counts_and_ratio(self, tmp_path):
        _write(tmp_path, "a", 1420.0, [4, 9, 7])
        _write(tmp_path, "b", 95.0, [5])
        rows = collect(str(tmp_path))
        assert [r["name"] for r in rows] == ["a", "b"]
        assert rows[0]["max_windows"] == 47          # floor(1420 / 30)
        assert rows[1]["max_windows"] == 3           # floor(95 / 30)，不进位
        assert rows[0]["segments"] == 3

    def test_median_and_max_of_window_counts(self, tmp_path):
        _write(tmp_path, "a", 300.0, [9, 4, 7, 19])
        row = collect(str(tmp_path))[0]
        assert row["max"] == 19
        assert row["median"] == 9                    # 排序后取中间偏右

    def test_no_segments(self, tmp_path):
        _write(tmp_path, "a", 300.0, [])
        row = collect(str(tmp_path))[0]
        assert (row["segments"], row["median"], row["max"]) == (0, 0, 0)

    def test_window_seconds_comes_from_the_run_not_a_constant(self, tmp_path):
        """窗口长度读的是那次运行写下的参数，改了 config 也不会让旧结果被误读。"""
        _write(tmp_path, "a", 300.0, [4], window=15.0)
        assert collect(str(tmp_path))[0]["max_windows"] == 20

    def test_empty_root(self, tmp_path):
        assert collect(str(tmp_path)) == []


class TestMinTrackSeconds:
    """最短出镜时长门槛：对应人工标注的「出镜 >1s」口径，流水线此前没有实现。"""

    def _tracks(self, min_seconds, spans):
        import numpy as np
        from config import Config
        from detectors import Detection
        from main import FaceTracker, filter_short_tracks

        # 每条轨迹用一个独立位置的框，避免相互 IoU 匹配串成一条。
        tracker = FaceTracker(Config())
        last = max(n for _x, n in spans)
        for i in range(last):
            dets = []
            for slot, (x, n) in enumerate(spans):
                if i < n:
                    dets.append(Detection(
                        frame_index=i, time=i * 0.3,
                        bbox=(x, 0, x + 10, 10), confidence=1.0, label="anime_face",
                        blur_var=100.0, crop=np.zeros((4, 4, 3), dtype=np.uint8),
                    ))
            tracker.update(i, False, dets)
        return filter_short_tracks(tracker.finish(), min_seconds)

    def test_off_keeps_single_frame_tracks(self):
        # 默认 0：一闪而过的脸也留着（当前默认行为）。
        assert len(self._tracks(0.0, [(0, 1), (100, 8)])) == 2

    def test_drops_tracks_shorter_than_threshold(self):
        # 8 个采样帧 = 2.1 秒（时长按 end-start 算，n 帧是 (n-1)×0.3）。
        tracks = self._tracks(1.0, [(0, 1), (100, 8)])
        assert len(tracks) == 1
        assert len(tracks[0].detections) == 8

    def test_boundary_is_inclusive(self):
        # 5 帧 = 1.2 秒 >= 1.2，恰好保留。
        assert len(self._tracks(1.2, [(0, 5)])) == 1
        assert len(self._tracks(1.3, [(0, 5)])) == 0
