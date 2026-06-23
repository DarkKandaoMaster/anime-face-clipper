"""Anime face clipper — end-to-end pipeline.

From an anime video, find and cut every "qualified" 15-second segment, where a
segment qualifies when at least ``min_events_per_window`` distinct face-track
*starts* fall inside it. A track is a chain of IoU-overlapping face boxes across
adjacent frames; a shot cut forcibly breaks tracks so a different character
appearing in the same spot counts as a new event.

Run from the project root:

    python -m src.main                      # processes data/1.mp4 -> output/1/
    python -m src.main data/1.mp4 --viz 8   # also dump annotated sample frames

Stages (organized by section comments below):
    抽帧 -> 检测 -> 过滤 -> 跟踪 -> 选段 -> 截取
"""

import argparse
import bisect
import dataclasses
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

import cv2

from config import Config
from detectors import Detection, Detector, get_detector

# Re-export so callers can also do ``from src.main import Detection``. The class
# is defined in detectors.py to keep the detector module free of cycles.
__all__ = ["Detection", "Track", "process_video", "run_pipeline", "main"]


@dataclasses.dataclass
class Track:
    """A single face-appearance event: a chain of detections over time.

    Attributes:
        track_id: Unique id within one video.
        label: Category inherited from the detections.
        start_time: Timestamp of the first detection (used for window counting).
        end_time: Timestamp of the last detection.
        detections: The member detections in time order.
        representative_frame: Frame index of the sharpest detection.
        representative_time: Timestamp of that detection.
        representative_bbox: Box of that detection (for re-fetching from video).
        representative_crop: Path to the saved crop image (relative to output).
    """

    track_id: int
    label: str
    start_time: float
    end_time: float
    detections: List[Detection]
    representative_frame: int = -1
    representative_time: float = 0.0
    representative_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    representative_crop: str = ""


# Module-level guard so the active ONNX providers are reported only once.
_providers_reported = False


# === 抽帧 ===

def extract_frames(config: Config, video_path: str, frames_dir: str) -> List[Tuple[int, float, str]]:
    """Sample frames with ffmpeg at a fixed interval.

    Args:
        config: Pipeline configuration.
        video_path: Source video.
        frames_dir: Existing directory to write JPEGs into.

    Returns:
        A list of ``(frame_index, time_seconds, frame_path)`` in time order.
    """
    pattern = os.path.join(frames_dir, "%06d.jpg")
    cmd = [
        config.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", video_path,
        "-vf", f"fps=1/{config.frame_interval}",
        "-q:v", "2",
        pattern,
    ]
    subprocess.run(cmd, check=True)

    files = sorted(
        f for f in os.listdir(frames_dir) if f.lower().endswith(".jpg")
    )
    frames = []
    for index, name in enumerate(files):
        time = index * config.frame_interval
        frames.append((index, time, os.path.join(frames_dir, name)))
    return frames


def compute_hsv_hist(image_bgr):
    """Compute a normalized HSV (H, S) histogram for shot-cut comparison."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


# === 过滤 ===

def laplacian_variance(image_bgr, bbox: Tuple[int, int, int, int]) -> float:
    """Laplacian variance of a bbox crop (focus/blur measure).

    Returns ``0.0`` for empty/degenerate crops.
    """
    x1, y1, x2, y2 = bbox
    h, w = image_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    crop = image_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def passes_quality(detection: Detection, frame_height: int, config: Config) -> bool:
    """Apply the three quality gates: confidence, face size, sharpness."""
    if detection.confidence < config.conf_threshold:
        return False
    face_height = detection.bbox[3] - detection.bbox[1]
    if face_height < config.min_face_height_ratio * frame_height:
        return False
    if (detection.blur_var or 0.0) < config.blur_var_threshold:
        return False
    return True


# === 跟踪 ===

def iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """Intersection-over-union of two ``(x1, y1, x2, y2)`` boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _cut_between(is_cut: List[bool], last_index: int, current_index: int) -> bool:
    """True if a shot cut occurs in frames (last_index, current_index]."""
    return any(is_cut[last_index + 1:current_index + 1])


def track_faces(
    frame_detections: List[List[Detection]],
    is_cut: List[bool],
    config: Config,
) -> List[Track]:
    """Link per-frame detections into tracks with IoU and shot-cut breaking.

    Adjacent detections with IoU >= ``iou_threshold`` and matching label join
    the same track. Up to ``track_gap_tolerance`` missed frames are tolerated.
    A shot cut between two frames forbids linking across it, even at high IoU
    (gap-tolerant re-links may not span a cut either).

    Args:
        frame_detections: Per-frame lists of *quality-passed* detections.
        is_cut: Per-frame flags; ``is_cut[i]`` marks a cut between frame i-1, i.
        config: Pipeline configuration.

    Returns:
        All tracks, sorted by start time.
    """
    active: List[Dict] = []  # each: {id, label, last_index, dets:[Detection]}
    finalized: List[Track] = []
    next_id = 1

    def _finalize(track: Dict) -> None:
        dets = track["dets"]
        finalized.append(
            Track(
                track_id=track["id"],
                label=track["label"],
                start_time=dets[0].time,
                end_time=dets[-1].time,
                detections=dets,
            )
        )

    for i, dets in enumerate(frame_detections):
        # Drop tracks that can no longer be revived: gap exceeded, or a cut now
        # sits between their last detection and the current frame.
        still_active = []
        for tr in active:
            gap = i - tr["last_index"] - 1
            if gap > config.track_gap_tolerance or _cut_between(is_cut, tr["last_index"], i):
                _finalize(tr)
            else:
                still_active.append(tr)
        active = still_active

        # Greedy IoU matching: best pairs first, each track/detection used once.
        pairs = []
        for ti, tr in enumerate(active):
            last_box = tr["dets"][-1].bbox
            for di, det in enumerate(dets):
                if det.label != tr["label"]:
                    continue
                score = iou(last_box, det.bbox)
                if score >= config.iou_threshold:
                    pairs.append((score, ti, di))
        pairs.sort(reverse=True)

        matched_tracks, matched_dets = set(), set()
        for _score, ti, di in pairs:
            if ti in matched_tracks or di in matched_dets:
                continue
            matched_tracks.add(ti)
            matched_dets.add(di)
            active[ti]["dets"].append(dets[di])
            active[ti]["last_index"] = i

        # Unmatched detections start new tracks.
        for di, det in enumerate(dets):
            if di in matched_dets:
                continue
            active.append({"id": next_id, "label": det.label, "last_index": i, "dets": [det]})
            next_id += 1

    for tr in active:
        _finalize(tr)

    finalized.sort(key=lambda t: t.start_time)
    return finalized


def assign_representatives(tracks: List[Track], frame_paths: Dict[int, str], crops_dir: str) -> None:
    """Pick each track's sharpest detection and save its crop.

    The representative maximizes ``blur_var * confidence`` over the track's
    detections. The crop is read back from the corresponding sampled frame and
    written to ``crops_dir``; the track records the path and source location.
    """
    os.makedirs(crops_dir, exist_ok=True)
    for track in tracks:
        best = max(
            track.detections,
            key=lambda d: (d.blur_var or 0.0) * d.confidence,
        )
        track.representative_frame = best.frame_index
        track.representative_time = best.time
        track.representative_bbox = best.bbox

        frame_path = frame_paths.get(best.frame_index)
        if not frame_path:
            continue
        image = cv2.imread(frame_path)
        if image is None:
            continue
        x1, y1, x2, y2 = best.bbox
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        crop_name = f"track_{track.track_id}.jpg"
        cv2.imwrite(os.path.join(crops_dir, crop_name), image[y1:y2, x1:x2])
        track.representative_crop = os.path.join("crops", crop_name)


# === 选段 ===

def select_segments(
    tracks: List[Track],
    duration: float,
    config: Config,
) -> Tuple[List[Dict], int]:
    """Slide a window over track-start times and greedily pick segments.

    Candidate window starts step by ``frame_interval``. A window ``[t, t+W)``
    qualifies when it contains at least ``min_events_per_window`` track starts.
    On a qualifying window the segment ``[t, t+W]`` is emitted and the next
    candidate jumps to ``>= t+W`` so segments never overlap.

    Returns:
        A tuple ``(segments, num_qualified_windows)`` where each segment is a
        dict with ``start``, ``end``, ``event_count`` and ``track_ids``.
    """
    starts = sorted(t.start_time for t in tracks)
    window = config.window_seconds
    step = config.frame_interval

    segments: List[Dict] = []
    num_qualified = 0
    k = 0
    while True:
        t = k * step
        if t + window > duration + 1e-6:
            break
        lo = bisect.bisect_left(starts, t)
        hi = bisect.bisect_left(starts, t + window)
        count = hi - lo
        if count >= config.min_events_per_window:
            num_qualified += 1
            track_ids = [
                tr.track_id for tr in tracks if t <= tr.start_time < t + window
            ]
            segments.append(
                {
                    "start": round(t, 3),
                    "end": round(t + window, 3),
                    "event_count": count,
                    "track_ids": track_ids,
                }
            )
            k = math.ceil((t + window) / step - 1e-9)
        else:
            k += 1
    return segments, num_qualified


# === 截取 ===

def _encode_clip(config: Config, video_path: str, start: float, out_path: str, encoder: str) -> bool:
    """Cut one re-encoded, frame-accurate clip with the given encoder."""
    cmd = [
        config.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start:.3f}",
        "-i", video_path,
        "-t", f"{config.window_seconds:.3f}",
        "-c:v", encoder,
        "-c:a", "aac",
        "-movflags", "+faststart",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
    return result.returncode == 0


def clip_segments(config: Config, video_path: str, segments: List[Dict], clips_dir: str) -> List[str]:
    """Cut all selected segments, preferring GPU encoder with CPU fallback."""
    os.makedirs(clips_dir, exist_ok=True)
    encoder = config.encoder
    out_paths = []
    for idx, segment in enumerate(segments, start=1):
        out_path = os.path.join(clips_dir, f"clip_{idx:03d}.mp4")
        ok = _encode_clip(config, video_path, segment["start"], out_path, encoder)
        if not ok and encoder != config.encoder_fallback:
            print(f"  encoder {encoder!r} failed, falling back to {config.encoder_fallback!r}")
            encoder = config.encoder_fallback
            ok = _encode_clip(config, video_path, segment["start"], out_path, encoder)
        if ok:
            out_paths.append(out_path)
        else:
            print(f"  failed to cut segment {idx} at {segment['start']}s")
    return out_paths


# === 工具 ===

def probe_duration(config: Config, video_path: str) -> float:
    """Return the video duration in seconds via ffprobe."""
    cmd = [
        config.ffprobe, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def _write_json(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _detection_record(det: Detection, kept: bool) -> Dict:
    return {
        "frame_index": det.frame_index,
        "time": round(det.time, 3),
        "bbox": list(det.bbox),
        "confidence": round(det.confidence, 4),
        "label": det.label,
        "blur_var": round(det.blur_var, 2) if det.blur_var is not None else None,
        "kept": kept,
    }


def _track_record(track: Track) -> Dict:
    return {
        "track_id": track.track_id,
        "label": track.label,
        "start_time": round(track.start_time, 3),
        "end_time": round(track.end_time, 3),
        "num_detections": len(track.detections),
        "representative_frame": track.representative_frame,
        "representative_time": round(track.representative_time, 3),
        "representative_bbox": list(track.representative_bbox),
        "representative_crop": track.representative_crop,
    }


def _report_providers(detector: Detector) -> None:
    """Print ONNX providers once, to confirm GPU usage (verification step 1)."""
    global _providers_reported
    if _providers_reported:
        return
    _providers_reported = True
    try:
        import onnxruntime

        print(f"  onnxruntime available providers: {onnxruntime.get_available_providers()}")
    except Exception:
        pass
    active = detector.actual_providers()
    if active:
        print(f"  active session providers: {active}")


def _save_visualization(viz_dir: str, frame_path: str, detections: List[Detection]) -> None:
    """Draw boxes + scores on a frame and save it (verification step 2)."""
    image = cv2.imread(frame_path)
    if image is None:
        return
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image, f"{det.confidence:.2f}", (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
        )
    os.makedirs(viz_dir, exist_ok=True)
    cv2.imwrite(os.path.join(viz_dir, os.path.basename(frame_path)), image)


# === 编排 ===

def process_video(
    config: Config,
    video_path: str,
    output_root: str,
    detector: Optional[Detector] = None,
    limit_seconds: Optional[float] = None,
    viz_count: int = 0,
    keep_frames: bool = False,
) -> Dict:
    """Run the full pipeline for one video.

    Args:
        config: Pipeline configuration.
        video_path: Source video path.
        output_root: Base output directory; results go under ``<root>/<stem>/``.
        detector: A shared detector instance (created if ``None``). Passing one
            in lets a batch reuse a single loaded model.
        limit_seconds: If set, only process frames before this timestamp (quick
            calibration runs).
        viz_count: Number of random annotated sample frames to dump.
        keep_frames: Keep the temporary extracted frames instead of deleting.

    Returns:
        A summary dict (also persisted across the JSON files).
    """
    if detector is None:
        detector = get_detector(config.detector, config)

    stem = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join(output_root, stem)
    os.makedirs(out_dir, exist_ok=True)
    frames_dir = tempfile.mkdtemp(prefix=f"afc_{stem}_")

    try:
        print(f"[{stem}] extracting frames -> {frames_dir}")
        frames = extract_frames(config, video_path, frames_dir)
        if limit_seconds is not None:
            frames = [f for f in frames if f[1] < limit_seconds]
        print(f"[{stem}] {len(frames)} frames; detecting + filtering...")

        frame_paths: Dict[int, str] = {}
        frame_detections: List[List[Detection]] = []
        is_cut: List[bool] = []
        detection_records: List[Dict] = []
        prev_hist = None
        viz_candidates: List[Tuple[str, List[Detection]]] = []

        for index, time, path in frames:
            frame_paths[index] = path
            image = cv2.imread(path)
            if image is None:
                is_cut.append(False)
                frame_detections.append([])
                continue
            frame_h = image.shape[0]

            # Shot-cut flag vs previous sampled frame.
            hist = compute_hsv_hist(image)
            if prev_hist is None:
                is_cut.append(False)
            else:
                corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                is_cut.append(corr < config.scene_cut_threshold)
            prev_hist = hist

            # Detect, then apply the three quality gates.
            raw = detector.detect(path, index, time)
            _report_providers(detector)
            kept = []
            for det in raw:
                det.blur_var = laplacian_variance(image, det.bbox)
                ok = passes_quality(det, frame_h, config)
                detection_records.append(_detection_record(det, ok))
                if ok:
                    kept.append(det)
            frame_detections.append(kept)
            if kept:
                viz_candidates.append((path, kept))

        # Tracking with shot-cut breaking.
        print(f"[{stem}] tracking...")
        tracks = track_faces(frame_detections, is_cut, config)
        crops_dir = os.path.join(out_dir, "crops")
        assign_representatives(tracks, frame_paths, crops_dir)

        # Segment selection.
        duration = probe_duration(config, video_path)
        if limit_seconds is not None:
            duration = min(duration, limit_seconds)
        segments, num_qualified = select_segments(tracks, duration, config)
        print(
            f"[{stem}] {len(tracks)} tracks, {num_qualified} qualified windows, "
            f"{len(segments)} segments selected"
        )

        # Cut clips.
        clips_dir = os.path.join(out_dir, "clips")
        clip_paths = clip_segments(config, video_path, segments, clips_dir)

        # Optional detection visualization.
        if viz_count > 0 and viz_candidates:
            viz_dir = os.path.join(out_dir, "viz")
            sample = random.sample(viz_candidates, min(viz_count, len(viz_candidates)))
            for path, dets in sample:
                _save_visualization(viz_dir, path, dets)

        # Persist outputs.
        _write_json(os.path.join(out_dir, "detections.json"), detection_records)
        _write_json(
            os.path.join(out_dir, "tracks.json"),
            [_track_record(t) for t in tracks],
        )
        _write_json(
            os.path.join(out_dir, "windows.json"),
            {
                "video": video_path,
                "duration": round(duration, 3),
                "num_tracks": len(tracks),
                "num_qualified_windows": num_qualified,
                "params": dataclasses.asdict(config),
                "segments": segments,
                "clips": [os.path.relpath(p, out_dir) for p in clip_paths],
            },
        )

        summary = {
            "video": video_path,
            "frames": len(frames),
            "tracks": len(tracks),
            "qualified_windows": num_qualified,
            "segments": len(segments),
            "clips": len(clip_paths),
            "output_dir": out_dir,
        }
        print(f"[{stem}] done: {summary}")
        return summary
    finally:
        if keep_frames:
            print(f"[{stem}] frames kept at {frames_dir}")
        else:
            shutil.rmtree(frames_dir, ignore_errors=True)


def run_pipeline(
    config: Config,
    video_paths: List[str],
    output_root: str,
    **kwargs,
) -> List[Dict]:
    """Process one or more videos, reusing a single loaded detector.

    Batch support is intentionally minimal in v1: the loop and shared detector
    are here, the CLI just passes a single video.
    """
    detector = get_detector(config.detector, config)
    summaries = []
    for video_path in video_paths:
        summaries.append(
            process_video(config, video_path, output_root, detector=detector, **kwargs)
        )
    return summaries


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Anime face clipper.")
    parser.add_argument(
        "videos", nargs="*", default=["data/1.mp4"],
        help="Input video path(s). Default: data/1.mp4",
    )
    parser.add_argument("--output-dir", default="output", help="Output base directory.")
    # Calibration-friendly overrides (verification steps 2/3).
    parser.add_argument("--conf", type=float, help="Override conf_threshold.")
    parser.add_argument("--blur-var", type=float, help="Override blur_var_threshold.")
    parser.add_argument("--scene-cut", type=float, help="Override scene_cut_threshold.")
    parser.add_argument("--min-events", type=int, help="Override min_events_per_window.")
    parser.add_argument("--frame-interval", type=float, help="Override frame_interval.")
    parser.add_argument("--encoder", help="Override video encoder (e.g. libx264).")
    parser.add_argument("--limit-seconds", type=float, help="Only process first N seconds.")
    parser.add_argument("--viz", type=int, default=0, help="Dump N annotated sample frames.")
    parser.add_argument("--keep-frames", action="store_true", help="Keep temp frames.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point."""
    args = _build_arg_parser().parse_args(argv)

    config = Config()
    if args.conf is not None:
        config.conf_threshold = args.conf
    if args.blur_var is not None:
        config.blur_var_threshold = args.blur_var
    if args.scene_cut is not None:
        config.scene_cut_threshold = args.scene_cut
    if args.min_events is not None:
        config.min_events_per_window = args.min_events
    if args.frame_interval is not None:
        config.frame_interval = args.frame_interval
    if args.encoder is not None:
        config.encoder = args.encoder

    run_pipeline(
        config,
        args.videos,
        args.output_dir,
        limit_seconds=args.limit_seconds,
        viz_count=args.viz,
        keep_frames=args.keep_frames,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
