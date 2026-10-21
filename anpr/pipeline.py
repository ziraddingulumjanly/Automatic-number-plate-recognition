"""Run the notebook's video inference pipeline with the same tracking and vote rules."""
from pathlib import Path
import math
import subprocess
import time

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .detection import read_plates, plate_quality, normalize_plate, valid_plate, vote
from .tracking import associate, annotate


def create_alpr(cfg):
    # Import lazily so inspect/config commands never download neural model weights.
    from fast_alpr import ALPR
    return ALPR(
        detector_model=cfg.PLATE_DETECTOR,
        detector_conf_thresh=cfg.DETECTION_CONF,
        detector_providers=["CPUExecutionProvider"],
        ocr_model=cfg.PLATE_OCR,
        ocr_device="cpu",
        ocr_providers=["CPUExecutionProvider"],
    )


def inspect_video(cfg):
    if not cfg.VIDEO_PATH.is_file():
        raise FileNotFoundError(f"Input video not found: {cfg.VIDEO_PATH}")
    cap = cv2.VideoCapture(str(cfg.VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {cfg.VIDEO_PATH}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if np.isfinite(fps) and fps > 0 else 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if width <= 0 or height <= 0:
        raise RuntimeError("Input video has invalid dimensions")
    return fps, width, height, count


def run(cfg):
    VIDEO_PATH, OUTPUT_DIR = cfg.VIDEO_PATH, cfg.OUTPUT_DIR
    video_fps, video_w, video_h, video_frames = inspect_video(cfg)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Input: {VIDEO_PATH} | {video_w}x{video_h} | {video_fps:.2f} FPS")
    print(f"Models: {cfg.PLATE_DETECTOR} + {cfg.PLATE_OCR}")
    alpr = create_alpr(cfg)
    scan_stats = {"calls": 0, "regions": 0, "seconds": 0.0}
    OBS_CSV = OUTPUT_DIR / "plate_observations.csv"
    PLATES_CSV = OUTPUT_DIR / "recognized_plates.csv"
    RAW_VIDEO = OUTPUT_DIR / "annotated_raw.mp4"
    ANNOTATED_VIDEO = OUTPUT_DIR / "annotated_video.mp4"
    CROPS_DIR = OUTPUT_DIR / "best_plate_crops"
    if cfg.KEEP_BEST_CROPS:
        CROPS_DIR.mkdir(exist_ok=True, parents=True)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError("Cannot read the source video")
    source_fps = cap.get(cv2.CAP_PROP_FPS) or video_fps
    source_fps = source_fps if np.isfinite(source_fps) and source_fps > 0 else video_fps
    output_fps = min(cfg.SAMPLE_FPS, source_fps)
    writer = cv2.VideoWriter(str(RAW_VIDEO), cv2.VideoWriter_fourcc(*"mp4v"),
                             output_fps, (video_w, video_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Unable to create output MP4")

    tracks = {}
    next_tid = 1
    frame_no = 0
    sampled_frames = 0
    extra_frames = 0
    frames_with_plates = 0
    ocr_empty = 0
    next_sample_time = 0.0
    last_sample_time = None
    active_until = -1.0
    pending = []  # Only frames skipped since the last video-output sample (~1 / cfg.SAMPLE_FPS seconds).
    all_observations = []
    scan_stats.update(calls=0, regions=0, seconds=0.0)
    start = time.perf_counter()
    end_time = (video_frames / source_fps) if cfg.MAX_SECONDS is None and video_frames > 0 else cfg.MAX_SECONDS
    expected = math.ceil((min(video_frames/source_fps, cfg.MAX_SECONDS) if cfg.MAX_SECONDS is not None and video_frames > 0 else (end_time or 0)) * output_fps)
    pbar = tqdm(total=expected or None, desc="ANPR output frames")


    def current_focus_boxes(t_s):
        if not cfg.FOCUSED_RECHECK:
            return []
        active = sorted((tr for tr in tracks.values() if 0 <= t_s - tr.last_t <= cfg.TRACK_MAX_GAP_S),
                        key=lambda tr: tr.last_t, reverse=True)
        return [tr.box for tr in active[:cfg.MAX_FOCUS_REGIONS]]


    def record_frame_detections(frame, idx, t_s, detections):
        """Update tracks in chronological order; once per source frame (not once per tile)."""
        nonlocal next_tid, frames_with_plates, ocr_empty, active_until
        if detections:
            frames_with_plates += 1
            active_until = max(active_until, t_s + cfg.ACTIVE_HOLD_S)
        mapping, next_tid = associate(detections, tracks, t_s, next_tid, cfg)
        for di, detection in enumerate(detections):
            tid = mapping[di]
            track = tracks[tid]
            b = detection["box"]
            quality, sharpness = plate_quality(frame, b)
            text = normalize_plate(detection["raw_text"])
            if not text:
                ocr_empty += 1
            ob = {"track_id":tid, "frame":idx, "time_s":round(t_s, 3),
                  "x1":b[0], "y1":b[1], "x2":b[2], "y2":b[3],
                  "width_px":b[2]-b[0], "raw_text":detection["raw_text"],
                  "text":text, "valid_format":valid_plate(text, cfg),
                  "detection_conf":round(detection["detection_conf"], 3),
                  "ocr_conf":round(detection["ocr_conf"], 3),
                  "quality":quality, "sharpness":sharpness, "source":detection["source"]}
            all_observations.append(ob)
            track.observations.append(ob)
            if cfg.KEEP_BEST_CROPS and tid <= cfg.MAX_DEBUG_TRACKS and quality > track.best_quality:
                crop = frame[b[1]:b[3], b[0]:b[2]]
                if crop.size:
                    path = CROPS_DIR / f"plate_track_{tid:04d}_best.jpg"
                    if cv2.imwrite(str(path), crop):
                        track.best_crop_file = str(path)
                        track.best_quality = quality
        return mapping


    def select_intermediate_frames(buffer, left_t, right_t):
        """Select distinct real source frames near higher-FPS grid times between output samples."""
        if not buffer or left_t is None or cfg.ACTIVE_SCAN_FPS <= output_fps:
            return []
        selected = {}
        n = 1
        while left_t + n / cfg.ACTIVE_SCAN_FPS < right_t - 1 / (2 * source_fps):
            target = left_t + n / cfg.ACTIVE_SCAN_FPS
            idx, ts, frm = min(buffer, key=lambda item: abs(item[1] - target))
            if abs(ts - target) <= 0.5 / source_fps + 1e-6:
                selected[idx] = (idx, ts, frm)
            n += 1
        return [selected[i] for i in sorted(selected)]


    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t_s = frame_no / source_fps
            idx = frame_no
            frame_no += 1
            if cfg.MAX_SECONDS is not None and t_s >= cfg.MAX_SECONDS:
                break
            if t_s + 1e-6 < next_sample_time:
                if cfg.ADAPTIVE_SAMPLING:
                    pending.append((idx, t_s, frame))
                continue

            next_sample_time += 1.0 / output_fps
            sampled_frames += 1
            # Detect the anchor FIRST, but add intermediate observations BEFORE its tracking update.
            base_detections = read_plates(frame, alpr, scan_stats, cfg, current_focus_boxes(t_s))
            if cfg.ADAPTIVE_SAMPLING and (base_detections or t_s <= active_until):
                for extra_idx, extra_t, extra_frame in select_intermediate_frames(pending, last_sample_time, t_s):
                    extra_detections = read_plates(extra_frame, alpr, scan_stats, cfg, current_focus_boxes(extra_t))
                    record_frame_detections(extra_frame, extra_idx, extra_t, extra_detections)
                    extra_frames += 1
            pending.clear()
            mapping = record_frame_detections(frame, idx, t_s, base_detections)
            writer.write(annotate(frame, base_detections, mapping, tracks, t_s, cfg))
            last_sample_time = t_s
            pbar.update(1)
    finally:
        cap.release()
        writer.release()
        pbar.close()

    ob_columns = ["track_id","frame","time_s","x1","y1","x2","y2","width_px",
                  "raw_text","text","valid_format","detection_conf","ocr_conf","quality","sharpness","source"]
    pd.DataFrame(all_observations, columns=ob_columns).to_csv(OBS_CSV, index=False)
    summary_rows = []
    for tr in sorted(tracks.values(), key=lambda x: x.tid):
        decision = vote(tr.observations, cfg)
        best_obs = max(tr.observations, key=lambda o:o["quality"], default={})
        summary_rows.append({"plate_track_id":tr.tid,"plate":decision["plate"],
                             "status":decision["status"],"votes":decision["votes"],
                             "support_share":decision["support"],
                             "mean_ocr_conf":decision["ocr_conf"],
                             "first_seen_s":round(tr.first_t, 3),
                             "last_seen_s":round(tr.last_t, 3),"observations":len(tr.observations),
                             "best_width_px":best_obs.get("width_px",0),
                             "best_crop_file":tr.best_crop_file})
    summary_columns = ["plate_track_id","plate","status","votes","support_share","mean_ocr_conf",
                       "first_seen_s","last_seen_s","observations","best_width_px","best_crop_file"]
    summary = pd.DataFrame(summary_rows, columns=summary_columns)
    summary.to_csv(PLATES_CSV, index=False)

    # Convert mp4v to a broadly playable H.264 MP4 when FFmpeg is available.
    ffmpeg = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(RAW_VIDEO),
              "-c:v", "libx264", "-preset", "ultrafast", "-crf", "25",
              "-movflags", "+faststart", str(ANNOTATED_VIDEO)]
    try:
        converted = subprocess.run(ffmpeg, capture_output=True, text=True)
        converted_ok = converted.returncode == 0
    except FileNotFoundError:
        converted_ok = False
    if not converted_ok:
        print("FFmpeg encoding unavailable. Raw MP4 is still saved:", RAW_VIDEO)
        ANNOTATED_VIDEO = RAW_VIDEO

    elapsed = time.perf_counter() - start
    print(f"Finished in {elapsed:.1f}s | output frames: {sampled_frames} | extra analyzed frames: {extra_frames}")
    print(f"Detected-plate inference frames: {frames_with_plates} | OCR-empty detections: {ocr_empty}")
    print(f"Inference calls: {scan_stats['calls']} | image regions: {scan_stats['regions']}")
    print(f"Neural scan time: {scan_stats['seconds']:.1f}s | avg {1000 * scan_stats['seconds'] / max(1, scan_stats['calls']):.1f} ms/inference frame")
    print(f"Plate tracks: {len(tracks)} | Accepted: {(summary.status == 'accepted').sum()}")
    print("Outputs:", OUTPUT_DIR)

    return {"observations": OBS_CSV, "plates": PLATES_CSV, "video": ANNOTATED_VIDEO, "crops": CROPS_DIR}
