"""Lightweight notebook-original plate-track association and video annotations."""
from dataclasses import dataclass, field
import math
import cv2
from .detection import iou, vote

@dataclass
class PlateTrack:
    tid: int
    box: tuple
    last_t: float
    first_t: float
    seen: int = 0
    observations: list = field(default_factory=list)
    best_quality: float = -1.0
    best_crop_file: str = ""


def box_center(box):
    return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)


def association_cost(old, new):
    cx, cy = box_center(old)
    dx, dy = box_center(new)
    movement = math.hypot(cx-dx, cy-dy)
    width = max(1, old[2]-old[0], new[2]-new[0])
    height = max(1, old[3]-old[1], new[3]-new[1])
    norm_dist = movement / max(width, height)
    # Allow growth of an approaching plate but reject implausibly distant matches.
    if norm_dist > 2.4:
        return float("inf")
    return norm_dist - 0.6 * iou(old, new)


def associate(detections, tracks, now_s, next_track_id, cfg):
    # Greedy one-to-one nearest matches. Useful as a first baseline, not a full MOT tracker.
    active = [tr for tr in tracks.values() if now_s - tr.last_t <= cfg.TRACK_MAX_GAP_S]
    pairs = []
    for di, det in enumerate(detections):
        for tr in active:
            cost = association_cost(tr.box, det["box"])
            if math.isfinite(cost):
                pairs.append((cost, di, tr.tid))
    pairs.sort(key=lambda item: item[0])
    assigned_det, assigned_track = set(), set()
    mapping = {}
    for _, di, tid in pairs:
        if di not in assigned_det and tid not in assigned_track:
            mapping[di] = tid
            assigned_det.add(di)
            assigned_track.add(tid)
    for di, det in enumerate(detections):
        if di not in mapping:
            tid = next_track_id
            next_track_id += 1
            tracks[tid] = PlateTrack(tid, det["box"], now_s, now_s)
            mapping[di] = tid
        tr = tracks[mapping[di]]
        tr.box = det["box"]
        tr.last_t = now_s
        tr.seen += 1
    return mapping, next_track_id


def annotate(frame, detections, mapping, tracks, t_s, cfg):
    result = frame.copy()
    for di, det in enumerate(detections):
        track = tracks[mapping[di]]
        recent = [o for o in track.observations if t_s - o["time_s"] <= cfg.CONSENSUS_WINDOW_S]
        decision = vote(recent, cfg)
        x1,y1,x2,y2 = det["box"]
        color = (70, 195, 70) if decision["status"] == "accepted" else (0, 185, 255)
        cv2.rectangle(result, (x1,y1), (x2,y2), color, 2)
        label = f"P{track.tid} {decision['plate'] or 'READING'} [{decision['status']}]"
        font = 0.65 if frame.shape[1] >= 960 else 0.48
        cv2.putText(result, label, (max(4,x1), max(23,y1-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, font, (0,0,0), 4, cv2.LINE_AA)
        cv2.putText(result, label, (max(4,x1), max(23,y1-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, font, color, 2, cv2.LINE_AA)
    return result
