"""Plate model inference, overlapping crops, original-pixel quality, weighted consensus."""
from collections import Counter, defaultdict
from statistics import mean
import math
import re
import time

import cv2
import numpy as np

def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return intersection / max(1, area_a + area_b - intersection)


def tile_starts(length, size, overlap):
    """Cover the whole dimension with overlapping windows; include the final edge."""
    if length <= size:
        return [0]
    stride = max(1, round(size * (1.0 - overlap)))
    positions = list(range(0, length - size + 1, stride))
    if positions[-1] != length - size:
        positions.append(length - size)
    return positions


def crop_regions(frame, cfg, focus_boxes=()):
    """Yield original-pixel regions (x1, y1, x2, y2, source); no fake upscaling."""
    h, w = frame.shape[:2]
    regions = [(0, 0, w, h, "full")]
    elongated = max(h, w) / max(1, min(h, w)) > 1.45
    if cfg.TILING == "grid":
        tile_size = cfg.GRID_TILE_SIZE
        # Bound CPU work on 4K/8K frames rather than allocating hundreds of tiles.
        while True:
            xs = tile_starts(w, min(w, tile_size), cfg.TILE_OVERLAP)
            ys = tile_starts(h, min(h, tile_size), cfg.TILE_OVERLAP)
            if len(xs) * len(ys) <= cfg.MAX_GRID_TILES or tile_size >= max(w, h):
                break
            tile_size = int(tile_size * 1.5)
        tw, th = min(w, tile_size), min(h, tile_size)
        for y in ys:
            for x in xs:
                if (tw, th) != (w, h):
                    regions.append((x, y, x + tw, y + th, "grid"))
    elif cfg.TILING == "on" or (cfg.TILING == "auto" and elongated):
        if w >= h:
            middle, pad = w // 2, round(w * cfg.TILE_OVERLAP / 2)
            regions.extend([(0, 0, min(w, middle + pad), h, "half"),
                            (max(0, middle - pad), 0, w, h, "half")])
        else:
            middle, pad = h // 2, round(h * cfg.TILE_OVERLAP / 2)
            regions.extend([(0, 0, w, min(h, middle + pad), "half"),
                            (0, max(0, middle - pad), w, h, "half")])

    if cfg.FOCUSED_RECHECK and cfg.MAX_FOCUS_REGIONS:
        for b in list(focus_boxes)[:cfg.MAX_FOCUS_REGIONS]:
            cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
            fw, fh = min(cfg.FOCUS_WIDTH, w), min(cfg.FOCUS_HEIGHT, h)
            x1 = max(0, min(w - fw, round(cx - fw/2)))
            y1 = max(0, min(h - fh, round(cy - fh/2)))
            region = (x1, y1, x1 + fw, y1 + fh, "focus")
            # Avoid re-running inference on an already identical window.
            if not any(region[:4] == r[:4] for r in regions):
                regions.append(region)
    return regions


def mean_confidence(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return float(mean(value)) if len(value) else 0.0
    return float(value or 0.0)



def read_plates(frame, alpr, scan_stats, cfg, focus_boxes=()):
    """Run YOLO+OCR on original pixels, tiles and optional focus crops; merge duplicates."""
    H, W = frame.shape[:2]
    found = []
    begin = time.perf_counter()
    regions = crop_regions(frame, cfg, focus_boxes)
    for ax, ay, bx, by, source in regions:
        image = frame[ay:by, ax:bx]
        for prediction in alpr.predict(image):
            box = prediction.detection.bounding_box
            rect = (max(0, ax + int(box.x1)), max(0, ay + int(box.y1)),
                    min(W, ax + int(box.x2)), min(H, ay + int(box.y2)))
            if rect[2] <= rect[0] or rect[3] <= rect[1]:
                continue
            ocr = prediction.ocr
            found.append({
                "box": rect,
                "detection_conf": float(prediction.detection.confidence),
                "raw_text": (ocr.text or "") if ocr else "",
                "ocr_conf": mean_confidence(ocr.confidence) if ocr else 0.0,
                "source": source,
            })
    scan_stats["calls"] += 1
    scan_stats["regions"] += len(regions)
    scan_stats["seconds"] += time.perf_counter() - begin
    # Never count duplicate tile/focus readings from the SAME frame as independent votes.
    found.sort(key=lambda d: (d["detection_conf"] * (0.5 + 0.5*d["ocr_conf"])), reverse=True)
    unique = []
    for candidate in found:
        if not any(iou(candidate["box"], other["box"]) > 0.45 for other in unique):
            unique.append(candidate)
    return unique


def plate_quality(frame, box):
    x1, y1, x2, y2 = box
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0, 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    focus = float(cv2.Laplacian(gray, cv2.CV_64F).var()) if min(gray.shape) > 2 else 0.0
    # Quality measured on ORIGINAL pixels; zoom does not add independent evidence.
    relative_width = min(1.0, max(0, x2-x1) / 140.0)
    focus_score = min(1.0, math.log1p(focus) / math.log1p(180.0))
    return round(0.55*relative_width + 0.45*focus_score, 4), round(focus, 1)


def normalize_plate(text):
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


def valid_plate(text, cfg):
    if cfg.REGION_FORMAT == "AZ":
        return bool(re.fullmatch(r"[0-9]{2}[A-Z]{2}[0-9]{3}", text))
    return bool(re.fullmatch(r"[A-Z0-9]{5,10}", text))


def vote(observations, cfg):
    """Vote on observed full strings only; do not invent or auto-correct characters."""
    eligible = [o for o in observations if o["text"] and valid_plate(o["text"], cfg)
                and o["ocr_conf"] >= cfg.MIN_OCR_CONF and o["width_px"] >= cfg.MIN_PLATE_WIDTH]
    if not eligible:
        return {"plate": "", "status": "unreadable", "votes": 0, "support": 0.0, "ocr_conf": 0.0}
    weights = defaultdict(float)
    counts = Counter()
    for ob in eligible:
        weight = ob["ocr_conf"] * (0.4 + 0.6 * ob["quality"]) * max(0.2, ob["detection_conf"])
        weights[ob["text"]] += weight
        counts[ob["text"]] += 1
    best = max(weights, key=lambda p: (weights[p], counts[p]))
    support = weights[best] / max(0.001, sum(weights.values()))
    winner = [o for o in eligible if o["text"] == best]
    ocr_conf = sum(o["ocr_conf"] for o in winner) / len(winner)
    accepted = counts[best] >= cfg.MIN_INDEPENDENT_READS and support >= cfg.MIN_WEIGHTED_SUPPORT
    return {"plate": best, "status": "accepted" if accepted else "uncertain",
            "votes": counts[best], "support": round(support, 3), "ocr_conf": round(ocr_conf, 3)}
