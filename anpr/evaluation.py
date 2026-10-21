"""Ground-truth exact-plate evaluation; one manually labeled row per track."""
import csv
from pathlib import Path

from .detection import normalize_plate


def write_label_template(results_csv: Path, labels_csv: Path) -> int:
    with results_csv.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    labels_csv.parent.mkdir(parents=True, exist_ok=True)
    with labels_csv.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=["plate_track_id", "predicted_plate", "true_plate"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"plate_track_id": row["plate_track_id"],
                             "predicted_plate": row["plate"], "true_plate": ""})
    return len(rows)


def evaluate(labels_csv: Path) -> dict:
    with labels_csv.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    labeled = [r for r in rows if normalize_plate(r.get("true_plate", ""))]
    if not labeled:
        raise ValueError("Enter real plate numbers in the true_plate column before evaluating")
    correct = sum(normalize_plate(r["predicted_plate"]) == normalize_plate(r["true_plate"])
                  for r in labeled)
    return {"labeled_tracks": len(labeled), "exact_matches": correct,
            "exact_match_rate": correct / len(labeled)}
