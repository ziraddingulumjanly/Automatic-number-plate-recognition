"""Colorado ANPR command-line entrypoint."""
import argparse
from pathlib import Path

from config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description="Colorado ANPR — detect, read, and confirm license plates")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inspect", help="Show input video information without downloading models")
    sub.add_parser("run", help="Run YOLOv9 + CCT OCR and export video/CSVs")
    snap = sub.add_parser("snapshot", help="Inspect one frame for detected plates")
    snap.add_argument("--second", type=float, default=2.0)
    snap.add_argument("--output", type=Path, default=None)
    label = sub.add_parser("labels", help="Generate a CSV for manual ground-truth labeling")
    label.add_argument("--output", type=Path, default=None)
    assess = sub.add_parser("evaluate", help="Evaluate manually completed labels CSV")
    assess.add_argument("--labels", type=Path, default=None)
    args = parser.parse_args(argv)
    cfg = load_config()
    if args.command == "inspect":
        from anpr.pipeline import inspect_video
        fps, w, h, count = inspect_video(cfg)
        print(f"Video: {cfg.VIDEO_PATH} | {w}x{h} | {fps:.2f} FPS | {count} frames")
    elif args.command == "run":
        from anpr.pipeline import run
        for name, path in run(cfg).items():
            print(f"{name}: {path}")
    elif args.command == "snapshot":
        import cv2
        from anpr.pipeline import inspect_video, create_alpr
        from anpr.detection import read_plates
        inspect_video(cfg)
        cap = cv2.VideoCapture(str(cfg.VIDEO_PATH))
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0, args.second) * 1000)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError("Snapshot time is outside the video")
        detector = create_alpr(cfg)
        detections = read_plates(frame, detector, {"calls": 0, "regions": 0, "seconds": 0.0}, cfg)
        output = args.output or cfg.OUTPUT_DIR / "snapshot.jpg"
        output.parent.mkdir(parents=True, exist_ok=True)
        for item in detections:
            x1, y1, x2, y2 = item["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 195, 70), 2)
            print(item)
        if not cv2.imwrite(str(output), frame):
            raise RuntimeError(f"Could not save snapshot: {output}")
        print(f"Found {len(detections)} plates | Saved {output}")
    elif args.command in {"labels", "evaluate"}:
        from anpr.evaluation import write_label_template, evaluate
        labels_csv = args.output if args.command == "labels" else args.labels
        labels_csv = labels_csv or cfg.OUTPUT_DIR / "ground_truth.csv"
        if args.command == "labels":
            count = write_label_template(cfg.OUTPUT_DIR / "recognized_plates.csv", labels_csv)
            print(f"Created {count} labeling rows: {labels_csv}")
        else:
            result = evaluate(labels_csv)
            print(f"Exact full-plate matches: {result['exact_matches']}/{result['labeled_tracks']} "
                  f"({result['exact_match_rate']:.1%})")


if __name__ == "__main__":
    main()
