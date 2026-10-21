"""Deterministic tests without downloading pretrained model weights."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import csv
import shutil
import unittest

import cv2
import numpy as np

from config import load_config
from anpr.detection import crop_regions, normalize_plate, vote, read_plates
from anpr.evaluation import write_label_template, evaluate
from anpr.tracking import associate
from anpr.pipeline import run, inspect_video


ROOT = Path(__file__).resolve().parents[1]


class ModelStub:
    def predict(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)
        n, labels, stats, centers = cv2.connectedComponentsWithStats(binary)
        detections = []
        for x, y, w, h, area in stats[1:]:
            if area < 200 or w < 50 or h < 10:
                continue
            box = SimpleNamespace(x1=x, y1=y, x2=x+w, y2=y+h)
            d = SimpleNamespace(bounding_box=box, confidence=0.93)
            o = SimpleNamespace(text='10-AB-123', confidence=[0.91]*7)
            detections.append(SimpleNamespace(detection=d, ocr=o))
        return detections


class RecognitionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()

    def test_config_paths_resolve_relative_to_project(self):
        self.assertEqual(self.cfg.VIDEO_PATH, ROOT / 'data/input/video.mp4')
        self.assertEqual(self.cfg.OUTPUT_DIR, ROOT / 'data/output')
        self.assertEqual(self.cfg.TILING, 'auto')

    def test_tiling_covers_distant_plate_and_deduplicates(self):
        self.cfg.TILING = 'grid'
        self.cfg.GRID_TILE_SIZE = 320
        self.cfg.MAX_GRID_TILES = 16
        frame = np.zeros((440, 520, 3), dtype=np.uint8)
        cv2.rectangle(frame, (230, 165), (320, 192), (255,255,255), -1)
        regions = crop_regions(frame, self.cfg)
        self.assertGreater(len(regions), 1)
        stats = {'calls':0, 'regions':0, 'seconds':0.0}
        dets = read_plates(frame, ModelStub(), stats, self.cfg)
        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0]['box'], (230,165,321,193))
        self.assertEqual(stats['calls'], 1)
        self.assertEqual(stats['regions'], len(regions))

    def test_vote_original_confidence_rules(self):
        def ob(t):return dict(text=t,ocr_conf=.9,width_px=100,quality=.9,detection_conf=.9)
        self.assertEqual(vote([ob('10AB123'),ob('10AB123'),ob('10AB128')],self.cfg)['status'],'accepted')
        self.assertEqual(vote([ob('10AB123')],self.cfg)['status'],'uncertain')
        self.assertEqual(vote([ob('')],self.cfg)['status'],'unreadable')
        self.cfg.REGION_FORMAT = 'AZ'
        self.assertEqual(vote([ob('AB12CDE')],self.cfg)['status'],'unreadable')
        self.assertEqual(normalize_plate('10-ab-123'), '10AB123')

    def test_association_keeps_track(self):
        tracks={}
        mapping, nxt=associate([{'box':(10,10,100,45)}],tracks,0,1,self.cfg)
        self.assertEqual(mapping, {0:1})
        mapping, nxt=associate([{'box':(15,10,105,45)}],tracks,.1,nxt,self.cfg)
        self.assertEqual(mapping,{0:1})
        self.assertEqual(len(tracks),1)

    def test_end_to_end_mock_video_and_labeling(self):
        with TemporaryDirectory() as d:
            d=Path(d)
            video = d/'synthetic.mp4'
            writer=cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'),10,(480,270))
            if not writer.isOpened():self.skipTest('OpenCV MP4 codec unavailable')
            for i in range(12):
                frame=np.zeros((270,480,3),dtype=np.uint8)
                cv2.rectangle(frame,(90+i*2,100),(200+i*2,135),(255,255,255),-1)
                writer.write(frame)
            writer.release()
            self.cfg.VIDEO_PATH=video
            self.cfg.OUTPUT_DIR=d/'outputs'
            self.cfg.TILING='grid'
            self.cfg.GRID_TILE_SIZE=320
            self.cfg.FOCUSED_RECHECK=True
            self.cfg.ADAPTIVE_SAMPLING=True
            self.cfg.ACTIVE_SCAN_FPS=10
            self.cfg.MAX_SECONDS=None
            with patch('anpr.pipeline.create_alpr',return_value=ModelStub()):
                fps,w,h,n=inspect_video(self.cfg)
                self.assertEqual((w,h,n),(480,270,12))
                results=run(self.cfg)
            self.assertTrue(results['video'].exists())
            self.assertTrue(results['observations'].exists())
            self.assertTrue(results['plates'].exists())
            with results['plates'].open(newline='') as f:
                rows=list(csv.DictReader(f))
            self.assertGreaterEqual(len(rows),1)
            self.assertTrue(any(r['plate']=='10AB123' and r['status']=='accepted' for r in rows))
            with results['observations'].open(newline='') as f:
                obs=list(csv.DictReader(f))
            self.assertGreater(len(obs),2)
            labels=d/'ground_truth.csv'
            count=write_label_template(results['plates'],labels)
            self.assertEqual(count,len(rows))
            with labels.open(newline='') as f:
                label_rows=list(csv.DictReader(f))
            label_rows[0]['true_plate']='10AB123'
            with labels.open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=label_rows[0].keys())
                writer.writeheader()
                writer.writerows(label_rows)
            metric=evaluate(labels)
            self.assertEqual(metric['exact_match_rate'],1)


if __name__=='__main__':unittest.main()
