"""Editable project settings for ANPR."""
from pathlib import Path
from types import SimpleNamespace

# Change settings here, then run: python main.py run
PROJECT_ROOT = Path(__file__).resolve().parent

# Files
VIDEO_PATH = PROJECT_ROOT / "data/input/video.mp4"
OUTPUT_DIR = PROJECT_ROOT / "data/output"

# Pretrained neural models
PLATE_DETECTOR = "yolo-v9-t-640-license-plate-end2end"
PLATE_OCR = "cct-s-v2-global-model"
DETECTION_CONF = 0.25

# Video sampling
MAX_SECONDS = None  # e.g. 20 for a quick test; None = full video
SAMPLE_FPS = 6.0

# Distant / small plates
TILING = "auto"  # "off", "auto", "on", "grid"
TILE_OVERLAP = 0.20
GRID_TILE_SIZE = 960
MAX_GRID_TILES = 12
FOCUSED_RECHECK = True
FOCUS_WIDTH = 640
FOCUS_HEIGHT = 480
MAX_FOCUS_REGIONS = 3

# Fast vehicles: analyze extra original frames near detections
ADAPTIVE_SAMPLING = True
ACTIVE_SCAN_FPS = 12.0
ACTIVE_HOLD_S = 0.75

# Tracking and recognition confirmation
MIN_PLATE_WIDTH = 28
TRACK_MAX_GAP_S = 1.25
CONSENSUS_WINDOW_S = 6.0
MIN_INDEPENDENT_READS = 2
MIN_WEIGHTED_SUPPORT = 0.60
MIN_OCR_CONF = 0.50
REGION_FORMAT = "generic"  # "generic" or "AZ"

# Output options
KEEP_BEST_CROPS = True
MAX_DEBUG_TRACKS = 100


def load_config() -> SimpleNamespace:
    """Return an independent snapshot of these settings for a single run."""
    settings = {
        key: value for key, value in globals().items()
        if key.isupper() and key != "PROJECT_ROOT"
    }
    cfg = SimpleNamespace(**settings)
    cfg.VIDEO_PATH = Path(cfg.VIDEO_PATH).expanduser()
    cfg.OUTPUT_DIR = Path(cfg.OUTPUT_DIR).expanduser()
    if not cfg.VIDEO_PATH.is_absolute():
        cfg.VIDEO_PATH = PROJECT_ROOT / cfg.VIDEO_PATH
    if not cfg.OUTPUT_DIR.is_absolute():
        cfg.OUTPUT_DIR = PROJECT_ROOT / cfg.OUTPUT_DIR

    if cfg.TILING not in {"off", "auto", "on", "grid"}:
        raise ValueError("TILING must be off, auto, on or grid")
    if cfg.REGION_FORMAT not in {"generic", "AZ"}:
        raise ValueError("REGION_FORMAT must be generic or AZ")
    if cfg.SAMPLE_FPS <= 0 or not 0 < cfg.MIN_WEIGHTED_SUPPORT <= 1:
        raise ValueError("SAMPLE_FPS and MIN_WEIGHTED_SUPPORT must be positive")
    if not 0 <= cfg.TILE_OVERLAP < 0.5 or cfg.GRID_TILE_SIZE < 320 or cfg.MAX_GRID_TILES < 1:
        raise ValueError("Invalid tile configuration")
    if cfg.FOCUS_WIDTH < 64 or cfg.FOCUS_HEIGHT < 64 or cfg.MAX_FOCUS_REGIONS < 0:
        raise ValueError("Invalid focused crop configuration")
    if cfg.ACTIVE_SCAN_FPS < cfg.SAMPLE_FPS or cfg.ACTIVE_HOLD_S < 0:
        raise ValueError("Invalid adaptive sampling configuration")
    if cfg.MAX_SECONDS is not None and cfg.MAX_SECONDS <= 0:
        raise ValueError("MAX_SECONDS must be None or positive")
    return cfg
