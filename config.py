"""
config.py
---------
Single source of truth for paths and constants shared by every script in this
project (project_setup.py, train_and_eval.py, app_solution.py).

Keeping these in one place means:
  - project_setup.py downloads the dataset to the same place train_and_eval.py reads it from.
  - train_and_eval.py always saves the final weights to the same place app_solution.py loads them from.
  - Nobody has to remember a path or hunt through Ultralytics' auto-generated `runs/segment/trainN` folders.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

# Where Ultralytics downloads/looks for datasets (kept local to the project
# instead of the global ~/datasets folder so the whole capstone is portable).
DATASETS_DIR = PROJECT_ROOT / "datasets"

# Folder we copy the *final* trained weights into, so app_solution.py always
# has one stable path to load regardless of which "runs/segment/trainN" the
# training happened to land in.
WEIGHTS_DIR = PROJECT_ROOT / "weights"

# Where we drop small human-readable evidence of each run (metrics, run
# summary) for the "Documentation & Evidence of Execution" rubric item.
REPORTS_DIR = PROJECT_ROOT / "reports"

# Local sample video + pipeline output for the real-time analytics deliverable.
ASSETS_DIR = PROJECT_ROOT / "assets"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

for _d in (DATASETS_DIR, WEIGHTS_DIR, REPORTS_DIR, ASSETS_DIR, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
# "package-seg" is a real, public, pre-labeled YOLO instance-segmentation
# dataset (single class: "package", sourced from Roboflow Universe and
# redistributed by Ultralytics) — NOT the coco8 demo. It auto-downloads the
# first time it's referenced. Its subject (boxes/packages) also matches our
# "Smart Item Segmentation & Counting" use case, so the same model we train
# here is directly usable for the video-counting deliverable.
# Docs: https://docs.ultralytics.com/datasets/segment/package-seg/
DATA_YAML = "package-seg.yaml"

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# yolo11n-seg is the smallest, fastest official Ultralytics segmentation
# checkpoint -> fast enough to fine-tune for 20 epochs on a laptop/CPU within
# a tight deadline. Swap for "yolo26n-seg.pt" if you have the newer YOLO26
# weights available and want a stronger baseline.
BASE_SEG_MODEL = "yolo11n-seg.pt"

# Stable names for the artifacts produced by train_and_eval.py.
FINAL_PT_NAME = "package_seg_best.pt"
FINAL_ONNX_NAME = "package_seg_best.onnx"
FINAL_PT_PATH = WEIGHTS_DIR / FINAL_PT_NAME
FINAL_ONNX_PATH = WEIGHTS_DIR / FINAL_ONNX_NAME

# ---------------------------------------------------------------------------
# Training knobs (touched deliberately, see README "Training decisions")
# ---------------------------------------------------------------------------
# NOTE: imgsz=320 (rather than the 640 the base checkpoint was pretrained at)
# and epochs=10 (rather than 20) were chosen after timing the first few
# training iterations on this machine's CPU (no GPU): at imgsz=640 a single
# epoch over 1920 images was measured at ~20+ minutes/epoch (~7+ hours for
# the full run), which does not fit a same-day deadline. Halving imgsz cuts
# the per-image compute ~4x (pixel count scales quadratically with side
# length), and this is a documented, deliberate trade-off -- see README
# "Training decisions" for the accuracy implication and how to restore
# imgsz=640/epochs=20 on faster hardware or a GPU.
EPOCHS = 10
IMGSZ = 320
BATCH = 16
PATIENCE = 5            # early-stop if val loss doesn't improve for N epochs
FREEZE_BACKBONE = 10    # freeze first 10 layers (transfer learning) — see README

# ---------------------------------------------------------------------------
# Inference / deployment thresholds (see README "Threshold decisions")
# ---------------------------------------------------------------------------
CONF_THRESHOLD = 0.35   # confidence cutoff used at inference/counting time
IOU_THRESHOLD = 0.5     # NMS IoU cutoff used at inference/counting time

# ---------------------------------------------------------------------------
# Video analytics (app_solution.py)
# ---------------------------------------------------------------------------
# Public, small, license-free demo clip officially used by Ultralytics for
# its own `solutions` examples/CI. Used as a default so the pipeline is
# runnable out of the box with zero setup. It does NOT contain "package"
# objects, so counts will be near-zero with our custom model — that's
# expected and documented in the README. Point --source at a real
# package/box video for a meaningful count.
SAMPLE_VIDEO_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/solutions_ci_demo.mp4"
SAMPLE_VIDEO_PATH = ASSETS_DIR / "solutions_ci_demo.mp4"
OUTPUT_VIDEO_PATH = OUTPUTS_DIR / "counted_output.avi"
