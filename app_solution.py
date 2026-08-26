"""
app_solution.py
----------------
Step 3 of the pipeline: the "Real-World Solution & Video Analytics"
deliverable. A real OpenCV pipeline (capture -> process -> write) that runs
our fine-tuned segmentation model over video and uses
ultralytics.solutions.ObjectCounter to count instances crossing a region.

By default this:
  - loads our fine-tuned custom package-seg weights (falls back to the
    stock pretrained checkpoint with a warning if you haven't run
    train_and_eval.py yet, so the script never just crashes),
  - downloads a small public demo clip the first time it runs (so the
    pipeline is runnable with zero setup),
  - draws a horizontal counting line across the middle of the frame,
  - writes an annotated output video with live in/out/total counts.

IMPORTANT: the bundled default demo video shows a generic street scene, not
packages, so counts will be near-zero with the custom model -- that's
expected. Point --source at a video that actually contains the trained
class for a meaningful demo, e.g.:

    python app_solution.py --source path/to/warehouse_boxes.mp4

Run (defaults):
    python app_solution.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import solutions
from ultralytics.utils.downloads import safe_download

import config


def resolve_weights(weights_arg: str | None) -> str:
    """Prefer the custom-trained package-seg weights. Fall back to the
    stock pretrained checkpoint (still segmentation-capable, just not
    fine-tuned on "package") so the demo is never a hard failure just
    because train_and_eval.py hasn't been run yet."""
    if weights_arg:
        return weights_arg
    if config.FINAL_PT_PATH.exists():
        print(f"[app] Using fine-tuned custom weights: {config.FINAL_PT_PATH}")
        return str(config.FINAL_PT_PATH)
    print(f"[app] WARNING: {config.FINAL_PT_PATH} not found -- run "
          f"train_and_eval.py first for real 'package' counts. Falling back "
          f"to the stock '{config.BASE_SEG_MODEL}' checkpoint so the "
          f"pipeline still runs end-to-end.")
    return config.BASE_SEG_MODEL


def resolve_source(source_arg: str | None) -> str:
    """Use a user-supplied video if given, otherwise download the small
    public Ultralytics demo clip (used in their own solutions examples/CI)
    so the script works with zero setup."""
    if source_arg:
        return source_arg
    if not config.SAMPLE_VIDEO_PATH.exists():
        print(f"[app] Downloading sample video -> {config.SAMPLE_VIDEO_PATH}")
        safe_download(url=config.SAMPLE_VIDEO_URL, dir=config.ASSETS_DIR)
    return str(config.SAMPLE_VIDEO_PATH)


def default_region(width: int, height: int) -> list[tuple[int, int]]:
    """A horizontal line across the middle of the frame -- the simplest
    "objects crossing a boundary" counting region, easy to reason about
    for any video resolution."""
    y = height // 2
    return [(0, y), (width, y)]


def run_pipeline(weights: str, source: str, output: str,
                  conf: float, iou: float, classes: list[int] | None) -> None:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    region = default_region(w, h)
    print(f"[app] Video: {source} ({w}x{h} @ {fps:.1f}fps)")
    print(f"[app] Counting region (line): {region}")

    counter = solutions.ObjectCounter(
        show=False,
        region=region,
        model=weights,
        conf=conf,
        iou=iou,
        classes=classes,
    )

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = counter(frame)
        # Ultralytics `solutions` returns a SolutionResults object with
        # .plot_im on newer releases, but plain annotated frames on older
        # ones -- handle both so this script keeps working across versions.
        annotated = result.plot_im if hasattr(result, "plot_im") else result
        writer.write(annotated)
        frame_idx += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"[app] Processed {frame_idx} frames.")
    print(f"[app] Final counts -> in: {counter.in_count}, out: {counter.out_count}, "
          f"total crossings: {counter.in_count + counter.out_count}")
    print(f"[app] Annotated video written to: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=None,
                         help="Path to segmentation weights (.pt). "
                              "Defaults to weights/package_seg_best.pt if present.")
    parser.add_argument("--source", default=None,
                         help="Path/URL to a video. Defaults to a small public demo clip.")
    parser.add_argument("--output", default=str(config.OUTPUT_VIDEO_PATH),
                         help="Where to write the annotated output video.")
    parser.add_argument("--conf", type=float, default=config.CONF_THRESHOLD,
                         help="Confidence threshold for counting.")
    parser.add_argument("--iou", type=float, default=config.IOU_THRESHOLD,
                         help="NMS IoU threshold for counting.")
    parser.add_argument("--classes", type=int, nargs="*", default=None,
                         help="Optional list of class indices to count (default: all).")
    return parser.parse_args()


def main() -> None:
    print("=" * 70)
    print("Smart Item Segmentation & Counting -- Video Analytics")
    print("=" * 70)
    args = parse_args()
    weights = resolve_weights(args.weights)
    source = resolve_source(args.source)
    run_pipeline(weights, source, args.output, args.conf, args.iou, args.classes)


if __name__ == "__main__":
    main()
