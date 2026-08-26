"""
project_setup.py
-----------------
Step 1 of the pipeline: programmatically fetch a public, pre-labeled YOLO
instance-segmentation dataset and verify it is ready to train on.

Dataset: "package-seg" (single class: "package"), sourced from Roboflow
Universe and redistributed by Ultralytics for direct, no-auth download.
This is a real, non-trivial dataset (hundreds of images) -- not the 8-image
coco8 demo -- which satisfies the capstone's "Custom Data & Training"
requirement that the dataset be more than the lab warm-up.

What this script does:
  1. Points Ultralytics' dataset cache at a local ./datasets folder (keeps
     the whole capstone self-contained instead of writing to the user's
     global home directory).
  2. Triggers the download + extraction of package-seg via Ultralytics'
     own dataset-verification utility (the same one `model.train()` calls
     internally), so there is no separate/fragile URL-scraping code path.
  3. Verifies the resulting data.yaml resolves to real train/val(/test)
     image folders with a sane class list, and prints a short report.

Run:
    python project_setup.py
"""

from pathlib import Path

from ultralytics import settings
from ultralytics.data.utils import check_det_dataset

import config


def configure_dataset_cache() -> None:
    """Point Ultralytics at a project-local datasets/ folder instead of the
    global default, so `python project_setup.py` leaves everything inside
    this repo (easy to .gitignore, easy to zip up, easy to grade)."""
    settings.update({"datasets_dir": str(config.DATASETS_DIR)})
    print(f"[setup] Ultralytics datasets_dir -> {config.DATASETS_DIR}")


def download_and_verify_dataset() -> dict:
    """Download (if needed) and validate the package-seg dataset.

    check_det_dataset() is the same internal helper Ultralytics' own
    trainer calls before every run (it works for detect/segment/pose/obb
    alike, despite the "det" in its name) -- reusing it means we get
    identical download/caching/verification behavior to `model.train()`,
    instead of reinventing a downloader.
    """
    print(f"[setup] Resolving + downloading dataset: {config.DATA_YAML}")
    data_dict = check_det_dataset(config.DATA_YAML)
    return data_dict


def verify_structure(data_dict: dict) -> None:
    """Sanity-check the paths/classes Ultralytics resolved, and fail loudly
    (rather than silently) if something required is missing."""
    root = Path(data_dict["path"])
    print("\n[setup] ---- Dataset verification ----")
    print(f"  root path : {root}")
    print(f"  classes   : {data_dict['nc']} -> {data_dict['names']}")

    for split in ("train", "val", "test"):
        split_path = data_dict.get(split)
        if split_path is None:
            print(f"  {split:<5}     : not defined in data.yaml (skipped)")
            continue
        split_path = Path(split_path[0]) if isinstance(split_path, list) else Path(split_path)
        exists = split_path.exists()
        n_images = len(list(split_path.glob("*.*"))) if exists else 0
        status = "OK" if exists and n_images > 0 else "MISSING/EMPTY"
        print(f"  {split:<5}     : {split_path}  [{status}, {n_images} files]")

    assert data_dict.get("train") is not None, "data.yaml has no 'train' split -- cannot proceed."
    assert data_dict.get("val") is not None, "data.yaml has no 'val' split -- cannot proceed."
    print("[setup] Dataset structure looks good.\n")


def main() -> None:
    print("=" * 70)
    print("Smart Item Segmentation & Counting -- Dataset Setup")
    print("=" * 70)
    configure_dataset_cache()
    data_dict = download_and_verify_dataset()
    verify_structure(data_dict)
    print(f"[setup] Done. Train next with: python train_and_eval.py")


if __name__ == "__main__":
    main()
