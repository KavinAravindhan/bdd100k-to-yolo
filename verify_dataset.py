"""
Verifies image-label consistency in YOLO datasets.

Usage:
    python verify_dataset.py --dataset /path/to/dataset [--legacy]
"""

import argparse
from pathlib import Path

def verify_split(split_path_images: Path, split_path_labels: Path, split_name: str) -> bool:
    images = sorted(split_path_images.glob("*.jpg"))
    labels = sorted(split_path_labels.glob("*.txt"))

    print(f"\nVerifying split: {split_name}")
    print(f"Images: {len(images)} | Labels: {len(labels)}")

    images_set = {img.stem for img in images}
    labels_set = {lbl.stem for lbl in labels}

    missing_labels = images_set - labels_set
    missing_images = labels_set - images_set

    all_ok = True

    if missing_labels:
        all_ok = False
        print(f"{len(missing_labels)} images missing corresponding labels:")
        for name in sorted(missing_labels)[:10]:
            print(f"    - {name}.jpg -> [X] no {name}.txt")

    if missing_images:
        all_ok = False
        print(f"{len(missing_images)} labels missing corresponding images:")
        for name in sorted(missing_images)[:10]:
            print(f"    - {name}.txt -> [X] no {name}.jpg")

    if all_ok:
        print(f"{split_name} is consistent: all image-label pairs present.")

    return all_ok

def main():
    parser = argparse.ArgumentParser(description="Verify YOLO image-label consistency.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to YOLO dataset root")
    parser.add_argument("--legacy", action="store_true", help="Use legacy folder structure")
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    splits = ["train", "val", "test"]

    all_verified = True
    for split in splits:
        if args.legacy:
            img_dir = dataset_root / split / "images"
            lbl_dir = dataset_root / split / "labels"
        else:
            img_dir = dataset_root / "images" / split
            lbl_dir = dataset_root / "labels" / split

        if not img_dir.exists() or not lbl_dir.exists():
            print(f"Missing directories for split: {split}")
            all_verified = False
            continue

        if not verify_split(img_dir, lbl_dir, split):
            all_verified = False

    if all_verified:
        print("\nAll splits passed verification successfully!")
    else:
        print("\nSome issues found. Please check the logs above.")

if __name__ == "__main__":
    main()
