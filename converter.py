"""
Converts BDD100K dataset annotations to YOLO-compatible format.

Outputs include:
- YOLO label .txt files
- YAML configuration (for Ultralytics)
- Darknet-style .names and .data files
"""

import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm
from config import IMAGES_ROOT, LABELS_ROOT, OUTPUT_DATASET_DIR, USE_ULTRALYTICS_FORMAT

# === CATEGORY FILTERING ===
# Categories like lane markings or drivable areas are not used for object detection
IGNORED_CATEGORIES = {'lane', 'drivable area', 'area/drivable', 'area/alternative'}

# === UTILITIES ===

def get_categories_from_json(json_paths):
    """Extract unique object categories from BDD100K label JSON files."""
    category_set = set()
    for json_path in json_paths.values():
        with open(json_path) as f:
            data = json.load(f)
        for frame in data:
            for obj in frame.get("labels", frame.get("objects", [])):
                cat = obj.get("category", "")
                if cat not in IGNORED_CATEGORIES:
                    category_set.add(cat)
    categories = sorted(category_set)
    return {cat: i for i, cat in enumerate(categories)}

def parse_labels(json_path, label_output_dir, categories, img_size):
    """Convert annotations to YOLO .txt format with normalized coordinates."""
    with open(json_path) as f:
        data = json.load(f)

    label_output_dir.mkdir(parents=True, exist_ok=True)

    for frame in tqdm(data, desc=f"Converting {json_path.name}"):
        img_name = Path(frame["name"]).stem
        label_path = label_output_dir / f"{img_name}.txt"

        with open(label_path, "w") as f_out:
            for obj in frame.get("labels", frame.get("objects", [])):
                cat = obj.get("category")
                if cat in IGNORED_CATEGORIES or cat not in categories or "box2d" not in obj:
                    continue

                box = obj["box2d"]
                x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
                x_c = (x1 + x2) / 2 / img_size[0]
                y_c = (y1 + y2) / 2 / img_size[1]
                w = (x2 - x1) / img_size[0]
                h = (y2 - y1) / img_size[1]

                f_out.write(f"{categories[cat]} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

def copy_images(src_img_dir, dst_img_dir):
    """Copy images from source to destination folder."""
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    for img_file in Path(src_img_dir).glob("*.jpg"):
        shutil.copy(img_file, dst_img_dir / img_file.name)

def generate_split_txt(img_dir, label_dir, output_txt):
    """Create a split .txt file listing valid image paths (with labels)."""
    with open(output_txt, "w") as f:
        for img_file in sorted(Path(img_dir).glob("*.jpg")):
            label_file = Path(label_dir) / f"{img_file.stem}.txt"
            if label_file.exists():
                f.write(str(img_file.resolve()) + "\n")

def generate_yaml(output_root, yaml_path, categories, use_ultralytics=True):
    """Generate YAML file for Ultralytics YOLO training."""
    if use_ultralytics:
        train_path = "images/train"
        val_path = "images/val"
        test_path = "images/test"
    else:
        train_path = "train"
        val_path = "val"
        test_path = "test"

    yaml_content = f"""path: {output_root}
        train: {train_path}
        val: {val_path}
        test: {test_path}
        names:
        """
    for k, v in sorted(categories.items(), key=lambda x: x[1]):
        yaml_content += f"  {v}: {k}\n"

    with open(yaml_path, "w") as f:
        f.write(yaml_content)

def generate_darknet_files(output_dir, split_txts, categories):
    """Generate .names and .data files for legacy YOLO training (Darknet)."""
    names_path = output_dir / "bdd100k.names"
    data_path = output_dir / "bdd100k.data"

    with open(names_path, "w") as f:
        for name in sorted(categories, key=lambda x: categories[x]):
            f.write(name + "\n")

    with open(data_path, "w") as f:
        f.write(f"classes = {len(categories)}\n")
        f.write(f"train = {split_txts['train']}\n")
        f.write(f"valid = {split_txts['val']}\n")
        f.write(f"test = {split_txts['test']}\n")
        f.write(f"names = {names_path}\n")
        f.write(f"backup = {output_dir}/backup\n")

# === MAIN WORKFLOW ===

def main():
    json_files = {
        "train": LABELS_ROOT / "bdd100k_labels_images_train.json",
        "val": LABELS_ROOT / "bdd100k_labels_images_val.json",
        "test": LABELS_ROOT / "bdd100k_labels_images_test.json"
    }

    categories = get_categories_from_json(json_files)
    print(f"\nDetected categories: {categories}")

    yolo_files_dir = OUTPUT_DATASET_DIR / "yolo_files"
    yolo_files_dir.mkdir(parents=True, exist_ok=True)
    split_txts = {}

    for split in ["train", "val", "test"]:
        src_imgs = IMAGES_ROOT / split

        # Set output folders depending on YOLO format
        if USE_ULTRALYTICS_FORMAT:
            img_dst = OUTPUT_DATASET_DIR / "images" / split
            lbl_dst = OUTPUT_DATASET_DIR / "labels" / split
        else:
            img_dst = OUTPUT_DATASET_DIR / split / "images"
            lbl_dst = OUTPUT_DATASET_DIR / split / "labels"

        copy_images(src_imgs, img_dst)
        parse_labels(json_files[split], lbl_dst, categories, img_size=(1280, 720))

        split_txt = yolo_files_dir / f"{split}.txt"
        generate_split_txt(img_dst, lbl_dst, split_txt)
        split_txts[split] = split_txt

    generate_darknet_files(yolo_files_dir, split_txts, categories)

    yaml_path = OUTPUT_DATASET_DIR / "bdd100k_ultralytics.yaml"
    generate_yaml(OUTPUT_DATASET_DIR.resolve(), yaml_path, categories, use_ultralytics=USE_ULTRALYTICS_FORMAT)

    print(f"\nYAML saved to: {yaml_path}")
    print(f"Darknet files saved to: {yolo_files_dir}")
    print("\nDataset conversion complete!")

if __name__ == "__main__":
    main()