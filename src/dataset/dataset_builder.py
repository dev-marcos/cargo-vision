import os
import shutil
import random
from pathlib import Path
import yaml


class YoloDatasetBuilder:
    """Builds a YOLO-compatible dataset from raw images and label files.

    This class expects the following structure under raw_dataset_dir:

        raw_dataset_dir/
            images/
            labels/

    Label files are expected to be in YOLO format (.txt) with the same base
    filename as the corresponding image.
    """

    def __init__(self, config_path: str):
        self.config_path = config_path
        self._load_config()

    def _load_config(self) -> None:
        """Loads dataset configuration from YAML file."""
        with open(self.config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.raw_dataset_dir = Path(cfg["raw_dataset_dir"])
        self.yolo_dataset_dir = Path(cfg["yolo_dataset_dir"])
        self.train_split = float(cfg.get("train_split", 0.8))
        self.class_names = cfg["class_names"]

        self.images_raw = self.raw_dataset_dir / "images"
        self.labels_raw = self.raw_dataset_dir / "labels"

    def create_structure(self) -> None:
        """Creates the folder structure required by YOLO."""
        subdirs = [
            "images/train", "images/val",
            "labels/train", "labels/val",
        ]
        for s in subdirs:
            target_dir = self.yolo_dataset_dir / s
            target_dir.mkdir(parents=True, exist_ok=True)

        print(f"YOLO folder structure created at: {self.yolo_dataset_dir}")

    def split_and_copy(self) -> None:
        """Splits dataset into train/val subsets and copies files."""
        images = [
            f for f in os.listdir(self.images_raw)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        if not images:
            raise RuntimeError(f"No images found in {self.images_raw}")

        random.shuffle(images)
        split_index = int(len(images) * self.train_split)
        train_set = images[:split_index]
        val_set = images[split_index:]

        self._copy_subset(train_set, "train")
        self._copy_subset(val_set, "val")

        print(f"{len(train_set)} images in train set.")
        print(f"{len(val_set)} images in validation set.")

    def _copy_subset(self, image_list, subset_type: str) -> None:
        """Copies images and corresponding labels into subset folders."""
        for img_name in image_list:
            src_img = self.images_raw / img_name
            dst_img = self.yolo_dataset_dir / f"images/{subset_type}/{img_name}"

            label_name = f"{Path(img_name).stem}.txt"
            src_lbl = self.labels_raw / label_name
            dst_lbl = self.yolo_dataset_dir / f"labels/{subset_type}/{label_name}"

            if not src_lbl.exists():
                # In a strict pipeline you might raise an error instead.
                print(f"Warning: label file not found for image {img_name}, skipping.")
                continue

            shutil.copy(src_img, dst_img)
            shutil.copy(src_lbl, dst_lbl)

    def generate_data_yaml(self) -> None:
        """Generates the data.yaml file used by YOLO for training."""
        data_config = {
            "train": str((self.yolo_dataset_dir / "images/train").resolve()),
            "val": str((self.yolo_dataset_dir / "images/val").resolve()),
            "nc": len(self.class_names),
            "names": self.class_names,
        }

        out_path = self.yolo_dataset_dir / "data.yaml"
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.dump(data_config, f)

        print(f"data.yaml created at: {out_path}")


def main() -> None:
    """Entry point to build the YOLO dataset using configuration."""
    config_path = "configs/data_config.yaml"
    builder = YoloDatasetBuilder(config_path)
    builder.create_structure()
    builder.split_and_copy()
    builder.generate_data_yaml()


if __name__ == "__main__":
    main()
