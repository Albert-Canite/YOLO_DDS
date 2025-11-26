import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF

import config
from utils.boxes import cxcywh_to_xyxy


@dataclass
class Annotation:
    boxes: torch.Tensor  # (N, 4) normalized cx, cy, w, h
    labels: torch.Tensor  # (N,)


def parse_yolo_annotation(txt_path: Path) -> Annotation:
    boxes = []
    labels = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id, cx, cy, w, h = parts
            cls_idx = int(cls_id)
            if cls_idx < 0 or cls_idx >= config.NUM_CLASSES:
                continue
            labels.append(cls_idx)
            boxes.append([float(cx), float(cy), float(w), float(h)])
    if not boxes:
        return Annotation(boxes=torch.zeros((0, 4), dtype=torch.float32), labels=torch.zeros((0,), dtype=torch.int64))
    return Annotation(boxes=torch.tensor(boxes, dtype=torch.float32), labels=torch.tensor(labels, dtype=torch.int64))


def load_split(split_file: Path) -> List[str]:
    with open(split_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def build_class_mapping() -> Dict[str, int]:
    # Use class names from config; assumes DDS annotations match
    return {name: idx for idx, name in enumerate(config.CLASS_NAMES)}


class DentalDDS(torch.utils.data.Dataset):
    def __init__(self, split: str = "train", augment: bool = True):
        super().__init__()
        split_alias = "valid" if split == "val" else split
        assert split_alias in {"train", "valid", "test"}
        self.split = split_alias
        self.image_dir = config.IMAGES_DIRS[self.split]
        self.label_dir = config.LABELS_DIRS[self.split]
        self.idx_to_class = {idx: name for idx, name in enumerate(config.CLASS_NAMES)}
        self.class_to_idx = {name: idx for idx, name in self.idx_to_class.items()}
        if not self.image_dir.exists() or not self.label_dir.exists():
            raise FileNotFoundError(
                "DDS data folders not found. Expected directories such as "
                f"'{self.image_dir}' and '{self.label_dir}'. Please ensure the dataset folder is named "
                "either 'Dental_OPG_XRAY_Dataset' or 'Dental OPG XRAY Dataset' and contains Augmented_Data/train|valid|test/"
                "with images/ and labels/."
            )

        split_file = {
            "train": config.TRAIN_SPLIT_FILE,
            "valid": config.VAL_SPLIT_FILE,
            "test": config.TEST_SPLIT_FILE,
        }[self.split]
        if split_file.exists():
            self.ids = load_split(split_file)
        else:
            # Fallback: derive IDs from label filenames within the split directory
            txt_files = sorted(self.label_dir.glob("*.txt"))
            if not txt_files:
                raise FileNotFoundError(
                    "No split file found and no label txt files detected. "
                    f"Checked for .txt files under: {self.label_dir}. "
                    "Ensure label files reside in Augmented_Data/<split>/labels and are named like '<image_id>.txt'."
                )
            self.ids = [p.stem for p in txt_files]
        self.input_size = config.INPUT_SIZE
        self.augment = augment
        self.color_jitter = transforms.ColorJitter(0.1, 0.1, 0.1, 0.05)

    def __len__(self) -> int:
        return len(self.ids)

    def _load_image(self, image_id: str) -> Image.Image:
        img_path = self.image_dir / f"{image_id}.jpg"
        if not img_path.exists():
            img_path = self.image_dir / f"{image_id}.png"
        if not img_path.exists():
            raise FileNotFoundError(f"Image {image_id} not found in {self.image_dir}")
        return Image.open(img_path).convert("RGB")

    def _load_annotation(self, image_id: str) -> Annotation:
        txt_path = self.label_dir / f"{image_id}.txt"
        if not txt_path.exists():
            raise FileNotFoundError(f"Annotation {txt_path} not found")
        return parse_yolo_annotation(txt_path)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        image_id = self.ids[idx]
        img = self._load_image(image_id)
        ann = self._load_annotation(image_id)

        orig_w, orig_h = img.size
        if self.augment and self.split == "train":
            img = self.color_jitter(img)

        img, boxes = self.letterbox_and_adjust(img, ann.boxes)
        labels = ann.labels.clone()

        target = self.build_yolo_target(boxes, labels)
        meta = {
            "image_id": image_id,
            "orig_size": (orig_w, orig_h),
            "boxes": boxes,
            "labels": labels,
        }
        return img, target, meta

    def letterbox_and_adjust(self, img: Image.Image, boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Resize with aspect ratio preservation (letterbox) to INPUT_SIZE and adjust normalized boxes accordingly.
        Boxes are expected in normalized cx, cy, w, h relative to original image size.
        """
        orig_w, orig_h = img.size
        scale = min(config.INPUT_SIZE / orig_w, config.INPUT_SIZE / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        pad_w = config.INPUT_SIZE - new_w
        pad_h = config.INPUT_SIZE - new_h
        pad_left = pad_w // 2
        pad_top = pad_h // 2

        # Resize and pad
        img_resized = TF.resize(img, (new_h, new_w))
        img_padded = TF.pad(img_resized, (pad_left, pad_top, pad_w - pad_left, pad_h - pad_top), fill=0)
        img_tensor = transforms.ToTensor()(img_padded)

        if boxes.numel() == 0:
            return img_tensor, boxes

        # Convert normalized boxes to absolute, apply scale/pad, then renormalize to INPUT_SIZE
        boxes_abs = boxes.clone()
        boxes_abs[:, 0] = boxes[:, 0] * orig_w * scale + pad_left  # cx
        boxes_abs[:, 1] = boxes[:, 1] * orig_h * scale + pad_top   # cy
        boxes_abs[:, 2] = boxes[:, 2] * orig_w * scale             # w
        boxes_abs[:, 3] = boxes[:, 3] * orig_h * scale             # h

        boxes_abs[:, 0] /= config.INPUT_SIZE
        boxes_abs[:, 1] /= config.INPUT_SIZE
        boxes_abs[:, 2] /= config.INPUT_SIZE
        boxes_abs[:, 3] /= config.INPUT_SIZE
        return img_tensor, boxes_abs

    def build_yolo_target(self, boxes: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        num_anchors = len(config.ANCHORS)
        grid_size = config.INPUT_SIZE // config.STRIDE
        target = torch.zeros((num_anchors, grid_size, grid_size, 5 + config.NUM_CLASSES), dtype=torch.float32)
        if boxes.numel() == 0:
            return target

        for box, label in zip(boxes, labels):
            cx, cy, w, h = box.tolist()
            gx = cx * grid_size
            gy = cy * grid_size
            gi = int(min(max(gx, 0), grid_size - 1))
            gj = int(min(max(gy, 0), grid_size - 1))
            cell_x = gx - gi
            cell_y = gy - gj
            best_anchor = self._select_best_anchor(w, h)
            anchor_w, anchor_h = config.ANCHORS[best_anchor]
            tw = math.log((w * config.INPUT_SIZE) / anchor_w + 1e-8)
            th = math.log((h * config.INPUT_SIZE) / anchor_h + 1e-8)
            target[best_anchor, gj, gi, 0] = cell_x
            target[best_anchor, gj, gi, 1] = cell_y
            target[best_anchor, gj, gi, 2] = tw
            target[best_anchor, gj, gi, 3] = th
            target[best_anchor, gj, gi, 4] = 1.0
            target[best_anchor, gj, gi, 5 + label] = 1.0
        return target

    def _select_best_anchor(self, w: float, h: float) -> int:
        anchor_wh = torch.tensor(config.ANCHORS, dtype=torch.float32) / config.INPUT_SIZE
        box_wh = torch.tensor([w, h])
        inter = torch.min(anchor_wh[:, 0], box_wh[0]) * torch.min(anchor_wh[:, 1], box_wh[1])
        anchor_area = anchor_wh[:, 0] * anchor_wh[:, 1]
        box_area = box_wh[0] * box_wh[1]
        union = anchor_area + box_area - inter
        iou = inter / (union + 1e-8)
        return int(torch.argmax(iou).item())


def collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    targets = torch.stack([b[1] for b in batch])
    metas = [b[2] for b in batch]
    return images, targets, metas


def visualize_sample(dataset: DentalDDS, idx: int = 0, save_path: Path = None):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    img, target, meta = dataset[idx]
    img_np = img.permute(1, 2, 0).numpy()
    grid_size = config.INPUT_SIZE // config.STRIDE
    fig, ax = plt.subplots(1)
    ax.imshow(img_np)

    anchors = torch.tensor(config.ANCHORS, dtype=torch.float32)
    for a in range(target.shape[0]):
        for j in range(target.shape[1]):
            for i in range(target.shape[2]):
                if target[a, j, i, 4] > 0:
                    tx, ty, tw, th = target[a, j, i, :4]
                    cx = (torch.sigmoid(torch.tensor(tx)) + i) / grid_size
                    cy = (torch.sigmoid(torch.tensor(ty)) + j) / grid_size
                    bw = anchors[a, 0] * torch.exp(torch.tensor(tw)) / config.INPUT_SIZE
                    bh = anchors[a, 1] * torch.exp(torch.tensor(th)) / config.INPUT_SIZE
                    xmin, ymin, xmax, ymax = cxcywh_to_xyxy(torch.tensor([[cx, cy, bw, bh]])).squeeze(0)
                    rect = patches.Rectangle((xmin * config.INPUT_SIZE, ymin * config.INPUT_SIZE), bw * config.INPUT_SIZE, bh * config.INPUT_SIZE, linewidth=2, edgecolor="r", facecolor="none")
                    ax.add_patch(rect)
    if save_path:
        plt.savefig(save_path)
    plt.close(fig)

