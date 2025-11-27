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
                raise ValueError(
                    f"Annotation class id {cls_idx} in '{txt_path}' exceeds configured NUM_CLASSES={config.NUM_CLASSES}. "
                    "Update config.NUM_CLASSES/CLASS_NAMES to match the dataset."
                )
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


def compute_class_frequencies(label_dir: Path) -> List[int]:
    """Count per-class instances from YOLO txt annotations under label_dir."""
    counts = [0 for _ in range(config.NUM_CLASSES)]
    for txt in sorted(label_dir.glob("*.txt")):
        for line in txt.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            if 0 <= cls_id < config.NUM_CLASSES:
                counts[cls_id] += 1
    return counts


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
        self.to_tensor = transforms.ToTensor()
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
        boxes_orig = ann.boxes.clone()
        if self.augment and self.split == "train":
            img = self.color_jitter(img)
        img, boxes, scale, pad = self._letterbox(img, ann.boxes)
        img = self.to_tensor(img)

        labels = ann.labels.clone()
        target = self.build_yolo_target(boxes, labels)
        meta = {
            "image_id": image_id,
            "image_path": self._build_image_path(image_id),
            "orig_size": (orig_w, orig_h),
            "boxes": boxes,
            "boxes_orig": boxes_orig,
            "labels": labels,
            "letterbox_scale": scale,
            "letterbox_pad": pad,
        }
        return img, target, meta

    def _letterbox(self, img: Image.Image, boxes: torch.Tensor) -> Tuple[Image.Image, torch.Tensor, float, Tuple[int, int, int, int]]:
        """Resize with unchanged aspect ratio and pad to square, adjusting boxes accordingly.

        Returns padded image, normalized boxes in the padded space, the applied scale, and padding (l, t, r, b).
        """
        input_size = self.input_size
        orig_w, orig_h = img.size
        scale = min(input_size / orig_w, input_size / orig_h)
        new_w, new_h = int(round(orig_w * scale)), int(round(orig_h * scale))
        img_resized = img.resize((new_w, new_h), Image.BILINEAR)

        pad_w = input_size - new_w
        pad_h = input_size - new_h
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        img_padded = TF.pad(img_resized, padding=[pad_left, pad_top, pad_right, pad_bottom], fill=114)

        if boxes.numel() == 0:
            return img_padded, boxes, scale, (pad_left, pad_top, pad_right, pad_bottom)

        boxes_abs = boxes.clone()
        boxes_abs[:, 0] = boxes_abs[:, 0] * orig_w * scale + pad_left
        boxes_abs[:, 1] = boxes_abs[:, 1] * orig_h * scale + pad_top
        boxes_abs[:, 2] = boxes_abs[:, 2] * orig_w * scale
        boxes_abs[:, 3] = boxes_abs[:, 3] * orig_h * scale

        boxes_adj = torch.zeros_like(boxes_abs)
        boxes_adj[:, 0] = boxes_abs[:, 0] / input_size
        boxes_adj[:, 1] = boxes_abs[:, 1] / input_size
        boxes_adj[:, 2] = boxes_abs[:, 2] / input_size
        boxes_adj[:, 3] = boxes_abs[:, 3] / input_size
        return img_padded, boxes_adj, scale, (pad_left, pad_top, pad_right, pad_bottom)

    def _build_image_path(self, image_id: str) -> Path:
        jpg_path = self.image_dir / f"{image_id}.jpg"
        if jpg_path.exists():
            return jpg_path
        png_path = self.image_dir / f"{image_id}.png"
        return png_path

    @staticmethod
    def unletterbox_boxes(
        boxes_xyxy: torch.Tensor,
        scale: float,
        pad: Tuple[int, int, int, int],
        orig_size: Tuple[int, int],
        input_size: int = None,
    ) -> torch.Tensor:
        """Map boxes from padded square back to original image coordinates (absolute pixels)."""
        if boxes_xyxy.numel() == 0:
            return boxes_xyxy
        if input_size is None:
            input_size = config.INPUT_SIZE
        pad_left, pad_top, _, _ = pad
        orig_w, orig_h = orig_size
        boxes = boxes_xyxy.clone()
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] * input_size - pad_left) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] * input_size - pad_top) / scale
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, orig_w)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, orig_h)
        return boxes

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
            gi = int(gx)
            gj = int(gy)
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

