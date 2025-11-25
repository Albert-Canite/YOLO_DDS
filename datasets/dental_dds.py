import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any

import torch
from PIL import Image
from torchvision import transforms

import config
from utils.boxes import xyxy_to_cxcywh, cxcywh_to_xyxy


@dataclass
class Annotation:
    boxes: torch.Tensor  # (N, 4) normalized cx, cy, w, h
    labels: torch.Tensor  # (N,)


def parse_voc_annotation(xml_path: Path, class_to_idx: Dict[str, int]) -> Annotation:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    labels = []
    size = root.find("size")
    if size is None:
        raise ValueError(f"Annotation {xml_path} missing size tag")
    img_w = float(size.find("width").text)
    img_h = float(size.find("height").text)
    for obj in root.iter("object"):
        name = obj.find("name").text
        if name not in class_to_idx:
            continue
        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)
        cx, cy, w, h = xyxy_to_cxcywh(torch.tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32))[0].tolist()
        boxes.append([cx / img_w, cy / img_h, w / img_w, h / img_h])
        labels.append(class_to_idx[name])
    if not boxes:
        return Annotation(boxes=torch.zeros((0, 4), dtype=torch.float32), labels=torch.zeros((0,), dtype=torch.int64))
    return Annotation(boxes=torch.tensor(boxes, dtype=torch.float32), labels=torch.tensor(labels, dtype=torch.int64))


def load_split(split_file: Path) -> List[str]:
    if not split_file.exists():
        raise FileNotFoundError(f"Split file {split_file} not found. Ensure DDS splits are present.")
    with open(split_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def build_class_mapping() -> Dict[str, int]:
    # Use class names from config; assumes DDS annotations match
    return {name: idx for idx, name in enumerate(config.CLASS_NAMES)}


class DentalDDS(torch.utils.data.Dataset):
    def __init__(self, split: str = "train", augment: bool = True):
        super().__init__()
        assert split in {"train", "val", "test"}
        self.split = split
        self.image_dir = config.IMAGES_DIR / split
        self.ann_dir = config.ANNOTATIONS_DIR / split
        self.class_to_idx = build_class_mapping()
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        split_file = {
            "train": config.TRAIN_SPLIT_FILE,
            "val": config.VAL_SPLIT_FILE,
            "test": config.TEST_SPLIT_FILE,
        }[split]
        self.ids = load_split(split_file)
        self.input_size = config.INPUT_SIZE
        self.augment = augment
        self.to_tensor = transforms.Compose(
            [
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
            ]
        )
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
        xml_path = self.ann_dir / f"{image_id}.xml"
        if not xml_path.exists():
            raise FileNotFoundError(f"Annotation {xml_path} not found")
        return parse_voc_annotation(xml_path, self.class_to_idx)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        image_id = self.ids[idx]
        img = self._load_image(image_id)
        ann = self._load_annotation(image_id)

        orig_w, orig_h = img.size
        if self.augment and self.split == "train":
            img = self.color_jitter(img)
        img = self.to_tensor(img)

        boxes = ann.boxes.clone()
        labels = ann.labels.clone()

        if boxes.numel() > 0:
            # Normalize boxes already 0-1; adjust for resize with same ratio
            boxes = boxes
        target = self.build_yolo_target(boxes, labels)
        meta = {
            "image_id": image_id,
            "orig_size": (orig_w, orig_h),
            "boxes": boxes,
            "labels": labels,
        }
        return img, target, meta

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

