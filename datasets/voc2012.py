import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF

import config
import config_voc2012
from utils.boxes import cxcywh_to_xyxy


@dataclass
class Annotation:
    boxes: torch.Tensor  # (N, 4) normalized cx, cy, w, h
    labels: torch.Tensor  # (N,)


VOC_CLASS_TO_IDX = {name: idx for idx, name in enumerate(config_voc2012.CLASS_NAMES)}


def _is_valid_box(cx: float, cy: float, w: float, h: float) -> bool:
    if w <= 0 or h <= 0:
        return False
    if not (0 <= cx <= 1 and 0 <= cy <= 1):
        return False
    if not (0 < w <= 1 and 0 < h <= 1):
        return False
    return True


def _parse_annotation(xml_path: Path) -> Annotation:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    width = float(size.findtext("width"))
    height = float(size.findtext("height"))

    boxes: List[List[float]] = []
    labels: List[int] = []
    dropped = 0
    for obj in root.findall("object"):
        difficult = obj.findtext("difficult")
        if difficult is not None and difficult.strip() == "1":
            continue  # skip difficult objects by default
        name = obj.findtext("name")
        if name not in VOC_CLASS_TO_IDX:
            continue
        bndbox = obj.find("bndbox")
        xmin = float(bndbox.findtext("xmin"))
        ymin = float(bndbox.findtext("ymin"))
        xmax = float(bndbox.findtext("xmax"))
        ymax = float(bndbox.findtext("ymax"))
        cx = (xmin + xmax) / 2.0 / width
        cy = (ymin + ymax) / 2.0 / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height
        if not _is_valid_box(cx, cy, w, h):
            dropped += 1
            continue
        boxes.append([cx, cy, w, h])
        labels.append(VOC_CLASS_TO_IDX[name])
    if dropped > 0:
        print(f"[WARN] Dropped {dropped} invalid boxes in {xml_path.name}")
    if not boxes:
        return Annotation(boxes=torch.zeros((0, 4), dtype=torch.float32), labels=torch.zeros((0,), dtype=torch.int64))
    return Annotation(boxes=torch.tensor(boxes, dtype=torch.float32), labels=torch.tensor(labels, dtype=torch.int64))


def load_split(split_file: Path) -> List[str]:
    with open(split_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def compute_class_frequencies(ann_dir: Path) -> List[int]:
    counts = [0 for _ in range(config_voc2012.NUM_CLASSES)]
    for xml in sorted(ann_dir.glob("*.xml")):
        ann = _parse_annotation(xml)
        for cls_idx in ann.labels.tolist():
            counts[cls_idx] += 1
    return counts


def gather_label_stats(ann_dir: Path) -> Dict[str, Any]:
    per_class = [0 for _ in range(config_voc2012.NUM_CLASSES)]
    per_class_images = [0 for _ in range(config_voc2012.NUM_CLASSES)]
    per_image_counts = []
    invalid_boxes = 0
    for xml in sorted(ann_dir.glob("*.xml")):
        ann = _parse_annotation(xml)
        valid_boxes = ann.boxes.shape[0]
        per_image_counts.append(valid_boxes)
        seen = set()
        for cls_idx in ann.labels.tolist():
            per_class[cls_idx] += 1
            seen.add(cls_idx)
        for cls_idx in seen:
            per_class_images[cls_idx] += 1
        invalid_boxes += max(0, valid_boxes - ann.boxes.shape[0])

    total_boxes = sum(per_class)
    nonzero_counts = [c for c in per_class if c > 0]
    imbalance_ratio = (max(nonzero_counts) / min(nonzero_counts)) if len(nonzero_counts) > 1 else 1.0
    presence_ratio = [
        (count / len(per_image_counts)) if per_image_counts else 0.0 for count in per_class_images
    ]
    missing_classes = [idx for idx, count in enumerate(per_class) if count == 0]
    return {
        "annotation_dir": str(ann_dir),
        "num_images": len(per_image_counts),
        "images_with_no_boxes": int(sum(1 for c in per_image_counts if c == 0)),
        "total_valid_boxes": total_boxes,
        "invalid_boxes": invalid_boxes,
        "per_class_counts": per_class,
        "per_class_image_freq": per_class_images,
        "class_presence_ratio": presence_ratio,
        "missing_classes": missing_classes,
        "class_imbalance_ratio": imbalance_ratio,
        "boxes_per_image": {
            "min": min(per_image_counts) if per_image_counts else 0,
            "max": max(per_image_counts) if per_image_counts else 0,
            "mean": float(sum(per_image_counts) / len(per_image_counts)) if per_image_counts else 0.0,
        },
    }


class PascalVOC2012(torch.utils.data.Dataset):
    def __init__(self, split: str = "train", augment: bool = True):
        super().__init__()
        split_alias = "val" if split == "valid" else split
        assert split_alias in {"train", "val", "test"}
        self.split = split_alias
        self.dataset_root = config_voc2012.DATASET_DIRS[self.split]
        self.images_dir = config_voc2012.IMAGES_DIRS[self.split]
        self.ann_dir = config_voc2012.ANNOTATION_DIRS[self.split]
        self.split_file = config_voc2012.SPLIT_FILES[self.split]

        if not self.images_dir.exists():
            raise FileNotFoundError(
                f"VOC2012 images folder not found. Expected '{self.images_dir}'. "
                "Set VOC_ROOT or update config_voc2012 paths to point to your dataset."
            )

        self.require_annotations = self.split != "test"
        self.has_annotations = self.ann_dir.exists()
        if self.require_annotations and not self.has_annotations:
            raise FileNotFoundError(
                f"VOC2012 annotation folder not found. Expected '{self.ann_dir}'. "
                "Set VOC_ROOT or update config_voc2012 paths to point to your dataset."
            )
        if not self.require_annotations and not self.has_annotations:
            print(
                f"[WARN] Annotations for split '{self.split}' not found at {self.ann_dir}. "
                "Proceeding without ground-truth boxes (inference-only)."
            )

        if self.split_file.exists():
            self.ids = load_split(self.split_file)
        elif self.split == "test":
            # Allow missing test.txt: fall back to all images in JPEGImages
            image_stems = {p.stem for p in self.images_dir.glob("*.jpg")}
            image_stems.update({p.stem for p in self.images_dir.glob("*.png")})
            if not image_stems:
                raise FileNotFoundError(
                    f"No images found under {self.images_dir}; cannot build test split list."
                )
            self.ids = sorted(image_stems)
            print(
                f"[WARN] Split file '{self.split_file}' missing; using all {len(self.ids)} images in {self.images_dir}"
            )
        else:
            raise FileNotFoundError(
                f"Split file '{self.split_file}' missing. Ensure ImageSets/Main/{{train,val}}.txt are present."
            )

        self.input_size = config_voc2012.INPUT_SIZE
        self.augment = augment
        self.to_tensor = transforms.ToTensor()
        self.color_jitter = transforms.ColorJitter(0.1, 0.1, 0.1, 0.05)

    def __len__(self) -> int:
        return len(self.ids)

    def _load_image(self, image_id: str) -> Image.Image:
        img_path = self.images_dir / f"{image_id}.jpg"
        if not img_path.exists():
            img_path = self.images_dir / f"{image_id}.png"
        if not img_path.exists():
            raise FileNotFoundError(f"Image {image_id} not found in {self.images_dir}")
        return Image.open(img_path).convert("RGB")

    def _load_annotation(self, image_id: str) -> Annotation:
        xml_path = self.ann_dir / f"{image_id}.xml"
        if not self.has_annotations:
            return Annotation(
                boxes=torch.zeros((0, 4), dtype=torch.float32),
                labels=torch.zeros((0,), dtype=torch.int64),
            )
        if not xml_path.exists():
            if self.split == "test":
                print(f"[WARN] Missing annotation for test image {image_id} at {xml_path}; skipping GT.")
                return Annotation(
                    boxes=torch.zeros((0, 4), dtype=torch.float32),
                    labels=torch.zeros((0,), dtype=torch.int64),
                )
            raise FileNotFoundError(f"Annotation {xml_path} not found")
        return _parse_annotation(xml_path)

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
        jpg_path = self.images_dir / f"{image_id}.jpg"
        if jpg_path.exists():
            return jpg_path
        png_path = self.images_dir / f"{image_id}.png"
        return png_path

    @staticmethod
    def unletterbox_boxes(
        boxes_xyxy: torch.Tensor,
        scale: float,
        pad: Tuple[int, int, int, int],
        orig_size: Tuple[int, int],
        input_size: int = None,
    ) -> torch.Tensor:
        if boxes_xyxy.numel() == 0:
            return boxes_xyxy
        if input_size is None:
            input_size = config_voc2012.INPUT_SIZE
        pad_left, pad_top, _, _ = pad
        orig_w, orig_h = orig_size
        boxes = boxes_xyxy.clone()
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] * input_size - pad_left) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] * input_size - pad_top) / scale
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, orig_w)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, orig_h)
        return boxes

    def build_yolo_target(self, boxes: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        num_anchors = len(config_voc2012.ANCHORS)
        grid_size = config_voc2012.INPUT_SIZE // config_voc2012.STRIDE
        target = torch.zeros(
            (num_anchors, grid_size, grid_size, 5 + config_voc2012.NUM_CLASSES), dtype=torch.float32
        )
        if boxes.numel() == 0:
            return target

        filled = torch.zeros((num_anchors, grid_size, grid_size), dtype=torch.bool)
        for box, label in zip(boxes, labels):
            cx, cy, w, h = box.tolist()
            gx = cx * grid_size
            gy = cy * grid_size
            gi = max(min(int(gx), grid_size - 1), 0)
            gj = max(min(int(gy), grid_size - 1), 0)
            cell_x = gx - gi
            cell_y = gy - gj

            anchor_wh = torch.tensor(config_voc2012.ANCHORS, dtype=torch.float32) / config_voc2012.INPUT_SIZE
            box_wh = torch.tensor([w, h])
            inter = torch.min(anchor_wh[:, 0], box_wh[0]) * torch.min(anchor_wh[:, 1], box_wh[1])
            anchor_area = anchor_wh[:, 0] * anchor_wh[:, 1]
            box_area = box_wh[0] * box_wh[1]
            iou = inter / (anchor_area + box_area - inter + 1e-8)
            best_anchors = torch.argsort(iou, descending=True)

            chosen_anchor = None
            for a_idx in best_anchors:
                if not filled[a_idx, gj, gi]:
                    chosen_anchor = int(a_idx)
                    break
            if chosen_anchor is None:
                chosen_anchor = int(best_anchors[0])

            anchor_w, anchor_h = config_voc2012.ANCHORS[chosen_anchor]
            tw = math.log((w * config_voc2012.INPUT_SIZE) / anchor_w + 1e-8)
            th = math.log((h * config_voc2012.INPUT_SIZE) / anchor_h + 1e-8)
            target[chosen_anchor, gj, gi, 0] = cell_x
            target[chosen_anchor, gj, gi, 1] = cell_y
            target[chosen_anchor, gj, gi, 2] = tw
            target[chosen_anchor, gj, gi, 3] = th
            target[chosen_anchor, gj, gi, 4] = 1.0
            target[chosen_anchor, gj, gi, 5 + label] = 1.0
            filled[chosen_anchor, gj, gi] = True
        return target


def collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    targets = torch.stack([b[1] for b in batch])
    metas = [b[2] for b in batch]
    return images, targets, metas


__all__ = [
    "PascalVOC2012",
    "collate_fn",
    "compute_class_frequencies",
    "gather_label_stats",
]
