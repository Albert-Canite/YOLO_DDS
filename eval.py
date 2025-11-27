import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from PIL import ImageDraw, ImageFont

import config
from datasets.dental_dds import DentalDDS, collate_fn
from models.yolotiny import YOLOTiny
from utils.metrics import Evaluator
from utils.boxes import cxcywh_to_xyxy


def _draw_boxes(pil_img, boxes, labels, scores=None, color="red"):
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.tolist()
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label_text = config.CLASS_NAMES[int(labels[i])] if int(labels[i]) < len(config.CLASS_NAMES) else str(int(labels[i]))
        if scores is not None:
            label_text += f" {scores[i]:.2f}"
        draw.text((x1, y1), label_text, fill=color, font=font)
    return pil_img


def load_checkpoint(model: YOLOTiny, checkpoint_path: Path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])


def evaluate(split: str, checkpoint: Path, save_vis: Path = None, conf_threshold: float = None):
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    dataset = DentalDDS(split=split, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
    )
    model = YOLOTiny().to(device)
    load_checkpoint(model, checkpoint, device)
    model.eval()
    evaluator = Evaluator(iou_threshold=0.5)
    vis_dir = None
    if save_vis is not None:
        vis_dir = Path(save_vis)
        vis_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for images, targets, metas in loader:
            images = images.to(device)
            preds = model(images)
            decoded = model.decode(preds, conf_threshold=conf_threshold)
            evaluator.add_batch(decoded, metas)
            if vis_dir is not None:
                for i, pred in enumerate(decoded):
                    orig_img = Image.open(metas[i]["image_path"]).convert("RGB")
                    gt_boxes = cxcywh_to_xyxy(metas[i]["boxes_orig"])
                    orig_w, orig_h = metas[i]["orig_size"]
                    gt_boxes[:, 0::2] *= orig_w
                    gt_boxes[:, 1::2] *= orig_h
                    _draw_boxes(orig_img, gt_boxes, metas[i]["labels"], color="green")
                    if pred.numel() > 0:
                        boxes = DentalDDS.unletterbox_boxes(
                            pred[:, :4],
                            scale=metas[i]["letterbox_scale"],
                            pad=metas[i]["letterbox_pad"],
                            orig_size=metas[i]["orig_size"],
                        )
                        scores = pred[:, 4]
                        labels = pred[:, 5]
                        _draw_boxes(orig_img, boxes, labels, scores=scores, color="red")
                    orig_img.save(vis_dir / f"{metas[i]['image_id']}.png")
    metrics = evaluator.compute()
    print(f"Evaluation on {split}: mAP@0.5={metrics['mAP']:.4f}, mean IoU={metrics['mean_iou']:.4f}")
    for cls_idx, ap in metrics["per_class_ap"].items():
        name = config.CLASS_NAMES[cls_idx] if cls_idx < len(config.CLASS_NAMES) else str(cls_idx)
        print(f"  Class {name}: AP={ap:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test", choices=["valid", "test"], help="Dataset split to evaluate")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--save-vis", type=str, default=None, help="Optional directory to save predicted vs GT overlays")
    parser.add_argument("--conf", type=float, default=None, help="Optional confidence threshold for decoding (defaults to config.CONF_THRESHOLD)")
    args = parser.parse_args()
    evaluate(args.split, Path(args.checkpoint), save_vis=Path(args.save_vis) if args.save_vis else None, conf_threshold=args.conf)


if __name__ == "__main__":
    main()
