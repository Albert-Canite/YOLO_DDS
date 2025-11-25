import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import config
from datasets.dental_dds import DentalDDS, collate_fn
from models.yolotiny import YOLOTiny
from utils.metrics import Evaluator


def load_checkpoint(model: YOLOTiny, checkpoint_path: Path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])


def evaluate(split: str, checkpoint: Path):
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

    with torch.no_grad():
        for images, targets, metas in loader:
            images = images.to(device)
            preds = model(images)
            decoded = model.decode(preds)
            evaluator.add_batch(decoded, metas)
    metrics = evaluator.compute()
    print(f"Evaluation on {split}: mAP@0.5={metrics['mAP']:.4f}, mean IoU={metrics['mean_iou']:.4f}")
    for cls_idx, ap in metrics["per_class_ap"].items():
        name = config.CLASS_NAMES[cls_idx] if cls_idx < len(config.CLASS_NAMES) else str(cls_idx)
        print(f"  Class {name}: AP={ap:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    args = parser.parse_args()
    evaluate(args.split, Path(args.checkpoint))


if __name__ == "__main__":
    main()
