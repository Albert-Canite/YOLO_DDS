import argparse
import json
import random
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

import config
from datasets.dental_dds import DentalDDS, collate_fn, compute_class_frequencies, gather_label_stats
from losses.detection_loss import DetectionLoss
from models.yolotiny import YOLOTiny
from utils.metrics import Evaluator
from utils.boxes import cxcywh_to_xyxy


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_dataloaders():
    train_dataset = DentalDDS(split="train", augment=True)
    val_dataset = DentalDDS(split="valid", augment=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    return train_loader, val_loader, train_dataset


def _mean_pos_per_image(targets: torch.Tensor) -> float:
    pos = targets[..., 4] > 0
    pos_per_img = pos.view(targets.size(0), -1).sum(dim=1).float()
    return float(pos_per_img.mean().item())


def train_one_epoch(model, criterion, optimizer, loader, device) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    pos_meter = 0.0
    for images, targets, _ in loader:
        images = images.to(device)
        targets = targets.to(device)
        preds = model(images)
        loss_dict = criterion(preds, targets)
        loss = loss_dict["total"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pos_meter += _mean_pos_per_image(targets)
    return {
        "loss": total_loss / len(loader),
        "avg_pos_per_image": pos_meter / len(loader),
    }


def validate(model, criterion, loader, device, evaluator: Evaluator):
    model.eval()
    total_loss = 0.0
    pos_meter = 0.0
    debug_batch: Dict[str, Any] = {}
    with torch.no_grad():
        for batch_idx, (images, targets, metas) in enumerate(loader):
            images = images.to(device)
            targets = targets.to(device)
            preds = model(images)
            loss_dict = criterion(preds, targets)
            total_loss += loss_dict["total"].item()
            pos_meter += _mean_pos_per_image(targets)
            decoded = model.decode(preds, conf_threshold=config.VAL_CONF_THRESHOLD)
            evaluator.add_batch(decoded, metas)
            # Capture first validation batch for deeper debugging
            if batch_idx == 0:
                debug_batch = _collect_debug_batch(model, decoded, preds, metas)
    avg_loss = total_loss / max(1, len(loader))
    metrics = evaluator.compute()
    evaluator.reset()
    return avg_loss, metrics, pos_meter / max(1, len(loader)), debug_batch


def _collect_debug_batch(model, decoded, preds, metas) -> Dict[str, Any]:
    """Package a lightweight snapshot of predictions/targets for post-mortem analysis."""
    batch_info = []
    for det, meta, raw in zip(decoded, metas, preds):
        entry: Dict[str, Any] = {
            "image_id": meta["image_id"],
            "orig_size": meta["orig_size"],
            "num_gt": int(meta["boxes_orig"].shape[0]),
            "num_pred_val_thresh": int(det.shape[0]),
            "avg_obj_logit": float(raw[..., 4].mean().item()),
        }
        low_thresh = model.decode(raw.unsqueeze(0), conf_threshold=0.05)[0]
        entry["num_pred_low_thresh"] = int(low_thresh.shape[0])
        if low_thresh.numel() and meta["boxes_orig"].numel():
            gt_xyxy = cxcywh_to_xyxy(meta["boxes_orig"]).cpu()
            orig_w, orig_h = meta["orig_size"]
            gt_xyxy[:, 0::2] *= orig_w
            gt_xyxy[:, 1::2] *= orig_h
            pred_xyxy = DentalDDS.unletterbox_boxes(
                low_thresh[:, :4].cpu(),
                scale=meta["letterbox_scale"],
                pad=meta["letterbox_pad"],
                orig_size=meta["orig_size"],
            )
            ious = box_iou(pred_xyxy, gt_xyxy)
            entry.update(
                {
                    "pred_to_gt_iou_mean": float(ious.max(dim=1).values.mean().item()),
                    "gt_to_pred_iou_mean": float(ious.max(dim=0).values.mean().item()),
                }
            )
        batch_info.append(entry)
    return {"images": batch_info}


def _save_debug(epoch: int, payload: Dict[str, Any]):
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.LOG_DIR / f"debug_epoch_{epoch:03d}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[DEBUG] Wrote {out_path}")


def main(args):
    set_seed(config.SEED)
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, train_dataset = create_dataloaders()
    freq = compute_class_frequencies(train_dataset.label_dir)
    label_stats = {
        "train": gather_label_stats(config.LABELS_DIRS["train"]),
        "valid": gather_label_stats(config.LABELS_DIRS["valid"]),
        "test": gather_label_stats(config.LABELS_DIRS["test"]),
    }
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.LOG_DIR / "dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(label_stats, f, indent=2)
    class_weights = torch.tensor([1.0 / max(c, 1) for c in freq], dtype=torch.float32, device=device)

    model = YOLOTiny().to(device)
    criterion = DetectionLoss(class_weights=class_weights)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.LEARNING_RATE, momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

    best_map = 0.0
    for epoch in range(1, config.EPOCHS + 1):
        train_stats = train_one_epoch(model, criterion, optimizer, train_loader, device)
        evaluator = Evaluator(iou_threshold=0.5)
        val_loss, metrics, val_pos_avg, debug_batch = validate(model, criterion, val_loader, device, evaluator)
        scheduler.step()
        map50 = metrics.get("mAP", 0.0)
        mean_iou = metrics.get("mean_iou", 0.0)
        log_line = (
            f"Epoch {epoch:03d} | train_loss: {train_stats['loss']:.4f} | val_loss: {val_loss:.4f} | "
            f"train_pos/img: {train_stats['avg_pos_per_image']:.2f} | val_pos/img: {val_pos_avg:.2f} | "
            f"val_mAP@0.5: {map50:.4f} | val_meanIoU: {mean_iou:.4f}"
        )
        print(log_line)
        _save_debug(
            epoch,
            {
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "val_loss": val_loss,
                "train_pos_per_image": train_stats["avg_pos_per_image"],
                "val_pos_per_image": val_pos_avg,
                "metrics": metrics,
                "label_stats": label_stats,
                "val_debug_batch": debug_batch,
            },
        )
        if map50 > best_map:
            best_map = map50
            ckpt_path = config.CHECKPOINT_DIR / "best.pt"
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "metrics": metrics,
            }, ckpt_path)
    ckpt_last = config.CHECKPOINT_DIR / "last.pt"
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": config.EPOCHS,
    }, ckpt_last)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Override epochs")
    args_parsed = parser.parse_args()
    if args_parsed.epochs != config.EPOCHS:
        config.EPOCHS = args_parsed.epochs
    main(args_parsed)
