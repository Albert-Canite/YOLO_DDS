import argparse
import random
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

import config
from datasets.dental_dds import DentalDDS, collate_fn, compute_class_frequencies
from losses.detection_loss import DetectionLoss
from models.yolotiny import YOLOTiny
from utils.metrics import Evaluator


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


def train_one_epoch(model, criterion, optimizer, loader, device) -> float:
    model.train()
    total_loss = 0.0
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
    return total_loss / len(loader)


def validate(model, criterion, loader, device, evaluator: Evaluator):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, targets, metas in loader:
            images = images.to(device)
            targets = targets.to(device)
            preds = model(images)
            loss_dict = criterion(preds, targets)
            total_loss += loss_dict["total"].item()
            decoded = model.decode(preds, conf_threshold=config.VAL_CONF_THRESHOLD)
            evaluator.add_batch(decoded, metas)
    avg_loss = total_loss / max(1, len(loader))
    metrics = evaluator.compute()
    evaluator.reset()
    return avg_loss, metrics


def main(args):
    set_seed(config.SEED)
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, train_dataset = create_dataloaders()
    freq = compute_class_frequencies(train_dataset.label_dir)
    class_weights = torch.tensor([1.0 / max(c, 1) for c in freq], dtype=torch.float32, device=device)

    model = YOLOTiny().to(device)
    criterion = DetectionLoss(class_weights=class_weights)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.LEARNING_RATE, momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

    best_map = 0.0
    for epoch in range(1, config.EPOCHS + 1):
        train_loss = train_one_epoch(model, criterion, optimizer, train_loader, device)
        evaluator = Evaluator(iou_threshold=0.5)
        val_loss, metrics = validate(model, criterion, val_loader, device, evaluator)
        scheduler.step()
        map50 = metrics.get("mAP", 0.0)
        mean_iou = metrics.get("mean_iou", 0.0)
        log_line = f"Epoch {epoch:03d} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | val_mAP@0.5: {map50:.4f} | val_meanIoU: {mean_iou:.4f}"
        print(log_line)
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
