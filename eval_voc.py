import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

import config
import config_voc2012
from datasets.voc2012 import PascalVOC2012, collate_fn
from models.yolotiny import YOLOTiny
from utils.metrics import Evaluator


def _apply_voc_config():
    for attr in dir(config_voc2012):
        if attr.isupper():
            setattr(config, attr, getattr(config_voc2012, attr))


def _load_checkpoint(model: torch.nn.Module, ckpt_path: Path):
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location="cpu")
    state = payload.get("model_state", payload)
    model.load_state_dict(state)
    return payload


def _decoded_to_records(decoded: List[torch.Tensor], metas) -> List[Dict[str, Any]]:
    batch_results: List[Dict[str, Any]] = []
    for det, meta in zip(decoded, metas):
        if det.numel() == 0:
            batch_results.append({"image_id": meta["image_id"], "detections": []})
            continue
        boxes_unletterboxed = PascalVOC2012.unletterbox_boxes(
            det[:, :4].cpu(),
            scale=meta["letterbox_scale"],
            pad=meta["letterbox_pad"],
            orig_size=meta["orig_size"],
            input_size=config.INPUT_SIZE,
        )
        scores = det[:, 4].cpu()
        labels = det[:, 5].long().cpu()
        records = []
        for box, score, label in zip(boxes_unletterboxed, scores, labels):
            records.append(
                {
                    "bbox": box.tolist(),
                    "score": float(score.item()),
                    "label": int(label.item()),
                    "class_name": config_voc2012.CLASS_NAMES[int(label.item())],
                }
            )
        batch_results.append(
            {
                "image_id": meta["image_id"],
                "detections": records,
                "orig_size": meta["orig_size"],
                "image_path": str(meta.get("image_path", "")),
            }
        )
    return batch_results


def evaluate_split(model, loader, device, conf_threshold: float):
    model.eval()
    evaluator = Evaluator(iou_threshold=0.5, unletterbox_fn=PascalVOC2012.unletterbox_boxes)
    all_preds: List[Dict[str, Any]] = []
    with torch.no_grad():
        for images, _, metas in loader:
            images = images.to(device)
            preds = model(images)
            decoded_batch = model.decode(preds, conf_threshold=conf_threshold)
            evaluator.add_batch(decoded_batch, metas)
            all_preds.extend(_decoded_to_records(decoded_batch, metas))
    metrics = evaluator.compute()
    evaluator.reset()
    return metrics, all_preds


def summarize_predictions(preds: List[Dict[str, Any]], save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(preds, f, indent=2)
    print(f"[INFO] Saved predictions to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate VOC2012 checkpoint and export predictions")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to voc checkpoint (voc_best.pt)")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--conf", type=float, default=config_voc2012.CONF_THRESHOLD, help="Confidence threshold")
    parser.add_argument("--save-json", type=Path, default=config_voc2012.LOG_DIR / "voc_predictions.json")
    args = parser.parse_args()

    _apply_voc_config()
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    dataset = PascalVOC2012(split=args.split, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    model = YOLOTiny().to(device)
    _load_checkpoint(model, args.checkpoint)

    metrics, preds = evaluate_split(model, loader, device, args.conf)
    print(json.dumps({"split": args.split, "metrics": metrics}, indent=2))
    summarize_predictions(preds, args.save_json)


if __name__ == "__main__":
    main()
