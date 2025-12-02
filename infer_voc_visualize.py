import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

import config
import config_voc2012
from datasets.voc2012 import PascalVOC2012
from models.yolotiny import YOLOTiny
from utils.boxes import cxcywh_to_xyxy


def _apply_voc_config():
    for attr in dir(config_voc2012):
        if attr.isupper():
            setattr(config, attr, getattr(config_voc2012, attr))


def load_model(checkpoint_path: Path, device: torch.device) -> YOLOTiny:
    model = YOLOTiny().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_state = ckpt.get("model_state", ckpt)
    model.load_state_dict(model_state)
    model.eval()
    return model


def _get_font():
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        return ImageFont.load_default()


def draw_boxes(img: Image.Image, boxes, labels, scores=None, color="red") -> Image.Image:
    draw = ImageDraw.Draw(img)
    font = _get_font()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.tolist()
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        cls_idx = int(labels[i]) if labels is not None else -1
        label_text = config_voc2012.CLASS_NAMES[cls_idx] if 0 <= cls_idx < len(config_voc2012.CLASS_NAMES) else str(cls_idx)
        if scores is not None:
            label_text += f" {float(scores[i]):.2f}"
        draw.text((x1, y1), label_text, fill=color, font=font)
    return img


def visualize_samples(model: YOLOTiny, dataset: PascalVOC2012, device: torch.device, output_dir: Path, num: int, conf: float):
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(min(num, len(dataset))):
        img_tensor, _, meta = dataset[idx]
        img_batch = img_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            preds = model(img_batch)
            decoded = model.decode(preds, conf_threshold=conf)[0]

        orig_img = Image.open(meta["image_path"]).convert("RGB")
        orig_w, orig_h = meta["orig_size"]

        gt_boxes = cxcywh_to_xyxy(meta["boxes_orig"]).clone()
        gt_boxes[:, 0::2] *= orig_w
        gt_boxes[:, 1::2] *= orig_h
        draw_gt = draw_boxes(orig_img.copy(), gt_boxes, meta["labels"], scores=None, color="green")

        if decoded.numel() > 0:
            boxes = PascalVOC2012.unletterbox_boxes(
                decoded[:, :4],
                scale=meta["letterbox_scale"],
                pad=meta["letterbox_pad"],
                orig_size=meta["orig_size"],
                input_size=config.INPUT_SIZE,
            )
            scores = decoded[:, 4]
            labels = decoded[:, 5]
            draw_boxes(draw_gt, boxes, labels, scores, color="red")

        save_path = output_dir / f"{meta['image_id']}.png"
        draw_gt.save(save_path)
        print(f"Saved visualization to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize VOC2012 predictions with ground truth overlays")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to VOC checkpoint")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--num", type=int, default=10, help="Number of images to visualize")
    parser.add_argument("--conf", type=float, default=config_voc2012.CONF_THRESHOLD, help="Confidence threshold for decoding")
    parser.add_argument("--output", type=Path, default=Path("inference_outputs/voc"), help="Directory to save visualizations")
    args = parser.parse_args()

    _apply_voc_config()
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    dataset = PascalVOC2012(split=args.split, augment=False)
    model = load_model(args.checkpoint, device)

    visualize_samples(model, dataset, device, args.output, args.num, args.conf)


if __name__ == "__main__":
    main()
