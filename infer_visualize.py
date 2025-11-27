import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

import config
from datasets.dental_dds import DentalDDS
from models.yolotiny import YOLOTiny
from utils.boxes import cxcywh_to_xyxy


def load_model(checkpoint_path: Path, device):
    model = YOLOTiny().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def draw_boxes(img: Image.Image, boxes: torch.Tensor, labels: torch.Tensor, scores: torch.Tensor = None, color="red"):
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.tolist()
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label_text = config.CLASS_NAMES[int(labels[i])] if labels is not None and int(labels[i]) < len(config.CLASS_NAMES) else str(int(labels[i]))
        if scores is not None:
            label_text += f" {scores[i]:.2f}"
        draw.text((x1, y1), label_text, fill=color, font=font)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--output", type=str, default="inference_outputs", help="Directory to save visualizations")
    parser.add_argument("--num", type=int, default=5, help="Number of test images to visualize")
    args = parser.parse_args()

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    model = load_model(Path(args.checkpoint), device)

    dataset = DentalDDS(split="test", augment=False)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(min(args.num, len(dataset))):
        img_tensor, target, meta = dataset[idx]
        img_batch = img_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            preds = model(img_batch)
            decoded = model.decode(preds)[0]
        orig_img = Image.open(meta["image_path"]).convert("RGB")
        gt_boxes = cxcywh_to_xyxy(meta["boxes_orig"]).clone()
        orig_w, orig_h = meta["orig_size"]
        gt_boxes[:, 0::2] *= orig_w
        gt_boxes[:, 1::2] *= orig_h
        draw_gt = draw_boxes(orig_img.copy(), gt_boxes, meta["labels"], scores=None, color="green")
        if decoded.numel() > 0:
            boxes = DentalDDS.unletterbox_boxes(
                decoded[:, :4],
                scale=meta["letterbox_scale"],
                pad=meta["letterbox_pad"],
                orig_size=meta["orig_size"],
            )
            scores = decoded[:, 4]
            labels = decoded[:, 5]
            draw_boxes(draw_gt, boxes, labels, scores, color="red")
        save_path = output_dir / f"{meta['image_id']}.png"
        draw_gt.save(save_path)
        print(f"Saved visualization to {save_path}")


if __name__ == "__main__":
    main()
