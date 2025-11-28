from typing import List, Dict, Tuple

import torch
from torchvision.ops import box_iou

import config
from datasets.dental_dds import DentalDDS
from utils.boxes import cxcywh_to_xyxy


def compute_ap(recall: torch.Tensor, precision: torch.Tensor) -> float:
    mrec = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
    mpre = torch.cat([torch.tensor([0.0]), precision, torch.tensor([0.0])])
    for i in range(mpre.size(0) - 1, 0, -1):
        mpre[i - 1] = torch.maximum(mpre[i - 1], mpre[i])
    idx = torch.where(mrec[1:] != mrec[:-1])[0]
    ap = torch.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).item()
    return ap


class Evaluator:
    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self):
        self.detections: Dict[int, List[Tuple[float, int, torch.Tensor]]] = {}
        self.ground_truths: Dict[int, List[Tuple[int, torch.Tensor]]] = {}
        self.image_id_map: List[str] = []
        self.total_iou = 0.0
        self.iou_count = 0

    def add_batch(self, outputs: List[torch.Tensor], metas: List[Dict]):
        for out, meta in zip(outputs, metas):
            out_cpu = out.detach().cpu()
            image_idx = len(self.image_id_map)
            self.image_id_map.append(meta["image_id"])
            gt_boxes_norm = meta["boxes_orig"].detach().cpu().float()
            gt_labels = meta["labels"].detach().cpu()
            orig_w, orig_h = meta["orig_size"]
            gt_xyxy_abs = cxcywh_to_xyxy(gt_boxes_norm)
            gt_xyxy_abs[:, 0::2] *= orig_w
            gt_xyxy_abs[:, 1::2] *= orig_h
            for label, box in zip(gt_labels, gt_xyxy_abs):
                self.ground_truths.setdefault(label.item(), []).append((image_idx, box))
            if out_cpu.numel() == 0:
                continue
            boxes = DentalDDS.unletterbox_boxes(
                out_cpu[:, :4],
                scale=meta["letterbox_scale"],
                pad=meta["letterbox_pad"],
                orig_size=meta["orig_size"],
            )
            scores = out_cpu[:, 4]
            labels = out_cpu[:, 5].long()
            for b, s, l in zip(boxes, scores, labels):
                self.detections.setdefault(l.item(), []).append((s.item(), image_idx, b))

            # mean IoU against matched GT in original coordinates (greedy match)
            if gt_xyxy_abs.numel() > 0:
                ious = box_iou(boxes, gt_xyxy_abs)
                if ious.numel() > 0:
                    # Greedy assign highest-IoU pairs above threshold to avoid
                    # counting thousands of unmatched low-score predictions.
                    matched = []
                    gt_used = set()
                    for pred_idx in torch.argsort(scores, descending=True):
                        iou_row = ious[pred_idx]
                        max_iou, max_gt = iou_row.max(dim=0)
                        if max_iou.item() >= self.iou_threshold and max_gt.item() not in gt_used:
                            matched.append(max_iou.item())
                            gt_used.add(max_gt.item())
                    if matched:
                        self.total_iou += sum(matched)
                        self.iou_count += len(matched)

    def compute(self) -> Dict[str, float]:
        aps = []
        per_class_ap = {}
        for cls in range(config.NUM_CLASSES):
            dets = self.detections.get(cls, [])
            gts = self.ground_truths.get(cls, [])
            npos = len(gts)
            if npos == 0:
                continue
            dets = sorted(dets, key=lambda x: x[0], reverse=True)
            tp = torch.zeros(len(dets))
            fp = torch.zeros(len(dets))
            gt_flags = {}
            for i, (score, img_id, box) in enumerate(dets):
                gt_candidates = [(idx, b) for idx, b in gts if idx == img_id]
                if len(gt_candidates) == 0:
                    fp[i] = 1
                    continue
                gt_boxes = torch.stack([b for _, b in gt_candidates])
                pred_box = box.unsqueeze(0)
                ious = box_iou(pred_box, gt_boxes)
                max_iou, max_idx = ious.max(dim=1)
                if max_iou.item() >= self.iou_threshold and gt_flags.get((img_id, max_idx.item()), False) is False:
                    tp[i] = 1
                    gt_flags[(img_id, max_idx.item())] = True
                else:
                    fp[i] = 1
            if tp.sum() == 0:
                per_class_ap[cls] = 0.0
                aps.append(0.0)
                continue
            fp = torch.cumsum(fp, dim=0)
            tp = torch.cumsum(tp, dim=0)
            recall = tp / npos
            precision = tp / torch.clamp(tp + fp, min=1e-8)
            ap = compute_ap(recall, precision)
            per_class_ap[cls] = ap
            aps.append(ap)
        mAP = float(torch.tensor(aps).mean().item()) if aps else 0.0
        mean_iou = self.total_iou / max(1, self.iou_count)
        return {
            "mAP": mAP,
            "per_class_ap": per_class_ap,
            "mean_iou": mean_iou,
        }
