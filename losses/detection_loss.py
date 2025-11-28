from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from utils.boxes import cxcywh_to_xyxy, giou_loss, box_iou


class DetectionLoss(nn.Module):
    def __init__(self, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        if class_weights is not None:
            # Normalize weights to have mean=1 to keep loss scale stable
            norm = class_weights.mean().clamp(min=1e-8)
            self.register_buffer("class_weights", class_weights / norm)
        else:
            self.register_buffer("class_weights", None)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        # preds: (B, A, H, W, 5 + C), raw outputs
        b, a, h, w, _ = preds.shape
        device = preds.device
        anchors = torch.tensor(config.ANCHORS, dtype=torch.float32, device=device)
        tx = preds[..., 0]
        ty = preds[..., 1]
        tw = preds[..., 2]
        th = preds[..., 3]
        obj_logit = preds[..., 4]
        cls_logit = preds[..., 5:]

        t_tx = targets[..., 0]
        t_ty = targets[..., 1]
        t_tw = targets[..., 2]
        t_th = targets[..., 3]
        t_obj = targets[..., 4]
        t_cls = targets[..., 5:]

        grid_y, grid_x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
        grid_x = grid_x.view(1, 1, h, w)
        grid_y = grid_y.view(1, 1, h, w)
        anchor_w = anchors[:, 0].view(1, a, 1, 1)
        anchor_h = anchors[:, 1].view(1, a, 1, 1)

        pred_cx = (torch.sigmoid(tx) + grid_x) / w
        pred_cy = (torch.sigmoid(ty) + grid_y) / h
        pred_w = anchor_w * torch.exp(tw) / config.INPUT_SIZE
        pred_h = anchor_h * torch.exp(th) / config.INPUT_SIZE
        pred_boxes = cxcywh_to_xyxy(torch.stack([pred_cx, pred_cy, pred_w, pred_h], dim=-1))

        tgt_cx = (t_tx + grid_x) / w
        tgt_cy = (t_ty + grid_y) / h
        tgt_w = anchor_w * torch.exp(t_tw) / config.INPUT_SIZE
        tgt_h = anchor_h * torch.exp(t_th) / config.INPUT_SIZE
        tgt_boxes = cxcywh_to_xyxy(torch.stack([tgt_cx, tgt_cy, tgt_w, tgt_h], dim=-1))

        pos_mask = t_obj > 0

        # Ignore objectness penalty for anchors whose predicted boxes heavily
        # overlap any GT, even if they are not the designated positive anchor
        # for that GT. This prevents punishing nearby anchors and collapsing
        # recall in crowded regions.
        ignore_mask = torch.zeros_like(t_obj, dtype=torch.bool)
        for bi in range(b):
            if pos_mask[bi].any():
                gt_boxes = tgt_boxes[bi][pos_mask[bi]]
                preds_flat = pred_boxes[bi].view(-1, 4)
                ious = box_iou(preds_flat, gt_boxes)
                max_iou = ious.max(dim=1).values.view(a, h, w)
                ignore_mask[bi] = max_iou > config.OBJ_IGNORE_IOU
        neg_mask = (~pos_mask) & (~ignore_mask)

        giou = giou_loss(pred_boxes, tgt_boxes) * pos_mask
        giou_loss_val = giou.sum() / (pos_mask.sum() + 1e-8)

        obj_loss = self.bce(obj_logit, t_obj)
        # emphasize positives to avoid collapse to empty predictions
        pos_obj = (obj_loss * pos_mask).sum()
        neg_obj = (obj_loss * neg_mask).sum()
        obj_loss_val = (5.0 * pos_obj + neg_obj) / (5.0 * pos_mask.sum() + neg_mask.sum() + 1e-8)

        if config.NUM_CLASSES > 1:
            cls_loss = F.binary_cross_entropy_with_logits(cls_logit, t_cls, reduction="none")
            if self.class_weights is not None:
                cls_loss = cls_loss * self.class_weights.view(1, 1, 1, 1, -1)
            cls_loss_val = (cls_loss * pos_mask.unsqueeze(-1)).sum() / (pos_mask.sum() + 1e-8)
        else:
            cls_loss_val = torch.tensor(0.0, device=device)

        total = giou_loss_val + obj_loss_val + cls_loss_val
        return {
            "total": total,
            "giou": giou_loss_val,
            "objectness": obj_loss_val,
            "class": cls_loss_val,
        }
