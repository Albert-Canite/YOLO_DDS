from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from utils.boxes import cxcywh_to_xyxy, giou_loss


class DetectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

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
        neg_mask = ~pos_mask

        giou = giou_loss(pred_boxes, tgt_boxes) * pos_mask
        giou_loss_val = giou.sum() / (pos_mask.sum() + 1e-8)

        obj_loss = self.bce(obj_logit, t_obj)
        obj_loss_val = obj_loss.sum() / obj_loss.numel()

        if config.NUM_CLASSES > 1:
            cls_loss = F.binary_cross_entropy_with_logits(cls_logit, t_cls, reduction="none")
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
