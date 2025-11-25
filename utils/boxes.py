import torch
from typing import Tuple


def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-8)


def giou_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    # pred_boxes, target_boxes: (..., 4) in xyxy format
    pred_area = (pred_boxes[..., 2] - pred_boxes[..., 0]).clamp(min=0) * (pred_boxes[..., 3] - pred_boxes[..., 1]).clamp(min=0)
    target_area = (target_boxes[..., 2] - target_boxes[..., 0]).clamp(min=0) * (target_boxes[..., 3] - target_boxes[..., 1]).clamp(min=0)

    lt = torch.max(pred_boxes[..., :2], target_boxes[..., :2])
    rb = torch.min(pred_boxes[..., 2:], target_boxes[..., 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = pred_area + target_area - inter

    iou = inter / (union + 1e-8)
    enclosing_lt = torch.min(pred_boxes[..., :2], target_boxes[..., :2])
    enclosing_rb = torch.max(pred_boxes[..., 2:], target_boxes[..., 2:])
    enclosing_wh = (enclosing_rb - enclosing_lt).clamp(min=0)
    enclosing_area = enclosing_wh[..., 0] * enclosing_wh[..., 1]
    giou = iou - (enclosing_area - union) / (enclosing_area + 1e-8)
    loss = 1 - giou
    return loss
