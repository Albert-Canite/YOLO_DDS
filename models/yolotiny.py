from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms

import config
from models.backbone import TinyBackbone
from models.head import DetectionHead
from utils.boxes import cxcywh_to_xyxy


class YOLOTiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = TinyBackbone()
        self.head = DetectionHead(in_channels=1024)
        self.stride = config.STRIDE
        self.register_buffer("anchors", torch.tensor(config.ANCHORS, dtype=torch.float32))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        pred = self.head(features)
        return pred

    def decode(self, pred: torch.Tensor, conf_threshold: float = None) -> List[torch.Tensor]:
        # pred: (B, A, H, W, 5 + C)
        if conf_threshold is None:
            conf_threshold = config.CONF_THRESHOLD
        b, a, h, w, c = pred.shape
        device = pred.device
        grid_y, grid_x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
        grid_x = grid_x.view(1, 1, h, w)
        grid_y = grid_y.view(1, 1, h, w)

        pred = pred.clone()
        tx = torch.sigmoid(pred[..., 0])
        ty = torch.sigmoid(pred[..., 1])
        tw = pred[..., 2]
        th = pred[..., 3]
        obj = torch.sigmoid(pred[..., 4])
        cls_logits = pred[..., 5:]
        cls_scores = torch.softmax(cls_logits, dim=-1) if config.NUM_CLASSES > 1 else torch.ones_like(cls_logits)

        anchor_w = self.anchors[:, 0].view(1, a, 1, 1)
        anchor_h = self.anchors[:, 1].view(1, a, 1, 1)

        cx = (tx + grid_x) / w
        cy = (ty + grid_y) / h
        bw = anchor_w * torch.exp(tw) / config.INPUT_SIZE
        bh = anchor_h * torch.exp(th) / config.INPUT_SIZE
        boxes = cxcywh_to_xyxy(torch.stack([cx, cy, bw, bh], dim=-1)).clamp(0, 1)

        outputs = []
        for bi in range(b):
            boxes_b = boxes[bi].reshape(-1, 4)
            obj_b = obj[bi].reshape(-1)
            cls_b = cls_scores[bi].reshape(-1, config.NUM_CLASSES)
            scores, labels = cls_b.max(dim=-1)
            scores = scores * obj_b
            mask = scores > conf_threshold
            if mask.sum() == 0:
                outputs.append(torch.zeros((0, 6), device=device))
                continue
            boxes_sel = boxes_b[mask]
            scores_sel = scores[mask]
            labels_sel = labels[mask]
            keep = nms(boxes_sel, scores_sel, config.NMS_IOU_THRESHOLD)
            boxes_sel = boxes_sel[keep]
            scores_sel = scores_sel[keep]
            labels_sel = labels_sel[keep]
            outputs.append(torch.cat([boxes_sel, scores_sel.unsqueeze(1), labels_sel.unsqueeze(1).float()], dim=1))
        return outputs
