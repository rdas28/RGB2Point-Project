
import torch
import torch.nn.functional as F

from utils.chamfer_distance import chamfer_distance


def geometry_loss(pred, gt, bbox_weight=0.1):
    # Standard Chamfer Distance
    cd = chamfer_distance(pred, gt)

    # Bounding-box dimensions
    pred_dims = pred.max(dim=1).values - pred.min(dim=1).values
    gt_dims = gt.max(dim=1).values - gt.min(dim=1).values

    # Penalize differences in X, Y, and Z dimensions
    bbox_loss = F.smooth_l1_loss(pred_dims, gt_dims)

    total_loss = cd + bbox_weight * bbox_loss

    return total_loss, cd, bbox_loss

