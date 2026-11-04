import torch
import torch.nn.functional as F

from v4.models.coverage_loss import coverage_chamfer_loss


def geometry_aware_loss(
    predicted,
    ground_truth,
    coverage_weight=2.0,
    dimension_weight=0.1,
):
    coverage_loss, p2g, g2p = coverage_chamfer_loss(
        predicted,
        ground_truth,
        coverage_weight=coverage_weight
    )

    pred_dims = (
        predicted.amax(dim=1)
        - predicted.amin(dim=1)
    )

    gt_dims = (
        ground_truth.amax(dim=1)
        - ground_truth.amin(dim=1)
    )

    dimension_loss = F.smooth_l1_loss(
        pred_dims,
        gt_dims
    )

    total_loss = (
        coverage_loss
        + dimension_weight * dimension_loss
    )

    return total_loss, coverage_loss, dimension_loss
