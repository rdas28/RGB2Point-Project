import torch
import torch.nn.functional as F

from v4.models.coverage_loss import coverage_chamfer_loss


def local_spacing_loss(pred, gt, k=4):
    pred_dist = torch.cdist(pred.float(), pred.float())
    gt_dist = torch.cdist(gt.float(), gt.float())

    pred_neighbors = pred_dist.topk(
        k + 1, dim=-1, largest=False
    ).values[..., 1:]

    gt_neighbors = gt_dist.topk(
        k + 1, dim=-1, largest=False
    ).values[..., 1:]

    target_spacing = gt_neighbors.detach().mean(
        dim=(1, 2), keepdim=True
    )

    minimum_spacing = 0.75 * target_spacing

    penalty = F.relu(
        minimum_spacing - pred_neighbors
    )

    return penalty.square().mean()


def surface_aware_loss(
    pred,
    gt,
    coverage_weight=2.0,
    dimension_weight=0.1,
    spacing_weight=0.05,
    k=4
):
    coverage, _, _ = coverage_chamfer_loss(
        pred, gt, coverage_weight=coverage_weight
    )

    pred_dims = pred.max(dim=1).values - pred.min(dim=1).values
    gt_dims = gt.max(dim=1).values - gt.min(dim=1).values

    dimension = F.smooth_l1_loss(pred_dims, gt_dims)

    spacing = local_spacing_loss(pred, gt, k=k)

    total = (
        coverage
        + dimension_weight * dimension
        + spacing_weight * spacing
    )

    return total, coverage, dimension, spacing
