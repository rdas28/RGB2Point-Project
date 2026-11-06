import torch

from v4.models.geometry_aware_loss import geometry_aware_loss


def symmetric_chamfer(a, b):
    distances = torch.cdist(a, b, p=2)

    a_to_b = distances.min(dim=2).values.mean()
    b_to_a = distances.min(dim=1).values.mean()

    return a_to_b + b_to_a


def cross_view_loss(
    pred_a,
    pred_b,
    gt,
    coverage_weight=2.0,
    dimension_weight=0.1,
    consistency_weight=0.1
):
    geometry_a, _, _ = geometry_aware_loss(
        pred_a,
        gt,
        coverage_weight=coverage_weight,
        dimension_weight=dimension_weight
    )

    geometry_b, _, _ = geometry_aware_loss(
        pred_b,
        gt,
        coverage_weight=coverage_weight,
        dimension_weight=dimension_weight
    )

    geometry = 0.5 * (geometry_a + geometry_b)

    consistency = symmetric_chamfer(pred_a, pred_b)

    total = geometry + consistency_weight * consistency

    return total, geometry, consistency
