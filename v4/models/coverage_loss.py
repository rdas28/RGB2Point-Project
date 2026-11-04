import torch


def coverage_chamfer_loss(
    predicted,
    ground_truth,
    coverage_weight=2.0,
):
    """
    Weighted bidirectional Chamfer Distance.

    predicted:    (B, N, 3)
    ground_truth: (B, M, 3)

    Increasing coverage_weight places more emphasis
    on covering ground-truth surface points.
    """

    distances = torch.cdist(
        predicted,
        ground_truth,
        p=2
    )

    pred_to_gt = distances.min(dim=2).values.mean()
    gt_to_pred = distances.min(dim=1).values.mean()

    loss = (
        pred_to_gt
        + coverage_weight * gt_to_pred
    )

    return loss, pred_to_gt, gt_to_pred
