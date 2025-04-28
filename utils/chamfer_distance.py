import torch

def chamfer_distance(pred, gt):
    pred_expand = pred.unsqueeze(2)
    gt_expand = gt.unsqueeze(1)
    dist = torch.norm(pred_expand - gt_expand, dim=3)
    min_dist_pred_to_gt = dist.min(dim=2)[0]
    min_dist_gt_to_pred = dist.min(dim=1)[0]
    loss = (min_dist_pred_to_gt.mean(dim=1) + min_dist_gt_to_pred.mean(dim=1)).mean()
    return loss
