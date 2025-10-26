import numpy as np
import torch


def compute_metrics(pred, gt):
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.cpu().numpy()

    valid_mask = gt > 1e-6
    pred = pred[valid_mask]
    gt = gt[valid_mask]

    if gt.shape[0] == 0:
        return {
            'rmse': 0.0,
            'abs_rel': 0.0,
            'a1': 0.0,
            'a2': 0.0,
            'a3': 0.0,
        }

    # RMSE
    rmse = np.sqrt(np.mean((pred - gt) ** 2))

    # AbsRel
    abs_rel = np.mean(np.abs(pred - gt) / gt)

    # Thresholds (a1, a2, a3)
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25**2).mean()
    a3 = (thresh < 1.25**3).mean()

    return {
        'rmse': rmse,
        'abs_rel': abs_rel,
        'a1': a1,
        'a2': a2,
        'a3': a3,
    }


def compute_sample_rmse(pred, gt):
    valid_mask = gt > 1e-6
    if np.sum(valid_mask) == 0:
        return 0.0

    pred_masked = pred[valid_mask]
    gt_masked = gt[valid_mask]

    return np.sqrt(np.mean((pred_masked - gt_masked) ** 2))
