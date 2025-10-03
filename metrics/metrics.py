import torch


def compute_metrics(pred, gt):
    pred = torch.clamp(pred, min=1e-3)
    gt = torch.clamp(gt, min=1e-3)

    scale = torch.median(gt) / torch.median(pred)
    pred = pred * scale

    abs_rel = torch.mean(torch.abs(pred - gt) / gt)
    sq_rel = torch.mean(((pred - gt) ** 2) / gt)
    rmse = torch.sqrt(torch.mean((pred - gt) ** 2))
    rmse_log = torch.sqrt(torch.mean((torch.log(pred) - torch.log(gt)) ** 2))

    max_ratio = torch.max(pred / gt, gt / pred)
    delta1 = torch.mean((max_ratio < 1.25).float())
    delta2 = torch.mean((max_ratio < 1.25**2).float())
    delta3 = torch.mean((max_ratio < 1.25**3).float())

    return {
        "AbsRel": abs_rel.item(),
        "SqRel": sq_rel.item(),
        "RMSE": rmse.item(),
        "RMSElog": rmse_log.item(),
        "δ1": delta1.item(),
        "δ2": delta2.item(),
        "δ3": delta3.item(),
    }
