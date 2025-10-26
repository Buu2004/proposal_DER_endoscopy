import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from src.uncertainty_depth.models.backbone import SurgicalDINOUncertaintyModel


def load_model(mode, checkpoint_path, model_config, device, dropout_p=0.1):
    print(f"Loading model for {mode} from {checkpoint_path}")

    image_size = tuple(model_config['image_size'])

    model = SurgicalDINOUncertaintyModel(
        backbone_size=model_config['backbone_size'],
        r=model_config['r'],
        image_shape=image_size,
        decode_type=model_config['decode_type'],
        lam=model_config.get('lam', 0.2),
        mode=mode,
        dropout_p=dropout_p,
    ).to(device)

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return None

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
    else:
        model_state = checkpoint

    try:
        model.load_state_dict(model_state)
    except RuntimeError:
        print(
            "Could not load state_dict directly. Attempting to load parameters manually..."
        )
        model.load_parameters(checkpoint_path, device=device)

    if mode == 'evidential':
        model.eval()
    else:
        model.train()

    print(f"Model for {mode} loaded successfully.")
    return model


def get_edl_outputs(model, img_batch, gt_shape):
    with torch.no_grad():
        out = model(pixel_values=img_batch)

        pred_depth = F.interpolate(
            out["predicted_depth"],
            size=gt_shape[2:],
            mode='bilinear',
            align_corners=False,
        )

        evidence = F.interpolate(
            out["evidence"],
            size=gt_shape[2:],
            mode='bilinear',
            align_corners=False,
        )

        aleatoric_unc = F.interpolate(
            out["aleatoric_uncertainty"],
            size=gt_shape[2:],
            mode='bilinear',
            align_corners=False,
        )

        epistemic_unc = F.interpolate(
            out["epistemic_uncertainty"],
            size=gt_shape[2:],
            mode='bilinear',
            align_corners=False,
        )

    return {
        'pred_depth': pred_depth[0, 0].cpu().numpy(),
        'evidence': evidence[0, 0].cpu().numpy(),
        'aleatoric_unc': aleatoric_unc[0, 0].cpu().numpy(),
        'epistemic_unc': epistemic_unc[0, 0].cpu().numpy(),
    }


def get_mcd_outputs(model, img_batch, gt_shape, mc_iterations):
    mc_predictions = []
    with torch.no_grad():
        for _ in range(mc_iterations):
            pred = model(pixel_values=img_batch)['predicted_depth']
            mc_predictions.append(
                F.interpolate(
                    pred, size=gt_shape[2:], mode='bilinear', align_corners=False
                ).cpu()
            )

    mc_stack = torch.stack(mc_predictions, dim=0)
    pred_depth_mean = torch.mean(mc_stack, dim=0)
    uncertainty_var = torch.var(mc_stack, dim=0)

    return {
        'pred_depth': pred_depth_mean[0, 0].numpy(),
        'variance': uncertainty_var[0, 0].numpy(),
    }


def plot_comparison_scatter(df_edl, df_mcd, save_path="rmse_vs_confidence.png"):
    df_edl['Method'] = 'DER'
    df_mcd['Method'] = 'MC Dropout'
    df_edl.rename(columns={'sample_confidence': 'confidence'}, inplace=True)
    df_mcd.rename(columns={'sample_confidence': 'confidence'}, inplace=True)

    combined_df = pd.concat([df_edl, df_mcd], ignore_index=True)

    scaler = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6)
    combined_df['confidence_normalized'] = combined_df.groupby('Method')[
        'confidence'
    ].transform(scaler)

    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")

    plot = sns.lmplot(
        data=combined_df,
        x='confidence_normalized',
        y='sample_rmse',
        hue='Method',
        palette='deep',
        height=7,
        aspect=1.5,
        scatter_kws={'alpha': 0.5},
        line_kws={'linestyle': '--'},
    )

    plot.set(
        title='Per-Image RMSE vs. Confidence',
        xlabel='Confidence',
        ylabel='RMSE',
    )
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison scatter plot to: {save_path}")
    plt.close()
