import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from src.uncertainty_depth.data import DepthDataset
from src.uncertainty_depth.utils import get_edl_outputs, get_mcd_outputs, load_model
from torch.utils.data import DataLoader
from tqdm import tqdm


def plot_qualitative_comparison(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg_data = config['data']
    cfg_model = config['model']
    cfg_eval = config['evaluation']
    cfg_plot = config['plotting']

    cfg_model['image_size'] = cfg_data['image_size']

    test_df = pd.read_csv(cfg_data['test_file'], sep="\t", header=None)
    num_samples = cfg_plot.get('num_samples_to_plot', 10)
    test_df = test_df.sample(num_samples) if num_samples < len(test_df) else test_df

    test_dataset = DepthDataset(test_df, tuple(cfg_data['image_size']))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    edl_model = load_model(
        'evidential', cfg_eval['evidential_checkpoint'], cfg_model, device
    )

    mcd_model = load_model(
        'mc_dropout',
        cfg_eval['mc_dropout_checkpoint'],
        cfg_model,
        device,
        dropout_p=cfg_model.get('dropout_p', 0.1),
    )

    if edl_model is None or mcd_model is None:
        print("Failed to load one or both models. Exiting.")
        return

    output_dir = cfg_plot.get('output_dir', 'results/qualitative_plots')
    os.makedirs(output_dir, exist_ok=True)
    epsilon = cfg_eval.get('epsilon', 1e-6)
    mc_iterations = cfg_eval.get('mc_iterations', 15)

    for i, batch in enumerate(tqdm(test_loader, desc="Generating Plots")):
        if batch is None:
            continue

        img_batch = batch["image"].to(device)
        gt_depth = batch["depth"][0, 0].cpu().numpy()
        original_img = (batch["image"][0].permute(1, 2, 0).cpu().numpy() * 255).astype(
            np.uint8
        )

        edl_out = get_edl_outputs(edl_model, img_batch, batch["depth"].shape)
        edl_pred_depth = edl_out['pred_depth']
        edl_evidence = edl_out['evidence']
        edl_abs_diff = np.abs(edl_pred_depth - gt_depth)

        mcd_out = get_mcd_outputs(
            mcd_model, img_batch, batch["depth"].shape, mc_iterations
        )
        mcd_pred_depth = mcd_out['pred_depth']
        mcd_variance = mcd_out['variance']
        mcd_abs_diff = np.abs(mcd_pred_depth - gt_depth)

        edl_conf_norm = (edl_evidence - edl_evidence.min()) / (
            edl_evidence.max() - edl_evidence.min() + epsilon
        )

        mcd_raw_conf = 1.0 / (mcd_variance + epsilon)
        mcd_conf_norm = (mcd_raw_conf - mcd_raw_conf.min()) / (
            mcd_raw_conf.max() - mcd_raw_conf.min() + epsilon
        )

        fig, axs = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(f"Qualitative Comparison - Sample {i}", fontsize=16)

        axs[0, 0].imshow(original_img)
        axs[0, 0].set_title("Input Image")
        axs[0, 0].axis('off')

        im = axs[0, 1].imshow(edl_abs_diff, cmap="inferno")
        axs[0, 1].set_title("DER Absolute Error")
        axs[0, 1].axis('off')
        plt.colorbar(im, ax=axs[0, 1], fraction=0.046, pad=0.04)

        im = axs[0, 2].imshow(edl_conf_norm, cmap="viridis")
        axs[0, 2].set_title("DER Confidence (Evidence)")
        axs[0, 2].axis('off')
        plt.colorbar(im, ax=axs[0, 2], fraction=0.046, pad=0.04)

        axs[1, 0].imshow(original_img)
        axs[1, 0].set_title("Input Image")
        axs[1, 0].axis('off')

        im = axs[1, 1].imshow(mcd_abs_diff, cmap="inferno")
        axs[1, 1].set_title("MC Dropout Absolute Error")
        axs[1, 1].axis('off')
        plt.colorbar(im, ax=axs[1, 1], fraction=0.046, pad=0.04)

        im = axs[1, 2].imshow(mcd_conf_norm, cmap="viridis")
        axs[1, 2].set_title("MC Dropout Confidence (1/Var)")
        axs[1, 2].axis('off')
        plt.colorbar(im, ax=axs[1, 2], fraction=0.046, pad=0.04)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        save_filename = os.path.join(output_dir, f'comparison_sample_{i}.png')
        plt.savefig(save_filename, dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Saved {num_samples} qualitative plots to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot qualitative depth model comparisons."
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help="Path to the analysis YAML config file.",
    )
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    plot_qualitative_comparison(config)
