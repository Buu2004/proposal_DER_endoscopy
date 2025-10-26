import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import spearmanr
from src.uncertainty_depth.data import DepthDataset
from src.uncertainty_depth.metrics import compute_metrics, compute_sample_rmse
from src.uncertainty_depth.utils import (
    get_edl_outputs,
    get_mcd_outputs,
    load_model,
    plot_comparison_scatter,
)
from torch.utils.data import DataLoader
from tqdm import tqdm


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    return torch.utils.data.default_collate(batch)


def run_evaluation(mode, model, loader, device, cfg_eval):
    all_preds_masked = []
    all_gt_masked = []
    sample_metrics_list = []

    mc_iterations = cfg_eval.get('mc_iterations', 10)
    epsilon = cfg_eval.get('epsilon', 1e-6)

    pbar = tqdm(loader, desc=f"Evaluating ({mode})")
    with torch.no_grad():
        for batch in pbar:
            if batch is None:
                continue

            img = batch["image"].to(device)
            gt = batch["depth"].to(device)
            gt_numpy = gt.cpu().numpy()[0, 0]

            valid_mask_numpy = gt_numpy > epsilon
            if np.sum(valid_mask_numpy) == 0:
                continue

            if mode == 'evidential':
                outputs = get_edl_outputs(model, img, gt.shape)
                pred_numpy = outputs['pred_depth']

                # Confidence is evidence
                sample_conf = np.mean(outputs['evidence'][valid_mask_numpy])
                # Uncertainty is sum of aleatoric and epistemic
                unc_map = outputs['aleatoric_unc'] + outputs['epistemic_unc']
                sample_unc = np.mean(unc_map[valid_mask_numpy])

            else:  # mc_dropout
                outputs = get_mcd_outputs(model, img, gt.shape, mc_iterations)
                pred_numpy = outputs['pred_depth']
                variance_numpy = outputs['variance']

                # Confidence is inverse variance
                inv_var = 1.0 / (variance_numpy[valid_mask_numpy] + epsilon)
                sample_conf = np.mean(inv_var)
                # Uncertainty is variance
                sample_unc = np.mean(variance_numpy[valid_mask_numpy])

            sample_rmse = compute_sample_rmse(pred_numpy, gt_numpy)
            sample_metrics_list.append(
                {
                    'sample_rmse': sample_rmse,
                    'sample_uncertainty': sample_unc,
                    'sample_confidence': sample_conf,
                }
            )

            pred_masked = pred_numpy[valid_mask_numpy]
            gt_masked = gt_numpy[valid_mask_numpy]

            all_preds_masked.append(pred_masked)
            all_gt_masked.append(gt_masked)

    all_preds_flat = np.concatenate(all_preds_masked)
    all_gt_flat = np.concatenate(all_gt_masked)
    overall_metrics = compute_metrics(all_preds_flat, all_gt_flat)

    sample_df = pd.DataFrame(sample_metrics_list)
    return overall_metrics, sample_df


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg_data = config['data']
    cfg_model = config['model']
    cfg_eval = config['evaluation']

    cfg_model['image_size'] = cfg_data['image_size']

    test_df = pd.read_csv(cfg_data['test_file'], sep="\t", header=None)
    test_dataset = DepthDataset(test_df, tuple(cfg_data['image_size']))
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # must be 1 for per-sample metrics
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
    )

    results_all_modes = {}

    modes_to_run = []
    if cfg_eval['run_evidential']:
        modes_to_run.append('evidential')
    if cfg_eval['run_mc_dropout']:
        modes_to_run.append('mc_dropout')

    for mode in modes_to_run:
        checkpoint_path = cfg_eval[f'{mode}_checkpoint']
        model = load_model(
            mode,
            checkpoint_path,
            cfg_model,
            device,
            dropout_p=cfg_model.get('dropout_p', 0.1),
        )
        if model is None:
            continue

        print(f"Starting evaluation for: {mode}")
        start_time = time.time()

        overall_metrics, sample_df = run_evaluation(
            mode, model, test_loader, device, cfg_eval
        )

        end_time = time.time()
        total_eval_time = end_time - start_time
        num_samples = len(sample_df)
        avg_time = total_eval_time / num_samples if num_samples > 0 else 0

        results_all_modes[mode] = (overall_metrics, sample_df, avg_time)

        print("\n" + "=" * 30)
        print(f"       RESULTS for {mode.upper()}       ")
        print("=" * 30)
        print("\n--- Overall Metrics ---")
        print(f"  RMSE     : {overall_metrics['rmse']:.4f}")
        print(f"  AbsRel   : {overall_metrics['abs_rel']:.4f}")
        print(f"  a1       : {overall_metrics['a1']:.4f}")
        print("\n--- Evaluation Time ---")
        print(f"  Total time : {total_eval_time:.2f} seconds")
        print(f"  Num samples: {num_samples}")
        print(f"  Avg/sample : {avg_time:.4f} seconds/sample")
        print("=" * 30 + "\n")

    if 'evidential' in results_all_modes and 'mc_dropout' in results_all_modes:
        df_edl = results_all_modes['evidential'][1].copy()
        df_mcd = results_all_modes['mc_dropout'][1].copy()

        scaler = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6)
        df_edl['confidence_normalized'] = scaler(df_edl['sample_confidence'])
        df_mcd['confidence_normalized'] = scaler(df_mcd['sample_confidence'])

        corr_edl, p_edl = spearmanr(
            df_edl['sample_rmse'], df_edl['confidence_normalized']
        )
        corr_mcd, p_mcd = spearmanr(
            df_mcd['sample_rmse'], df_mcd['confidence_normalized']
        )

        print("\n" + "=" * 50)
        print("   SPEARMAN CORRELATION (RMSE vs. Confidence)   ")
        print("=" * 50)
        print(f"  EDL (DER)  : {corr_edl:+.4f} (p-value: {p_edl:.2e})")
        print(f"  MC Dropout : {corr_mcd:+.4f} (p-value: {p_mcd:.2e})")
        print("=" * 50 + "\n")

        save_path = os.path.join(
            config.get('output_dir', 'results'),
            cfg_eval.get('scatter_plot_name', 'rmse_vs_confidence.png'),
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plot_comparison_scatter(df_edl, df_mcd, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate depth estimation models.")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help="Path to the analysis YAML config file.",
    )
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    main(config)
