import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from src.uncertainty_depth.data import DepthDataset
from src.uncertainty_depth.metrics import compute_metrics
from src.uncertainty_depth.models.backbone import SurgicalDINOUncertaintyModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train_one_epoch(model, loader, optimizer, scaler, epoch, device):
    model.train()
    running_loss = 0.0

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}", unit="batch")
    for batch in pbar:
        images = batch["image"].to(device)
        depths = batch["depth"].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            outputs = model(pixel_values=images, depth_gt=depths)
            loss = outputs['loss']

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return running_loss / len(loader)


def evaluate_evidential(model, loader, device):
    model.eval()
    total_loss = 0.0
    all_metrics = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            depths = batch["depth"].to(device)
            valid_mask = depths > 0

            outputs = model(pixel_values=images, depth_gt=depths)
            total_loss += outputs['loss'].item()

            pred_depth = outputs['predicted_depth'].detach()
            aleatoric_unc = outputs['aleatoric_uncertainty'].detach()
            epistemic_unc = outputs['epistemic_uncertainty'].detach()

            pred_depth = torch.nn.functional.interpolate(
                pred_depth, size=depths.shape[2:], mode='bilinear', align_corners=False
            )
            aleatoric_unc = torch.nn.functional.interpolate(
                aleatoric_unc,
                size=depths.shape[2:],
                mode='bilinear',
                align_corners=False,
            )
            epistemic_unc = torch.nn.functional.interpolate(
                epistemic_unc,
                size=depths.shape[2:],
                mode='bilinear',
                align_corners=False,
            )

            pred_depth = pred_depth[valid_mask]
            depths_masked = depths[valid_mask]

            avg_aleatoric = aleatoric_unc[valid_mask].mean().item()
            avg_epistemic = epistemic_unc[valid_mask].mean().item()

            batch_metrics = compute_metrics(pred_depth, depths_masked)
            all_metrics.append(
                {
                    **batch_metrics,
                    'aleatoric_unc': avg_aleatoric,
                    'epistemic_unc': avg_epistemic,
                }
            )

    avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
    return total_loss / len(loader), avg_metrics


def evaluate_mcdropout(model, loader, device, mc_iterations):
    model.train()  # Enable dropout
    total_loss = 0.0
    all_metrics = []
    loss_fn = nn.L1Loss()

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            depths = batch["depth"].to(device)
            valid_mask = depths > 0

            mc_predictions = []
            for _ in range(mc_iterations):
                outputs = model(pixel_values=images)
                mc_predictions.append(outputs['predicted_depth'].detach())

            mc_stack = torch.stack(mc_predictions, dim=0)
            pred_depth_mean = torch.mean(mc_stack, dim=0)
            pred_depth_variance = torch.var(mc_stack, dim=0)

            pred_depth_mean_interp = torch.nn.functional.interpolate(
                pred_depth_mean,
                size=depths.shape[2:],
                mode='bilinear',
                align_corners=False,
            )
            pred_depth_variance_interp = torch.nn.functional.interpolate(
                pred_depth_variance,
                size=depths.shape[2:],
                mode='bilinear',
                align_corners=False,
            )

            total_loss += loss_fn(
                pred_depth_mean_interp[valid_mask], depths[valid_mask]
            ).item()

            pred_depth = pred_depth_mean_interp[valid_mask]
            depths_masked = depths[valid_mask]
            avg_mc_variance = pred_depth_variance_interp[valid_mask].mean().item()

            batch_metrics = compute_metrics(pred_depth, depths_masked)
            all_metrics.append({**batch_metrics, 'mc_variance': avg_mc_variance})

    avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
    return total_loss / len(loader), avg_metrics


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg_data = config['data']
    cfg_model = config['model']
    cfg_train = config['training']
    UNCERTAINTY_MODE = config['mode']

    image_size = tuple(cfg_data['image_size'])

    run_name = cfg_train.get('run_name', f"{UNCERTAINTY_MODE}_run")
    log_dir = f"runs/{run_name}"
    checkpoint_dir = f"{config['output_dir']}/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)

    train_df = pd.read_csv(cfg_data['train_file'], sep="\t", header=None)
    test_df = pd.read_csv(cfg_data['test_file'], sep="\t", header=None)

    train_dataset = DepthDataset(train_df, new_size=image_size)
    test_dataset = DepthDataset(test_df, new_size=image_size)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg_train['batch_size'], shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg_train['batch_size'], shuffle=False, num_workers=2
    )

    model = SurgicalDINOUncertaintyModel(
        backbone_size=cfg_model['backbone_size'],
        r=cfg_model['r'],
        image_shape=image_size,
        decode_type=cfg_model['decode_type'],
        lam=cfg_model.get('lam', 0.2),
        mode=UNCERTAINTY_MODE,
        dropout_p=cfg_model.get('dropout_p', 0.1),
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=cfg_train['lr'], weight_decay=cfg_train['weight_decay']
    )
    scaler = torch.amp.GradScaler('cuda')
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg_train['epochs']
    )

    best_val_loss = float('inf')

    for epoch in range(1, cfg_train['epochs'] + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, epoch, device
        )

        if UNCERTAINTY_MODE == 'evidential':
            val_loss, val_metrics = evaluate_evidential(model, test_loader, device)
        else:
            val_loss, val_metrics = evaluate_mcdropout(
                model,
                test_loader,
                device,
                mc_iterations=cfg_train.get('mc_iterations_val', 5),
            )

        scheduler.step()

        print(f"\nEpoch {epoch}/{cfg_train['epochs']} (Mode: {UNCERTAINTY_MODE})")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Test  Loss: {val_loss:.4f}")
        print("  Metrics:")
        for k, v in val_metrics.items():
            print(f"    {k}: {v:.4f}")
        print("-" * 20)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Learning_Rate", scheduler.get_last_lr()[0], epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f"Val Metrics/{k}", v, epoch)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            model.save_parameters(f"{checkpoint_dir}/best_model.pth")

        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            },
            f"{checkpoint_dir}/model_epoch_{epoch}.pth",
        )

        if epoch == cfg_train['epochs']:
            model.save_parameters(f"{checkpoint_dir}/final_model.pth")

    print(f"Training complete. Checkpoints saved to {checkpoint_dir}")
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train depth estimation model.")
    parser.add_argument(
        '--config', type=str, required=True, help="Path to the YAML config file."
    )
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    main(config)
