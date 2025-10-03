import os

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from data.dataset import DepthDataset
from metrics.metrics import compute_metrics
from models.backbone import SurgicalDINOEvidential
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


def evaluate(model, loader, device):
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
            pred_depth = torch.nn.functional.interpolate(
                pred_depth,
                size=depths.shape[2:],
                mode='bilinear',
                align_corners=False,
            )
            pred_depth = pred_depth[valid_mask]
            depths_masked = depths[valid_mask]

            batch_metrics = compute_metrics(pred_depth, depths_masked)
            all_metrics.append(batch_metrics)

    avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
    return total_loss / len(loader), avg_metrics


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_file = "kaggle/input/mde-dataset-path/endoslam_train_files_with_gt_Colon.txt"
    test_file = "kaggle/input/mde-dataset-path/endoslam_test_files_with_gt_Colon.txt"

    batch_size = 8
    epochs = 2
    lr = 1e-5
    weight_decay = 1e-4
    image_size = (224, 224)

    os.makedirs("checkpoints", exist_ok=True)
    writer = SummaryWriter(log_dir="runs/depth_estimation")

    train_df = pd.read_csv(train_file, sep="\t", header=None)
    test_df = pd.read_csv(test_file, sep="\t", header=None)

    train_dataset = DepthDataset(train_df, new_size=image_size)
    test_dataset = DepthDataset(test_df, new_size=image_size)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    model = SurgicalDINOEvidential(
        backbone_size="base",
        r=4,
        image_shape=image_size,
        decode_type="linear4",
        lam=0.2,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler('cuda')
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, epoch, device
        )
        val_loss, val_metrics = evaluate(model, test_loader, device)

        scheduler.step()

        print(f"\nEpoch {epoch}/{epochs}")
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

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_metrics": val_metrics,
            },
            f"checkpoints/model_epoch_{epoch}.pth",
        )

    writer.close()


if __name__ == "__main__":
    main()
