import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from models.backbone import SurgicalDINOEvidential
from torch.utils.data import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DepthDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, new_size=(224, 224)):
        self.dataframe = dataframe
        self.new_size = new_size

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0][1:]
        depth_path = self.dataframe.iloc[idx, 1][1:]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.new_size, interpolation=cv2.INTER_LINEAR)
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth.ndim == 3 and depth.shape[2] == 4:
            depth = depth[:, :, 0]
        depth = cv2.resize(depth, self.new_size, interpolation=cv2.INTER_NEAREST)
        depth = torch.tensor(depth, dtype=torch.float32).unsqueeze(0)
        return {"image": image, "depth": depth}


test_file = "kaggle/input/mde-dataset-path/endoslam_test_files_with_gt_Stomach.txt"
test_df = pd.read_csv(test_file, sep="\t", header=None)[:5]
test_loader = DataLoader(DepthDataset(test_df, (224, 224)), batch_size=1, shuffle=False)

model = SurgicalDINOEvidential(
    backbone_size="base", r=4, image_shape=(224, 224), decode_type="linear4", lam=0.2
).to(device)
checkpoint = torch.load(
    "checkpoints/model_epoch_1.pth", map_location=device, weights_only=False
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

for i, batch in enumerate(test_loader):
    with torch.no_grad():
        img = batch["image"].to(device)
        gt_depth = batch["depth"][0, 0].cpu().numpy()
        out = model(pixel_values=img)
        pred_depth = out["predicted_depth"][0, 0].cpu().numpy()
        abs_diff = np.abs(pred_depth - gt_depth)
        evidence = out["evidence"][0, 0].cpu().numpy()
        original_img = (batch["image"][0].permute(1, 2, 0).cpu().numpy() * 255).astype(
            np.uint8
        )

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs[0, 0].imshow(original_img)
    axs[0, 0].set_title("Input Image")
    axs[0, 0].axis('off')

    pred_plot = axs[0, 1].imshow(pred_depth, cmap="magma")
    axs[0, 1].set_title("Predicted Depth")
    axs[0, 1].axis('off')
    plt.colorbar(pred_plot, ax=axs[0, 1], fraction=0.046, pad=0.04)

    diff_plot = axs[1, 0].imshow(abs_diff, cmap="inferno")
    axs[1, 0].set_title("Absolute Depth Error")
    axs[1, 0].axis('off')
    plt.colorbar(diff_plot, ax=axs[1, 0], fraction=0.046, pad=0.04)

    evidence_plot = axs[1, 1].imshow(evidence, cmap="plasma")
    axs[1, 1].set_title("Evidence")
    axs[1, 1].axis('off')
    plt.colorbar(evidence_plot, ax=axs[1, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(f"output_{i}.png", dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

    print(f"Saved output_{i}.png")
    print(f"Abs Error range: [{abs_diff.min():.3f}, {abs_diff.max():.3f}]")
    print(f"Evidence range: [{evidence.min():.3f}, {evidence.max():.3f}]")
    print("-" * 50)

    if i == 4:
        break
