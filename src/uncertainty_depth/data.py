import cv2
import torch
from torch.utils.data import Dataset


class DepthDataset(Dataset):
    def __init__(self, dataframe, new_size=(224, 224)):
        self.dataframe = dataframe
        self.new_size = new_size

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0].strip()[1:]
        depth_path = self.dataframe.iloc[idx, 1].strip()[1:]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.new_size, interpolation=cv2.INTER_LINEAR)
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth.ndim == 3 and depth.shape[2] == 4:
            depth = depth[:, :, 0]
        depth = torch.tensor(depth, dtype=torch.float32).unsqueeze(0)

        return {"image": image, "depth": depth}
