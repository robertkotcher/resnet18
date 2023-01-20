from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

import os
import json
import torch

# For classification, labels and data _had_ the following properties:
# 
# Labels shape:  torch.Size([batch_size])
# Dataset shape:  torch.Size([batch_size, 1, 28, 28])
# 
# Labels dtype:  torch.int64
# Dataset dtype:  torch.float32
# 
# ---
# 
# Now:
# 
# img: torch.Size([3, 1024, 1024])
#       dtype=torch.float32
# 
# label:
class FocalLengthDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        count = 0
        for path in os.listdir(data_dir):
            p=os.path.join(data_dir, path)
            if os.path.isfile(p) and path[0] != ".":
                count += 1
        self.count = int(count / 2)

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        # get image data
        img = Image.open(os.path.join(self.data_dir, f"{idx}.cam_default.f_1.rgb.png"))
        convert_tensor = transforms.ToTensor()
        img_tensor = convert_tensor(img)

        # get focal length
        with open(os.path.join(self.data_dir, f"{idx}.cam_default.f_1.info.json")) as f:
            j = json.loads(f.read())
            focal_length_mm = j["camera"]["focal_length_mm"]
            return img_tensor, torch.tensor(focal_length_mm)