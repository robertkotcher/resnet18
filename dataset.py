from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

import os
import json
import torch

#
# FocalLengthDataset
#
# loads a dataset from data_dir - expects to have a .png and .info in the format output
# from SAI's humans pipeline
#
class FocalLengthDataset(Dataset):
    def __init__(self, data_dir, factor=None):
        self.data_dir = data_dir
        count = 0
        for path in os.listdir(data_dir):
            p=os.path.join(data_dir, path)
            if os.path.isfile(p) and path[0] != ".":
                count += 1
        self.count = int(count / 2)
        self.factor = factor

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

            if self.factor != None:
                t = transforms.Compose([
                    # img_tensor.shape = [C, W, H]
                    transforms.RandomCrop((img_tensor.shape[1] // self.factor, img_tensor.shape[2] // self.factor)),
                    transforms.Resize(size=img_size)
                ])
                return t(img_tensor), torch.tensor(focal_length_mm * self.factor)
            else:
                return img_tensor, torch.tensor(focal_length_mm)

