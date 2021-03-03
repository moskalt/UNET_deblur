import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class BlurDataset(Dataset):
    def __init__(self, sharp_dir, blur_dir, transform=None):
            self.sharp_dir = sharp_dir
            self.blur_dir = blur_dir
            self.transform = transform
            self.sharp_images = os.listdir(sharp_dir)
            self.blur_images = os.listdir(blur_dir)

    def __len__(self):
        return len(self.sharp_images)

    def __getitem__(self, index):
        sharp_img_path = os.path.join(self.sharp_dir, self.sharp_images[index])
        blur_img_path = os.path.join(self.blur_dir, self.blur_images[index])
        sharp_img = (Image.open(sharp_img_path).convert("RGB"))
        blur_img = (Image.open(blur_img_path).convert("RGB"))

        if self.transform is not None:
            sharp_img = self.transform(sharp_img)
            blur_img = self.transform(blur_img)

        return blur_img, sharp_img
