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

    def __len__(self):
        return len(self.sharp_images)

    def __getitem__(self, index):
        sharp_img_path = os.path.join(self.sharp_dir, self.sharp_images[index])
        blur_img_path = os.path.join(self.blur_dir, self.sharp_images[index].replace("_S.jpg", "_M.jpg"))
        sharp_img = np.array(Image.open(sharp_img_path).convert("RGB"))
        blur_img = np.array(Image.open(blur_img_path).convert("RGB"), dtype=np.float32)

        if self.transform is not None:
            augmentations = self.transform(sharp_img=sharp_img, blur_img=blur_img)
            sharp_img = augmentations["sharp_img"]
            blur_img = augmentations["blur_img"]

        return sharp_img, blur_img
