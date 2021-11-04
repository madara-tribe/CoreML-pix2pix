import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from cfg import Cfg

cfg = Cfg
transform = [
      transforms.Resize((cfg.img_height, cfg.img_width), Image.BICUBIC),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
      
class ImageDataset(Dataset):
    def __init__(self, pathA, val_path=None, transform_=None, mode='train'):
        self.transform_ = transform_
        if self.transform_ is True:
            self.transform = transforms.Compose(transform)
        self.files = sorted(glob.glob(pathA))
        if mode == "train":
            self.files.extend(sorted(glob.glob(val_path)))

    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)])
        w, h = img.size
        img_A = img.crop((0, 0, w / 2, h))
        img_B = img.crop((w / 2, 0, w, h))

        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")
        
        if self.transform_ is True:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)
        
        return {"A": img_A, "B": img_B}

    def __len__(self):
        return len(self.files)



