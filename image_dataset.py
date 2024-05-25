import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import glob

class ImageDataset(Dataset):
    def __init__(self, directory_path, image_size, transform=None):
        self.directory_path = directory_path
        self.image_paths = []
        self.transform = transform
        self.image_size = image_size
        
        extensions = ["*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp", "*.tiff"]
        for ext in extensions:
            files = glob.glob(os.path.join(directory_path, ext))
            self.image_paths.extend(files)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).resize((self.image_size, self.image_size))
        
        if self.transform:
            image = self.transform(image)
        
        return image