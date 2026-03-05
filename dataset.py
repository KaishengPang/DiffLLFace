import torchvision
from torchvision import transforms
import torch
from PIL import Image
import io
import random
import os
import webdataset as wds  # pylint: disable=import-outside-toplevel
import copy
import glob
from torch.utils.data import Dataset
import pandas as pd

# def create_webdataset(data_dir, image_size=[512,512], random_flip=True):
def create_webdataset(jpg_dir, hint_dir, image_size=[256,256], random_flip=True):
    return ImageDataset(jpg_dir, hint_dir, image_size, random_flip)

class RandomNoise(object):
    def __init__(self, max_poisson=0.8, max_gaussian=0.4):
        self.max_poisson = max_poisson
        self.max_gaussian = max_gaussian

    def __call__(self, image_tensor):

        # Poisson Noise
        noise_poisson = torch.poisson(image_tensor) 
        noise_poisson_sign = torch.randint(low=0, high=2, size=image_tensor.shape) * 2 - 1
        noise_poisson = noise_poisson * noise_poisson_sign * random.uniform(0, self.max_poisson)

        # Gaussian Noise
        noise_gaussian = torch.randn(image_tensor.shape)
        noise_gaussian = noise_gaussian * random.uniform(0, self.max_gaussian)

        image_noise_tensor = image_tensor + noise_gaussian + noise_poisson 
        return torch.clamp(image_noise_tensor, 0, 1)


class ImageDataset(Dataset):
    def __init__(self, jpg_dir, hint_dir, image_size=[256,256], random_flip=True):
        self.jpg_file_list = sorted(glob.glob(jpg_dir))
        self.hint_file_list = sorted(glob.glob(hint_dir))
        # self.image_file_list = glob.glob(data_dir)
        assert len(self.jpg_file_list) == len(self.hint_file_list), "Number of jpg and hint images must be the same"
        # random.shuffle(self.image_file_list)
        # random.shuffle(self.jpg_file_list)
        # random.shuffle(self.hint_file_list)  # 保持两个列表的随机性一致
        # print("Found", len(self.image_file_list), "image files")
        print("Found", len(self.jpg_file_list), "jpg files in", jpg_dir)
        print("Found", len(self.hint_file_list), "hint files in", hint_dir)
        # assert len(self.image_file_list) > 0

        if random_flip:
            self.image_transform = transforms.Compose([
                # transforms.Resize(image_size[0]*3//2),
                # transforms.RandomCrop((image_size[0],image_size[1])),
                # transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),]
            )
        else:
            self.image_transform = transforms.Compose([
                # transforms.Resize(image_size[0]*3//2),
                # transforms.RandomCrop((image_size[0],image_size[1])),
                transforms.ToTensor()]
            )
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # self.apply_noise = torchvision.transforms.RandomApply([RandomNoise()], p=0.5)

    def __len__(self):
        # return len(self.image_file_list)
        return len(self.jpg_file_list)
    def process(self, jpg_path, hint_path):
        # print("preprocess!!!")
        output = {}

        # image = Image.open(video_path).convert('RGB')
        # image_tensor = self.image_transform(image)
        jpg_image = Image.open(jpg_path).convert('RGB')
        jpg_tensor = self.image_transform(jpg_image)

        hint_image = Image.open(hint_path).convert('RGB')
        hint_tensor = self.image_transform(hint_image)

        output["jpg"] = self.normalize(copy.deepcopy(jpg_tensor))
        # output["hint"] = self.apply_noise(image_tensor) 
        output["hint"] = hint_tensor
        output["txt"] = ""
        return output
    
    def __getitem__(self, idx):
        jpg_path = self.jpg_file_list[idx]
        hint_path = self.hint_file_list[idx]
        return self.process(jpg_path, hint_path)