from torchvision import transforms
from PIL import Image
import random

class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        return transforms.RandomCrop(self.size)(img)

class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return transforms.functional.hflip(img)
        return img

class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        return transforms.functional.normalize(img, mean=self.mean, std=self.std)

class ToTensor:
    def __call__(self, img):
        return transforms.ToTensor()(img)

def get_transforms():
    return transforms.Compose([
        RandomCrop(size=(224, 224)),
        RandomHorizontalFlip(p=0.5),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])