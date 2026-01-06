import pytest
from src.augmentation.image_augmentor import augment_image
from src.augmentation.transforms import RandomCrop, RandomFlip
from PIL import Image
import numpy as np

@pytest.fixture
def sample_image():
    return Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))

def test_augment_image(sample_image):
    augmented_image = augment_image(sample_image, [RandomCrop(size=(80, 80)), RandomFlip()])
    assert augmented_image.size == (80, 80)

def test_random_crop(sample_image):
    crop_transform = RandomCrop(size=(80, 80))
    cropped_image = crop_transform(sample_image)
    assert cropped_image.size == (80, 80)

def test_random_flip(sample_image):
    flip_transform = RandomFlip()
    flipped_image = flip_transform(sample_image)
    assert flipped_image.size == sample_image.size  # Size should remain the same after flip

def test_augmentation_pipeline(sample_image):
    transforms = [RandomCrop(size=(80, 80)), RandomFlip()]
    augmented_image = augment_image(sample_image, transforms)
    assert augmented_image.size == (80, 80)  # Final size should match the crop size