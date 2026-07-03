import random
from PIL import Image, ImageOps, ImageEnhance


def random_horizontal_flip(image: Image.Image) -> Image.Image:
    """Randomly flips the image horizontally with a 50% probability."""
    if random.random() > 0.5:
        return ImageOps.mirror(image)
    return image


def random_rotation(image: Image.Image, max_angle: float = 30.0) -> Image.Image:
    """Rotates the image by a random angle between -max_angle and max_angle."""
    angle = random.uniform(-max_angle, max_angle)
    return image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)


def random_brightness_contrast(image: Image.Image) -> Image.Image:
    """Applies random brightness and contrast adjustments."""
    if random.random() > 0.5:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(random.uniform(0.8, 1.2))
    if random.random() > 0.5:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(random.uniform(0.8, 1.2))
    return image


def augment_image(image: Image.Image) -> Image.Image:
    """Applies a sequence of random augmentations to the image."""
    image = random_horizontal_flip(image)
    image = random_rotation(image)
    image = random_brightness_contrast(image)
    return image
