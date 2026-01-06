def resize_image(image, target_size):
    """Resize the input image to the target size."""
    from PIL import Image
    return image.resize(target_size, Image.ANTIALIAS)

def normalize_image(image):
    """Normalize the image data to the range [0, 1]."""
    import numpy as np
    return np.array(image) / 255.0

def random_flip(image):
    """Randomly flip the image horizontally."""
    import random
    if random.choice([True, False]):
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    return image

def random_rotation(image, max_angle=30):
    """Randomly rotate the image by a certain angle."""
    import random
    angle = random.uniform(-max_angle, max_angle)
    return image.rotate(angle)

def augment_image(image):
    """Apply a series of augmentations to the input image."""
    image = random_flip(image)
    image = random_rotation(image)
    return image

def save_augmented_image(image, output_path):
    """Save the augmented image to the specified output path."""
    image.save(output_path)