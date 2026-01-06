from PIL import Image
import numpy as np
import random

class ImageAugmentor:
    def __init__(self, background_images, leaf_images):
        self.background_images = background_images
        self.leaf_images = leaf_images

    def augment_image(self, background_path, leaf_path, position):
        background = Image.open(background_path).convert("RGBA")
        leaf = Image.open(leaf_path).convert("RGBA")

        # Resize leaf image if necessary
        leaf = leaf.resize((int(leaf.width * 0.5), int(leaf.height * 0.5)))

        # Paste leaf onto background at the specified position
        background.paste(leaf, position, leaf)
        return background

    def generate_augmented_images(self, num_images, output_dir):
        for i in range(num_images):
            background_path = random.choice(self.background_images)
            leaf_path = random.choice(self.leaf_images)

            # Random position for the leaf
            background = Image.open(background_path)
            max_x = background.width - 1
            max_y = background.height - 1
            position = (random.randint(0, max_x), random.randint(0, max_y))

            augmented_image = self.augment_image(background_path, leaf_path, position)
            augmented_image.save(f"{output_dir}/augmented_image_{i}.png")

# Example usage
if __name__ == "__main__":
    background_images = ["path/to/background1.png", "path/to/background2.png"]
    leaf_images = ["path/to/leaf1.png", "path/to/leaf2.png"]
    output_dir = "data/augmented"

    augmentor = ImageAugmentor(background_images, leaf_images)
    augmentor.generate_augmented_images(num_images=10, output_dir=output_dir)