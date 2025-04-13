import os
from pathlib import Path
import random
from PIL import Image
import numpy as np

# Directory to save augmented images
AUGMENTED_DIR = "logos/augmented_canada_logos"
if not os.path.exists(AUGMENTED_DIR):
    os.makedirs(AUGMENTED_DIR)


# Function to shift the hue of an image
def shift_hue(img, hue_shift_factor=0.1):
    img = img.convert("HSV")  # Convert image to HSV mode
    np_img = np.array(img)  # Convert to a NumPy array
    np_img[..., 0] = (
        np_img[..., 0] + hue_shift_factor * 255
    ) % 255  # Shift the hue channel
    img = Image.fromarray(np_img, "HSV").convert("RGB")  # Convert back to RGB
    return img


# Function to convert image to black and white
def convert_to_bw(img):
    return img.convert("L").convert(
        "RGB"
    )  # Convert to grayscale and back to RGB to keep 3 channels


# Function to skew aspect ratio slightly
def skew_aspect_ratio(img, factor=0.1):
    width, height = img.size
    new_width = int(width * (1 + random.uniform(-factor, factor)))
    new_height = int(height * (1 + random.uniform(-factor, factor)))
    return img.resize((new_width, new_height))


# Function to add compression artifacts
def add_compression_artifacts(img, quality=50):
    # Convert image to RGB if it is in a different mode
    if img.mode != "RGB":
        img = img.convert("RGB")

    img_path = "temp_compressed.jpg"
    img.save(img_path, "JPEG", quality=quality)
    compressed_img = Image.open(img_path)
    os.remove(img_path)  # Clean up
    return compressed_img


# Function to apply all augmentations and save the images
def augment_images(image_paths, num_augmentations=5):
    for image_path in image_paths:
        img = Image.open(image_path).convert("RGBA")  # Ensure image is in RGBA mode
        img_name = Path(image_path).stem

        # Preprocess: Place image on a white background if it has transparency
        if img.mode == "RGBA":
            white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(white_bg, img).convert("RGB")

        # Save the unaugmented version
        img.save(os.path.join(AUGMENTED_DIR, f"{img_name}_original.png"))

        # Apply augmentations
        for i in range(num_augmentations):
            augmented_img = img

            # Randomly choose an augmentation
            augmentation_type = random.choice(["hue", "bw", "skew", "compression"])

            if augmentation_type == "hue":
                augmented_img = shift_hue(augmented_img)
            elif augmentation_type == "bw":
                augmented_img = convert_to_bw(augmented_img)
            elif augmentation_type == "skew":
                augmented_img = skew_aspect_ratio(augmented_img)
            elif augmentation_type == "compression":
                augmented_img = add_compression_artifacts(augmented_img)

            # Save augmented image
            augmented_img.save(os.path.join(AUGMENTED_DIR, f"{img_name}_aug_{i}.png"))


# Example: Apply augmentations to a selection of images from a folder
def main():
    # Your logos directory
    logos_dir = "logos/canada"  # Change this to your actual directory path
    image_paths = [
        os.path.join(logos_dir, img)
        for img in os.listdir(logos_dir)
        if img.endswith((".png", ".jpg", ".jpeg"))
    ]

    augment_images(
        image_paths, num_augmentations=3
    )  # 3 augmentations per image for example


if __name__ == "__main__":
    main()
