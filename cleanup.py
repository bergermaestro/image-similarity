from pathlib import Path
from PIL import Image


def cleanup_image(image_path: Path, output_size=(256, 256), save_image=False):
    img = Image.open(image_path).convert("RGBA")  # Ensure RGBA mode

    # Create a white background image with the desired output size
    white_bg = Image.new("RGBA", output_size, (255, 255, 255, 255))

    # Scale the image while maintaining aspect ratio
    img.thumbnail(output_size, Image.Resampling.LANCZOS)

    # Calculate position to center the image
    x_offset = (output_size[0] - img.size[0]) // 2
    y_offset = (output_size[1] - img.size[1]) // 2

    # Paste the scaled image onto the white background
    white_bg.paste(img, (x_offset, y_offset), img)

    # Convert back to RGB (remove alpha channel)
    img_rgb = white_bg.convert("RGB")

    if save_image:
        file_name = str(image_path).split("/")[-1]

        Path("tmp").mkdir(parents=True, exist_ok=True)
        img_rgb.save(f"tmp/{file_name}")

    return img_rgb
