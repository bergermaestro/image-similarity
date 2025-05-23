from pathlib import Path
from PIL import Image
from cv2.typing import MatLike

from utils import cv2_to_pil, pil_to_cv2


def _save_image_fn(image: Image.Image, file_name: str) -> None:
    output_path = Path("tests/output/cleanup_image")
    output_path.mkdir(parents=True, exist_ok=True)
    image.save(output_path / file_name)


def cleanup_image(image: MatLike, output_size=(256, 256), save_image=False) -> MatLike:
    img = cv2_to_pil(image).convert("RGBA")

    # img = Image.open(image).convert("RGBA")  # Ensure RGBA mode

    # Create a white background image with the desired output size
    white_bg = Image.new("RGBA", output_size, (255, 255, 255, 255))

    # Scale the image while maintaining aspect ratio
    img.thumbnail(output_size, Image.Resampling.BOX)

    # Calculate position to center the image
    x_offset = (output_size[0] - img.size[0]) // 2
    y_offset = (output_size[1] - img.size[1]) // 2

    # Paste the scaled image onto the white background
    white_bg.paste(img, (x_offset, y_offset), img)

    # Convert back to RGB (remove alpha channel)
    img_rgb = white_bg.convert("RGB")

    if save_image:
        _save_image_fn(img_rgb, f"4_rgb_convert_{id(img)}.png")

    return pil_to_cv2(img_rgb)


def add_white_bg(image: MatLike, save_image=False) -> MatLike:
    img = cv2_to_pil(image).convert("RGBA")

    # Create a white background image with the desired output size
    white_bg = Image.new("RGBA", (img.width, img.height), (255, 255, 255, 255))

    white_bg.paste(img, (0, 0), img)
    img_rgb = white_bg.convert("RGB")

    if save_image:
        _save_image_fn(img_rgb, f"5_white_bg_{id(img)}.png")

    return pil_to_cv2(img_rgb)


def fit_to_square(image: MatLike, output_size=(256, 256), save_image=False) -> MatLike:
    img = cv2_to_pil(image).convert("RGBA")
    white_bg = Image.new("RGBA", output_size, (255, 255, 255, 255))

    max_width = int(output_size[0] * 0.9)
    max_height = output_size[1]

    w, h = img.size
    if w > max_width or h > max_height or w < max_width:
        scale_ratio = min(max_width / w, max_height / h)
        new_size = (int(w * scale_ratio), int(h * scale_ratio))
        img = img.resize(new_size, Image.Resampling.BOX)

    x_offset = (output_size[0] - img.size[0]) // 2
    y_offset = (output_size[1] - img.size[1]) // 2

    white_bg.paste(img, (x_offset, y_offset), img)

    img_rgb = white_bg.convert("RGB")

    if save_image:
        _save_image_fn(img_rgb, f"4_rgb_convert_{id(img)}.png")

    return pil_to_cv2(img_rgb)
