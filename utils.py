from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from cv2.typing import MatLike
import cairosvg


def pil_to_cv2(image: Image.Image) -> MatLike:
    """
    Convert a PIL image to a CV2 image.
    """
    # Convert the PIL image to a NumPy array
    pil_image = np.array(image)

    # Convert RGB to BGR
    # cv2_image = cv2.cvtColor(pil_image, cv2.COLOR_RGB2BGR)
    cv2_image = pil_image

    return cv2_image


def cv2_to_pil(image: MatLike) -> Image.Image:
    """
    Convert a CV2 image to a PIL image.
    """
    # Convert BGR to RGB
    # cv2_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2_image = image

    # Convert the NumPy array to a PIL image
    pil_image = Image.fromarray(cv2_image)

    return pil_image


def load_image(image_path: Path) -> MatLike:
    # image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    # # Check if the image has an alpha channel
    # if image.shape[-1] == 4:
    #     # Convert BGRA to RGBA
    #     image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    # else:
    #     # Convert BGR to RGB
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # return image
    with Image.open(image_path) as img:
        # Preserve alpha if present
        if img.mode in ("RGBA", "LA") or (
            img.mode == "P" and "transparency" in img.info
        ):
            img = img.convert("RGBA")
        else:
            img = img.convert("RGB")

        arr = np.array(img)
        return arr


def save_image(image: MatLike | Image.Image, output_path: Path) -> None:
    """
    Save an image to the specified path.
    """
    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(image, Image.Image):
        image.save(output_path)
    else:
        # Save the CV2 image using cv2.imwrite
        cv2.imwrite(str(output_path), image)


def svg_to_png(svg_path: Path, png_path: Path, output_width: int = 300) -> None:
    """
    Convert an SVG file to PNG format.
    """
    # Ensure the output directory exists
    png_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert SVG to PNG using cairosvg
    cairosvg.svg2png(
        url=str(svg_path), write_to=str(png_path), output_width=output_width
    )
