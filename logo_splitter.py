from pathlib import Path
from typing import List
import cv2
import random

from cleanup import cleanup_image
import cleanup
from utils import load_image, save_cv2_image
from cv2.typing import MatLike, Rect

CONTOUR_THRESHOLD = 10
DILATE_ITERATIONS = 3
# settings for canada logo
# CONTOUR_THRESHOLD = 1000
# DILATE_ITERATIONS = 2


def get_image_binary(image: MatLike):
    # image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold to binary
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Optional: morphological opening to remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    dilated = cv2.dilate(thresh, kernel, iterations=DILATE_ITERATIONS)

    return dilated


def extract_logo_bounding_boxes(binary_image: MatLike):
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter out very small contours
    boxes = [
        cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > CONTOUR_THRESHOLD
    ]
    return boxes


def crop_logos(image: MatLike, boxes: List[Rect]) -> list[MatLike]:
    logos: List[MatLike] = []
    for x, y, w, h in boxes:
        logo = image[y : y + h, x : x + w]
        logos.append(logo)
    return logos


def split_logo(
    image: MatLike, draw_boxes=False
) -> tuple[list[MatLike], MatLike | None]:
    binary = get_image_binary(image)
    boxes = extract_logo_bounding_boxes(binary)
    logos = crop_logos(image, boxes)

    boxed_image = None
    if draw_boxes:
        boxed_image = image.copy()
        for i, (x, y, w, h) in enumerate(boxes):
            color = [random.randint(0, 255) for _ in range(3)]
            cv2.rectangle(boxed_image, (x, y), (x + w, y + h), color, 2)

    return logos, boxed_image


def main():
    # canada_17.png, canada_6.png
    canada_logo_dir = Path("sample_logos/canada")
    logo_files = list(canada_logo_dir.glob("*.png"))

    output_dir = Path("split_logo")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Delete anything in the output directory
    for file in output_dir.iterdir():
        if file.is_file():
            file.unlink()
        elif file.is_dir():
            for subfile in file.iterdir():
                subfile.unlink()
            file.rmdir()

    for logo_path in logo_files:
        logo = load_image(logo_path)

        logo_with_bg = cleanup.add_white_bg(logo, save_image=False)
        resized_logo = cleanup.fit_to_square(
            logo_with_bg, output_size=(1024, 1024), save_image=False
        )
        split_logos, boxed_image = split_logo(resized_logo, draw_boxes=True)

        print(f"Processing {logo_path}..., {len(split_logos)} logos found")

        # logo = cleanup_image(logo, save_image=False)
        # split_logos, boxed_image = split_logo(logo, draw_boxes=True)

        # Save each cropped logo
        base_name = logo_path.stem
        # for i, logo in enumerate(split_logos):
        #     output_path = output_dir / f"{base_name}_logo_{i}.png"

        #     logo_fitted = cleanup.fit_to_square(logo, output_size=(256, 256))
        #     save_cv2_image(logo_fitted, output_path)

        # Save the image with drawn boxes
        boxed_output_path = output_dir / f"{base_name}_boxed_original.png"
        if boxed_image is not None:
            save_cv2_image(boxed_image, boxed_output_path)
            print(f"Saved boxed image to {boxed_output_path}")


if __name__ == "__main__":
    main()
