from pathlib import Path
import cv2
import random


def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold to binary
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Optional: morphological opening to remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    return image, dilated


def extract_logo_bounding_boxes(binary_image):
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter out very small contours
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 2000]
    return boxes


def crop_logos(image, boxes):
    logos = []
    for x, y, w, h in boxes:
        logo = image[y : y + h, x : x + w]
        logos.append(logo)
    return logos


def split_logos_from_file(image_path, draw_boxes=False):
    image, binary = preprocess_image(image_path)
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
    # canada_17.png
    combined_logo_path = Path("logos/canada/canada_6.png")

    logos, boxed_image = split_logos_from_file(combined_logo_path, draw_boxes=True)

    output_dir = combined_logo_path.parent.parent.parent / "split_logo"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Delete anything in the output directory
    for file in output_dir.iterdir():
        if file.is_file():
            file.unlink()
        elif file.is_dir():
            for subfile in file.iterdir():
                subfile.unlink()
            file.rmdir()

    # Save each cropped logo
    for i, logo in enumerate(logos):
        output_path = output_dir / f"logo_{i}.png"
        cv2.imwrite(str(output_path), logo)
        print(f"Saved logo {i} to {output_path}")

    # Save the image with drawn boxes
    boxed_output_path = output_dir / "boxed_original.png"
    if boxed_image is not None:
        cv2.imwrite(str(boxed_output_path), boxed_image)
        print(f"Saved boxed image to {boxed_output_path}")


if __name__ == "__main__":
    main()
