import cv2
import numpy as np
from pathlib import Path
import cleanup
from logo_splitter import split_logo
from utils import load_image, save_cv2_image
from cv2.typing import MatLike


def match_template_multiscale(
    image: MatLike,
    template: MatLike,
    output_path: Path,
    scales=np.linspace(0.5, 2.0, 20),
):
    # Convert to HSV to filter for red regions
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Define red color ranges in HSV
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    # Create red masks
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Apply the red mask to the image
    red_only = cv2.bitwise_and(image, image, mask=red_mask)

    # Convert to grayscale for template matching
    img_gray = cv2.cvtColor(red_only, cv2.COLOR_RGB2GRAY)

    best_val = -np.inf
    best_loc = None
    best_scale = None
    best_template = None
    best_w, best_h = 0, 0

    for scale in scales:
        # Resize template
        template_resized = cv2.resize(
            template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
        template_gray = cv2.cvtColor(template_resized, cv2.COLOR_RGB2GRAY)

        t_h, t_w = template_gray.shape
        if t_h > img_gray.shape[0] or t_w > img_gray.shape[1]:
            continue  # Skip if template is larger than image

        # Perform template matching
        result = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # Update the best match if the current result is better
        if max_val > best_val:
            best_val = max_val
            best_loc = max_loc
            best_scale = scale
            best_template = template_resized
            best_w, best_h = t_w, t_h

    # Draw the best match rectangle
    if best_loc is not None:
        top_left = best_loc
        bottom_right = (top_left[0] + best_w, top_left[1] + best_h)
        outlined_image = image.copy()
        cv2.rectangle(outlined_image, top_left, bottom_right, (0, 255, 0), 3)

        # Save the outlined image
        save_cv2_image(outlined_image, output_path)
        print(
            f"Best match at scale {best_scale:.2f} with score {best_val:.4f}. Saved to {output_path}"
        )
    else:
        print("No good match found.")


def keep_red_parts(image: MatLike, output_path: Path = None) -> MatLike:
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    save_cv2_image(image, Path("template_matching/red_parts_hsv.png"))

    # Define the red color ranges in HSV
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for the red regions
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # Combine the two masks to get the full red range
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Apply the red mask to the image to keep only the red regions
    red_only = cv2.bitwise_and(image, image, mask=red_mask)

    # Optionally, save the output for testing
    if output_path is not None:
        cv2.imwrite(str(output_path), red_only)

    return red_only


def main():
    # Example usage
    image_path = "sample_logos/canada/canada-16.png"  # Replace with your image path
    output_path = (
        "template_matching/output_red_only_image.png"  # Where to save the result
    )

    # Load the image
    image = load_image(Path(image_path))

    # Keep only the red parts
    red_parts = keep_red_parts(image, output_path=Path(output_path))

    # Optionally display the image to check
    # cv2.imshow("Red Parts", red_parts)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    logo_with_bg = cleanup.add_white_bg(red_parts, save_image=False)
    resized_logo = cleanup.fit_to_square(
        logo_with_bg, output_size=(1024, 1024), save_image=False
    )

    # cv2.imshow("Resized Logo", resized_logo)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    split_logos, boxed_image = split_logo(resized_logo, draw_boxes=True)

    cv2.imshow("Boxed Image", boxed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Example usage
if __name__ == "__main__":
    main()
