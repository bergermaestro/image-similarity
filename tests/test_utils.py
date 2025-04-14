# import pytest


# # Example utility function to be tested
# def example_utility_function(x, y):
#     return x + y


# # Test cases for the utility function
# @pytest.mark.parametrize(
#     "x, y, expected",
#     [
#         (1, 2, 3),
#         (0, 0, 0),
#         (-1, -1, -2),
#         (100, 200, 300),
#     ],
# )
# def test_example_utility_function(x, y, expected):
#     assert example_utility_function(x, y) == expected


from pathlib import Path
from cleanup import cleanup_image
from utils import cv2_to_pil, pil_to_cv2, load_image
from PIL import Image
import numpy as np
import cv2


def test_pil_to_cv2():
    # Create a sample PIL image
    pil_image = Image.new("RGB", (100, 100), color="red")

    # Convert to CV2 image
    cv2_image = pil_to_cv2(pil_image)

    # Set up output paths
    output_dir = Path("tests/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    pil_image_path = output_dir / "pil_image.png"
    cv2_image_path = output_dir / "cv2_image.png"

    # Save images
    pil_image.save(pil_image_path)
    cv2.imwrite(str(cv2_image_path), cv2_image)

    # Check the shape and color
    assert cv2_image.shape == (100, 100, 3)
    assert np.array_equal(cv2_image[0, 0], [0, 0, 255])  # OpenCV uses BGR format


def test_cv2_to_pil():
    # Create a sample CV2 image
    cv2_image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2_image[:] = (0, 255, 0)  # Green color

    # Convert to PIL image
    pil_image = cv2_to_pil(cv2_image)

    # Set up output paths
    output_dir = Path("tests/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    pil_image_path = output_dir / "pil_image.png"
    cv2_image_path = output_dir / "cv2_image.png"

    # Save images
    pil_image.save(pil_image_path)
    cv2.imwrite(str(cv2_image_path), cv2_image)

    # Check the mode and color
    assert pil_image.mode == "RGB"
    assert pil_image.size == (100, 100)
    assert np.array_equal(np.array(pil_image)[0, 0], [0, 255, 0])  # RGB format


def test_cleanup_image():
    REFERENCE_IMAGE_PATH = Path("logos/canada/canada_17.png")
    REFERENCE_IMAGE = cleanup_image(
        load_image(REFERENCE_IMAGE_PATH), output_size=(256, 256), save_image=True
    )


def test_load_image():
    REFERENCE_IMAGE_PATH = Path("tests/input/red_pil_image.png")
    image = load_image(REFERENCE_IMAGE_PATH)

    Path("tests/output").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        "tests/output/test_load_image.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    )
