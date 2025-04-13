import cv2
import numpy as np
from PIL import Image


def pil_to_cv2(image):
    """
    Convert a PIL image to a CV2 image.
    """
    # Convert the PIL image to a NumPy array
    image = np.array(image)

    # Convert RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image


def cv2_to_pil(image):
    """
    Convert a CV2 image to a PIL image.
    """
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert the NumPy array to a PIL image
    image = Image.fromarray(image)

    return image
