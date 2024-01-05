from PIL.Image import Image as ImageType
from PIL import Image
import numpy as np


def image_to_array(image: ImageType) -> np.ndarray:
    """ Helper function """
    return np.asarray(image).astype(np.float32)


def array_to_image(image_array: np.ndarray) -> ImageType:
    """ Helper function """
    return Image.fromarray(np.uint8(np.clip(image_array, 0, 255)))
