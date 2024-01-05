# Helin Aslı Aksoy
# 150200705

from typing import List, Tuple
import numpy as np


def apply_filter(image_array: np.ndarray, kernel: np.ndarray, padding: List[List[int]]) -> np.ndarray:
    """ Apply a filter with the given kernel to the zero padded gray scaled (2D) input image array.
        **Note:** Kernels can be rectangular.
        **Do not** use ```np.convolve``` in this question.
        **Do not** use ```np.pad```. Use index assignment and slicing with numpy and do not loop
            over the pixels for padding.

    Args:
        image_array (np.ndarray): 2D Input array
        kernel (np.ndarray): 2D kernel array of odd edge sizes
        padding: (List[list[int]]): List of zero paddings. Example: [[3, 2], [1, 4]]. The first list
            [3, 3] determines the padding for the width of the image while [1, 4] determines the
            padding to apply to top and bottom of the image. The resulting image will have a shape
            of ((1 + H + 4), (3 + W + 2)).

    Raises:
        ValueError: If the length of kernel edges are not odd

    Returns:
        np.ndarray: Filtered array (May contain negative values)
    """

# Check if the kernel is odd
    if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise ValueError("Kernel is not odd")

    # Pad the image with zeros to handle the border pixels
    padded_image = np.pad(image_array, padding,
                          'constant', constant_values=(0, 0))

    # Keep track of the output image
    output_image = np.zeros(image_array.shape)

    # Get the center pixel of the kernel
    kernel_center = np.array(kernel.shape) // 2

    # Loop over the image and apply the filter
    for x in range(image_array.shape[0]):
        for y in range(image_array.shape[1]):
            # Get the patch of the image
            patch = padded_image[x: x +
                                 kernel.shape[0], y: y + kernel.shape[1]]

            # Multiply the patch with the kernel and sum the result
            output_image[x, y] = np.sum(patch * kernel)

    return output_image
    raise NotImplementedError

    raise NotImplementedError


def gaussian_filter(image_arr: np.ndarray, kernel_size: Tuple[int, int], sigma: float) -> np.ndarray:
    """ Apply Gauss filter that is centered and has the shared standard deviation ```sigma```
    **Note:** Remember to normalize kernel before applying.
    **Note:** You can use ```np.meshgrid``` (once again) to generate Gaussian kernels
    Args:
        image_arr (np.ndarray): 2D Input array of shape (H, W)
        kernel_size (Tuple[int]): 2D kernel size (H, W)
        sigma (float): Standard deviation
    Returns:
        ImageType: Filtered Image array
    """

    x, y = np.meshgrid(np.linspace(-kernel_size[1] // 2, kernel_size[1] // 2, kernel_size[1]),
                       np.linspace(-kernel_size[0] // 2, kernel_size[0] // 2, kernel_size[0]))
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel = kernel / np.sum(kernel)

    padding = ((kernel_size[0] // 2, kernel_size[0] // 2),
               (kernel_size[1] // 2, kernel_size[1] // 2))

    return apply_filter(image_arr, kernel, padding)

    raise NotImplementedError


def sobel_vertical(image_array: np.ndarray) -> np.ndarray:
    """ Return the output of the vertical Sobel operator with same padding.
        **Note**: This function may return negative values
    Args:
        image_array (np.ndarray): 2D Input array of shape (H, W)
    Returns:
        np.ndarray: Derivative array of shape (H, W).
    """
    kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Pad the image with zeros to handle the border pixels
    padding = (1, 1)

    return apply_filter(image_array, kernel, padding)
    raise NotImplementedError


def sobel_horizontal(image_array: np.ndarray) -> np.ndarray:
    """ Return the output of the horizontal Sobel operator with same padding.
        **Note**: This function may return negative values
    Args:
        image_array (np.ndarray): 2D Input array of shape (H, W)
    Returns:
        np.ndarray: Derivative array of shape (H, W).
    """
    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    # Pad the image with zeros to handle the border pixels
    padding = (1, 1)

    return apply_filter(image_array, kernel, padding)
    raise NotImplementedError
