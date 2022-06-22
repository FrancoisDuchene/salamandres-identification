from typing import Tuple

import cv2
import numpy as np
import os

ITERATIONS = 1

# https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html


def morph_shape(shape_type: int) -> int:
    if shape_type == 0:
        return cv2.MORPH_RECT
    elif shape_type == 1:
        return cv2.MORPH_CROSS
    elif shape_type == 2:
        return cv2.MORPH_ELLIPSE
    else:
        return cv2.MORPH_RECT


def compute_kernel_size(erosion_size: int) -> int:
    erosion_size = 2 * erosion_size + 1
    erosion_size = clip(erosion_size, 1, 21)
    return erosion_size


def build_kernel(erosion_size: int) -> Tuple[np.ndarray, int]:
    erosion_size = compute_kernel_size(erosion_size)
    kernel = np.ones((2 * erosion_size + 1, 2 * erosion_size + 1), np.uint8)
    return kernel, erosion_size


def clip(x: int, min_val: int, max_val: int) -> int:
    """
    clip value x between min and max. If x < min, then min is returned, if x > max, then max is returned,
    else x is returned
    """
    if x < min_val:
        return min_val
    else:
        if x > max_val:
            return max_val
        else:
            return x


def _apply_morphology_ex(image: np.ndarray, kernel_size: int, cv2_morph: int) -> np.ndarray:
    img_cpy = image.copy()
    kernel, _ = build_kernel(kernel_size)
    morphed_image = cv2.morphologyEx(img_cpy, cv2_morph, kernel, iterations=ITERATIONS)
    return morphed_image


def opening(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    apply an opening operation (erode+dilate) on the given image and returns the given image
    :param image:
    :param kernel_size: must be between [0,10]. The kernel size is always an even number included between 0 and 21,
        it is chosen for parameter 0 to 10. (example: erosion_size 3 -> 7, 4 -> 9)
    :return: an opened copy of the given image
    """
    return _apply_morphology_ex(image, kernel_size, cv2.MORPH_OPEN)


def closing(image: np.ndarray, kernel_size: int) -> np.ndarray:
    return _apply_morphology_ex(image, kernel_size, cv2.MORPH_CLOSE)


def erosion(image: np.ndarray, simple_erode: bool = True, shape_type: int = 0, erosion_size: int = 1) -> np.ndarray:
    img_cpy = image.copy()
    kernel, erosion_size = build_kernel(erosion_size)
    if simple_erode:
        eroded_image = cv2.erode(img_cpy, kernel=kernel, iterations=ITERATIONS)
    else:
        element = cv2.getStructuringElement(morph_shape(shape_type), (erosion_size, erosion_size))
        eroded_image = cv2.erode(img_cpy, element, iterations=ITERATIONS)
    return eroded_image


def dilatation(image: np.ndarray, simple_dilate: bool = True, shape_type: int = 0, dilation_size: int = 1) -> np.ndarray:
    img_cpy = image.copy()
    kernel, dilation_size = build_kernel(dilation_size)
    if simple_dilate:
        dilated_image = cv2.dilate(img_cpy, kernel=kernel, iterations=ITERATIONS)
    else:
        element = cv2.getStructuringElement(morph_shape(shape_type), (dilation_size, dilation_size))
        dilated_image = cv2.dilate(img_cpy, element, iterations=ITERATIONS)
    return dilated_image


if __name__ == "__main__":
    filename = "andenne_a47.png"
    filename_without_ext = filename.split(".")[0]
    test_file_path = os.path.join(os.getcwd(), "trainset_color_segmented_normalized_histflat_numclusters_2_all_images", filename)
    test_file = cv2.imread(test_file_path)
    kernel_size = 3     # 5
    real_kernel_size = compute_kernel_size(kernel_size)
    opened_file = opening(test_file, kernel_size)
    closed_file = closing(test_file, kernel_size)
    eroded_file = erosion(test_file, True, erosion_size=kernel_size)
    dilated_file = dilatation(test_file, True, dilation_size=kernel_size)

    cv2.imshow("original", test_file)
    cv2.imshow("opened", opened_file)
    cv2.imshow("closed", closed_file)
    cv2.imshow("eroded", eroded_file)
    cv2.imshow("dilated", dilated_file)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imwrite("{}.png".format(filename_without_ext), test_file)
    cv2.imwrite("{}_opened_{}.png".format(filename_without_ext,real_kernel_size), opened_file)
    # cv2.imwrite("{}_closed.png".format(filename_without_ext), closed_file)
    cv2.imwrite("{}_eroded_{}.png".format(filename_without_ext,real_kernel_size), eroded_file)
    # cv2.imwrite("{}_dilated.png".format(filename_without_ext), dilated_file)
