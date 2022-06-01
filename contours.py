from typing import List

import cv2
import numpy as np
import os

import fingerprint as fgp

# https://learnopencv.com/contour-detection-using-opencv-python-c/#Contour-Hierarchies


class ContoursData:
    image_name: str
    contours: List[np.ndarray]
    hierarchy: np.ndarray

    def __init__(self, image_name: str, contours: List[np.ndarray], hierarchy: np.ndarray):
        self.image_name = image_name
        self.contours = contours
        self.hierarchy = hierarchy


def create_histograms_from_contours(image: np.ndarray, image_filepath: str):
    """
    create histograms for the given image path
    :param image:
    :return:
    """
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # apply a threshold just to be sure
    ret, thresh = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    points = []
    # print("contours")
    for c in contours:
        for val in c:
            x, y = val[0]
            points.append([x,y])
    # print(len(points))
    # print("creating histogram")
    global_histogram = fgp.make_polar_histogram(np.array(points), image_filepath)
    return global_histogram, ContoursData(image_filepath, contours, hierarchy)


if __name__ == "__main__":
    file_test_path = os.path.join(os.getcwd(), "picture.png")
    hist, contourDT = create_histograms_from_contours(file_test_path)