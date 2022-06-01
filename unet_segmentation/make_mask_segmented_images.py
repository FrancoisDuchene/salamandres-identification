import cv2
import os
from enum import Enum

import numpy as np
from tqdm import tqdm


class NormalizationType(Enum):
    NONE = 0
    MINMAX = 1
    HIST_FLATTENING = 2


NORMALIZATION_TYPE_SELECTED = NormalizationType.HIST_FLATTENING


def make_new_images(inputs: list, masks: list, destination_dir: str):
    """
    takes images paths and their masks paths, and write into destination_dir the resulting
    segmented image
    :param inputs:
    :param masks:
    :param destination_dir:
    """
    for i in tqdm(range(0, len(inputs))):
        input_path = inputs[i]
        input_filename = input_path.split("\\")[-1]
        mask_path = masks[i]
        mask_filename = mask_path.split("\\")[-1]

        img_input: np.ndarray = cv2.imread(input_path)
        img_mask: np.ndarray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        img_final = img_input

        if NORMALIZATION_TYPE_SELECTED == NormalizationType.MINMAX:     # DOES NOT WORK
            height = img_input.shape[0]
            width = img_input.shape[1]
            norm_img = np.zeros((height, width))
            img_final = cv2.normalize(img_input, norm_img, 0, 255, cv2.NORM_MINMAX)

        # https://linuxtut.com/en/9c9fc6c0e9e8a9d05800/
        elif NORMALIZATION_TYPE_SELECTED == NormalizationType.HIST_FLATTENING:
            hsv = cv2.cvtColor(img_input, cv2.COLOR_BGR2HSV)    # convert to hsv color system
            h, s, v = cv2.split(hsv)  # divided into each component
            result = cv2.equalizeHist(v)
            hsv = cv2.merge((h, s, result))
            img_final = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        img_segmented = cv2.bitwise_and(img_final, img_final, mask=img_mask)
        cv2.imwrite(os.path.join(destination_dir, mask_filename), img_segmented)


if __name__ == "__main__":
    prefix_not_target = "not_salam_"
    source_dir = os.path.join(os.getcwd(), "images_ready")
    target_dir = os.path.join(os.getcwd(), "segmented_images_trainset")
    input_img_paths = []
    target_img_paths = []
    print("Loading Dataset")
    for folder in tqdm(os.listdir(source_dir)):
        if os.path.isdir(os.path.join(source_dir, folder)):
            sample_path = os.path.join(source_dir, folder)
            contents_dir_images = sorted(os.listdir(os.path.join(sample_path, "images")))
            contents_dir_masks = sorted(os.listdir(os.path.join(sample_path, "masks")))
            if folder.startswith("notsalam_"):
                # print("skipping ", os.listdir(os.path.join(sample_path, "images"))[0])
                continue

            img_path: str = os.path.join(sample_path,
                                         "images",
                                         os.listdir(os.path.join(sample_path, "images"))[0]
                                         )
            mask_path: str = os.path.join(sample_path,
                                          "masks",
                                          os.listdir(os.path.join(sample_path, "masks"))[0]
                                          )
            # Both img and mask are at the same index in both lists
            input_img_paths.append(img_path)
            target_img_paths.append(mask_path)
    sorted(input_img_paths)
    sorted(target_img_paths)

    print("Number of samples:", len(input_img_paths))

    make_new_images(input_img_paths, target_img_paths, target_dir)
