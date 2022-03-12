import cv2
import numpy as np
import os


def unet_mask_to_production_mask(source_path: str, mask_path: str, output_path: str):
    """
    Transform a crude mask given by unet into a mask that can be used in production
    It resize the mask to the source dimensions,
    clear the mask boundaries to eliminate the blurry borders,
    crop the image into the mask, and remove the "black" part of the mask to make it transparent
    :param source_path: source path to the original file
    :param mask_path: source path of the mask given by unet
    :return: the resulting masked image (with the same dimensions as image from source_path)
    """
    img_source: np.ndarray = cv2.imread(source_path)
    img_mask: np.ndarray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    width, height, _ = img_source.shape

    img_mask = cv2.resize(img_mask, (height, width), cv2.INTER_CUBIC)

    threshold = 100
    result, img_mask = cv2.threshold(img_mask, threshold, 255, cv2.THRESH_BINARY)
    fileextension = output_path[-4:]
    first_mask_path = output_path[:-4] + "_mask" + fileextension
    cv2.imwrite(first_mask_path, img_mask)
    img_mask = cv2.bitwise_and(img_source, img_source, mask=img_mask)
    cv2.imwrite(output_path, img_mask)

    return img_mask


if __name__ == '__main__':
    print("Transforming masks")
    source_path = os.path.join(
        os.getcwd(),
        "images_ready",
        "00000179",
        "images",
        "00000179.jpg"
    )
    mask_path = os.path.join(
        os.getcwd(),
        "output_pred_images",
        "00000179_pred.png"
    )
    output_path = os.path.join(
        os.getcwd(),
        "output_pred_images",
        "masked_images",
        "00000179_masked.png"
    )
    unet_mask_to_production_mask(source_path, mask_path, output_path)
