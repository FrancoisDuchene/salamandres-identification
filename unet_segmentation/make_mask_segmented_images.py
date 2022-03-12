import cv2
import os
from tqdm import tqdm


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
        # input_filename = inputs[i].split("\\")[-1]
        mask_path = masks[i]
        mask_filename = masks[i].split("\\")[-1]

        img_input = cv2.imread(input_path)
        img_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        img_segmented = cv2.bitwise_and(img_input, img_input, mask=img_mask)
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
            if len(os.listdir(os.path.join(sample_path, "images"))) == 1:
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
