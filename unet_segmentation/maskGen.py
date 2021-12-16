import os
from os import path

import cv2
import json
import numpy as np
import via_project


def generate_masks_v3(json_path: str, source_folder: str):
    count = 0  # Count of total images saved

    # Read JSON file
    project = via_project.ViaProject(json_path)

    views_list = project.views_list

    # We prepare the folders
    for file_name_img in os.listdir(source_folder):
        if os.path.isdir(file_name_img):    # if it is a folder, we pass to the next
            continue
        to_save_folder = os.path.join(source_folder, file_name_img[:-4])
        image_folder = os.path.join(to_save_folder, "images")
        mask_folder = os.path.join(to_save_folder, "masks")
        curr_img = os.path.join(source_folder, file_name_img)

        if not os.path.exists(to_save_folder):
            os.mkdir(to_save_folder)
        if not os.path.exists(image_folder):
            os.mkdir(image_folder)
            os.rename(curr_img, os.path.join(image_folder, file_name_img))
        if not os.path.exists(mask_folder):
            os.mkdir(mask_folder)


    # For each view, generate mask and save in corresponding
    # folder
    i5, i55 = None, None
    for view in project.views_list:
        file_img: via_project.ViaFile = project.find_file_by_id(view.files_associated[0])
        img = cv2.imread(
            os.path.join("images", file_img.file_name[:-4], "images", file_img.file_name),
            cv2.IMREAD_UNCHANGED
        )
        height, width, _ = img.shape

        to_save_folder = os.path.join(source_folder, file_img.file_name[:-4])
        mask_folder = os.path.join(to_save_folder, "masks")
        # cannot use the same dimensions as the original image because python/cv2/pillow fuck up the image dimensions at random
        dimension = max(width, height)
        mask = np.zeros((dimension, dimension), dtype="uint8")
        try:
            arr = np.array(view.all_points)
        except:
            print("Not found:", view)
            continue
        count += 1
        print("count: ", count)

        cv2.fillPoly(mask, [arr], color=(255))
        cv2.imwrite(os.path.join(mask_folder, file_img.file_name[:-4] + ".png"), mask)

    print("Images saved:", count)


def generate_masks_v2():
    source_folder = os.path.join(os.getcwd(), "images")
    json_path = "via_project_salamandres.json"  # Relative to root directory
    count = 0  # Count of total images saved
    file_bbs = {}  # Dictionary containing polygon coordinates for mask
    MASK_WIDTH = 512  # Dimensions should match those of ground truth image
    MASK_HEIGHT = 512

    # Read JSON file
    with open(json_path) as f:
        data = json.load(f)

    # Extract X and Y coordinates if available and update dictionary
    def add_to_dict(data, itr, key, count):
        try:
            x_points = data[itr]["regions"][count]["shape_attributes"]["all_points_x"]
            y_points = data[itr]["regions"][count]["shape_attributes"]["all_points_y"]
        except:
            print("No BB. Skipping", key)
            return

        all_points = []
        for i, x in enumerate(x_points):
            all_points.append([x, y_points[i]])

        file_bbs[key] = all_points

    for itr in data:
        file_name_json = data[itr]["filename"]
        sub_count = 0  # Contains count of masks for a single ground truth image

        if len(data[itr]["regions"]) > 1:
            for _ in range(len(data[itr]["regions"])):
                key = file_name_json[:-4] + "*" + str(sub_count + 1)
                add_to_dict(data, itr, key, sub_count)
                sub_count += 1
        else:
            add_to_dict(data, itr, file_name_json[:-4], 0)

    print("\nDict size: ", len(file_bbs))

    for file_name in os.listdir(source_folder):
        to_save_folder = os.path.join(source_folder, file_name[:-4])
        image_folder = os.path.join(to_save_folder, "images")
        mask_folder = os.path.join(to_save_folder, "masks")
        curr_img = os.path.join(source_folder, file_name)

        # make folders and copy image to new location
        os.mkdir(to_save_folder)
        os.mkdir(image_folder)
        os.mkdir(mask_folder)
        os.rename(curr_img, os.path.join(image_folder, file_name))

    # For each entry in dictionary, generate mask and save in correponding
    # folder
    for itr in file_bbs:
        num_masks = itr.split("*")
        to_save_folder = os.path.join(source_folder, num_masks[0])
        mask_folder = os.path.join(to_save_folder, "masks")
        mask = np.zeros((MASK_WIDTH, MASK_HEIGHT))
        try:
            arr = np.array(file_bbs[itr])
        except:
            print("Not found:", itr)
            continue
        count += 1
        cv2.fillPoly(mask, [arr], color=(255))

        if len(num_masks) > 1:
            cv2.imwrite(os.path.join(mask_folder, itr.replace("*", "_") + ".png"), mask)
        else:
            cv2.imwrite(os.path.join(mask_folder, itr + ".png"), mask)

    print("Images saved:", count)


if __name__ == '__main__':
    generate_masks_v3(
        os.path.join(os.getcwd(), "via_project_salamandres.json"),
        os.path.join(os.getcwd(), "images")
    )
