import os
from os import path
import shutil

from PIL import Image, ExifTags
import cv2
import json
import numpy as np
import via_project
from tqdm import tqdm


def generate_masks(json_path: str, source_folder: str, output_folder: str):
    count = 0  # Count of total images saved

    # Read JSON file
    project = via_project.ViaProject(json_path)

    # We prepare the folders
    print("Creating folders...")
    for file_name_img in tqdm(os.listdir(source_folder)):
        if os.path.isdir(file_name_img):    # if it is a folder, we pass to the next
            continue
        to_save_folder = os.path.join(output_folder, file_name_img[:-4])
        image_folder = os.path.join(to_save_folder, "images")
        mask_folder = os.path.join(to_save_folder, "masks")
        curr_img = os.path.join(source_folder, file_name_img)

        if not os.path.exists(to_save_folder):
            os.mkdir(to_save_folder)
        if not os.path.exists(image_folder):
            os.mkdir(image_folder)
            shutil.copyfile(curr_img, os.path.join(image_folder, file_name_img))
        if not os.path.exists(mask_folder):
            os.mkdir(mask_folder)


    # For each view, generate mask and save in corresponding
    # folder

    print("Drawing masks...")
    for view in tqdm(project.views_list):
        file_img: via_project.ViaFile = project.find_file_by_id(view.files_associated[0])
        img_path = os.path.join(output_folder, file_img.file_name[:-4], "images", file_img.file_name)

        # We must deal with the annoying "Orientation" tag from exif metadata
        try:
            image = Image.open(img_path)

            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break

            exif = image._getexif()

            if exif[orientation] == 3:
                image = image.rotate(180, expand=True)
            elif exif[orientation] == 6:
                image = image.rotate(270, expand=True)
            elif exif[orientation] == 8:
                image = image.rotate(90, expand=True)

            image.save(img_path)
            image.close()
        except (AttributeError, KeyError, IndexError):
            # cases: image don't have getexif
            pass

        image = Image.open(img_path)
        width = image.width
        height = image.height

        to_save_folder = os.path.join(output_folder, file_img.file_name[:-4])
        mask_folder = os.path.join(to_save_folder, "masks")
        mask_path = os.path.join(mask_folder, file_img.file_name[:-4] + ".png")
        if os.path.exists(mask_path):   # We dont regenerate a mask that already exists
            continue
        # cannot use the same dimensions as the original image because python/cv2/pillow fuck up the image dimensions at random
        dimension = max(width, height)
        mask = np.zeros((dimension, dimension))
        try:
            arr = np.array(view.all_points)
        except:
            print("Not found:", view)
            continue
        count += 1
        # print("count: ", count)

        cv2.fillPoly(mask, [arr], color=(255))

        cv2.imwrite(mask_path, mask)
        mask_reread = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        cropped_mask = mask_reread[0:height, 0:width]
        cv2.imwrite(mask_path, cropped_mask)

    print("Images saved:", count)


def generate_masks_for_not_salam_pictures(source_folder: str, output_folder: str):
    for file_name_img in tqdm(os.listdir(source_folder)):
        if os.path.isdir(file_name_img):    # if it is a folder, we pass to the next
            continue
        to_save_folder = os.path.join(output_folder, file_name_img[:-4])
        image_folder = os.path.join(to_save_folder, "images")
        mask_folder = os.path.join(to_save_folder, "masks")
        curr_img = os.path.join(source_folder, file_name_img)

        if not os.path.exists(to_save_folder):
            os.mkdir(to_save_folder)
        if not os.path.exists(image_folder):
            os.mkdir(image_folder)
            shutil.copyfile(curr_img, os.path.join(image_folder, file_name_img))
        if not os.path.exists(mask_folder):
            os.mkdir(mask_folder)

        # We must deal with the annoying "Orientation" tag from exif metadata
        try:
            image = Image.open(os.path.join(image_folder, file_name_img))

            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break

            exif = image._getexif()

            if exif[orientation] == 3:
                image = image.rotate(180, expand=True)
            elif exif[orientation] == 6:
                image = image.rotate(270, expand=True)
            elif exif[orientation] == 8:
                image = image.rotate(90, expand=True)

            image.save(os.path.join(image_folder, file_name_img))
            image.close()
        except (AttributeError, KeyError, IndexError):
            # cases: image don't have getexif
            pass

        image = Image.open(os.path.join(image_folder, file_name_img))
        width = image.width
        height = image.height

        mask_path = os.path.join(mask_folder, file_name_img[:-4] + ".png")
        if os.path.exists(mask_path):
            continue
        mask = np.zeros((height, width))
        cv2.imwrite(mask_path, mask)




if __name__ == '__main__':
    print("Generating masks")
    generate_masks(
        json_path=os.path.join(os.getcwd(), "via_project_salamandres.json"),
        source_folder=os.path.join(os.getcwd(), "images"),
        output_folder=os.path.join(os.getcwd(), "images_ready")
    )
    print("Generating non-target masks")
    generate_masks_for_not_salam_pictures(
        source_folder=os.path.join(os.getcwd(), "images_not_salam"),
        output_folder=os.path.join(os.getcwd(), "images_ready")
    )