import os
import random
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
        file_extension = file_img.file_name[-4:]
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
        except (TypeError):
            print(img_path)
            exit(0)

        image = Image.open(img_path)
        width = image.width
        height = image.height

        to_save_folder = os.path.join(output_folder, file_img.file_name[:-4])
        mask_folder = os.path.join(to_save_folder, "masks")
        mask_path = os.path.join(mask_folder, file_img.file_name[:-4])
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

        # Saving all masks
        mask = mask[0:height, 0:width]
        cv2.imwrite(mask_path + ".png", mask)

        original_img = cv2.imread(img_path)

        def make_rotations():
            mask_90 = cv2.rotate(mask, cv2.cv2.ROTATE_90_CLOCKWISE)
            mask_270 = cv2.rotate(mask, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
            mask_180 = cv2.rotate(mask, cv2.cv2.ROTATE_180)
            cv2.imwrite(mask_path + "__090.png", mask_90)
            cv2.imwrite(mask_path + "__270.png", mask_270)
            cv2.imwrite(mask_path + "__180.png", mask_180)
            # Saving images rotated
            img_90 = cv2.rotate(original_img, cv2.cv2.ROTATE_90_CLOCKWISE)
            img_270 = cv2.rotate(original_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            img_180 = cv2.rotate(original_img, cv2.ROTATE_180)
            cv2.imwrite(img_path[:-4] + "__090" + file_extension, img_90)
            cv2.imwrite(img_path[:-4] + "__270" + file_extension, img_270)
            cv2.imwrite(img_path[:-4] + "__180" + file_extension, img_180)

        def make_zoom():
            def zoom(img, value, dimensions: int, random_state):
                if value > 1 or value < 0:
                    print("Value for zoom should be less than 1 and greater than 0\ninstead got {}".format(value))
                    return img
                random.setstate(random_state)
                value = random.uniform(value, 1)
                h, w = img.shape[:2]
                h_taken = int(value*h)
                w_taken = int(value*w)
                random.setstate(random_state)
                h_start = random.randint(0, h-h_taken)
                random.setstate(random_state)
                w_start = random.randint(0, w-w_taken)
                if dimensions == 2:
                    img = img[h_start:h + h_taken, w_start:w_start + w_taken]
                else:
                    img = img[h_start:h + h_taken, w_start:w_start + w_taken, :]
                img = cv2.resize(img, (h, w), cv2.INTER_CUBIC)
                return img
            zoom_factor = 0.5
            rdmstate = random.getstate()
            mask_zoom = zoom(mask, zoom_factor, 2, rdmstate)
            cv2.imwrite(mask_path + "__zoom.png", mask_zoom)
            img_zoom = zoom(original_img, zoom_factor, 3, rdmstate)
            cv2.imwrite(img_path[:-4] + "__zoom" + file_extension, img_zoom)

        def make_flip():
            flip_horizontal = cv2.flip(original_img, 0)
            flip_vertical = cv2.flip(original_img, 1)
            flip_both = cv2.flip(original_img, -1)
            cv2.imwrite(img_path[:-4] + "__flipX" + file_extension, flip_horizontal)
            cv2.imwrite(img_path[:-4] + "__flipY" + file_extension, flip_vertical)
            cv2.imwrite(img_path[:-4] + "__flipXY" + file_extension, flip_both)
            mask_flip_horizontal = cv2.flip(mask, 0)
            mask_flip_vertical = cv2.flip(mask, 1)
            mask_flip_both = cv2.flip(mask, -1)
            cv2.imwrite(mask_path + "__flipX.png", mask_flip_horizontal)
            cv2.imwrite(mask_path + "__flipY.png", mask_flip_vertical)
            cv2.imwrite(mask_path + "__flipXY.png", mask_flip_both)

        make_zoom()
        make_rotations()
        make_flip()

    print("Images saved:", count)


def generate_masks_for_not_salam_pictures(source_folder: str, output_folder: str):
    prefix = "not_salam_"
    for file_name_img in tqdm(os.listdir(source_folder)):
        if os.path.isdir(file_name_img):    # if it is a folder, we pass to the next
            continue
        new_file_name_img = prefix + file_name_img
        to_save_folder = os.path.join(output_folder, file_name_img[:-4])
        image_folder = os.path.join(to_save_folder, "images")
        mask_folder = os.path.join(to_save_folder, "masks")
        curr_img = os.path.join(source_folder, file_name_img)

        if not os.path.exists(to_save_folder):
            os.mkdir(to_save_folder)
        if not os.path.exists(image_folder):
            os.mkdir(image_folder)
            shutil.copyfile(curr_img, os.path.join(image_folder, new_file_name_img))
        if not os.path.exists(mask_folder):
            os.mkdir(mask_folder)

        # We must deal with the annoying "Orientation" tag from exif metadata
        try:
            image = Image.open(os.path.join(image_folder, new_file_name_img))

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

            image.save(os.path.join(image_folder, new_file_name_img))
            image.close()
        except (AttributeError, KeyError, IndexError):
            # cases: image don't have getexif
            pass

        image = Image.open(os.path.join(image_folder, new_file_name_img))
        width = image.width
        height = image.height

        mask_path = os.path.join(mask_folder, new_file_name_img[:-4] + ".png")
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