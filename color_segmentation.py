#! /usr/bin/python3
import random

import numpy as np
from sklearn.cluster import KMeans
import argparse
import cv2
import os.path
import copy
import matplotlib.pyplot as plt

# code from https://nrsyed.com/2018/03/29/image-segmentation-via-k-means-clustering-with-opencv-python/
from typing import Dict


def app_get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
                    help="Path to the image file")
    ap.add_argument("-w", "--width", type=int, default=0,
                    help="Width to resize image to in pixels")
    ap.add_argument("-s", "--color-space", type=str, default="bgr",
                    help="Color space to use: BGR (default), HSV, Lav, YCrCb (YCC)")
    ap.add_argument('-c', '--channels', type=str, default='all',
                    help='Channel indices to use for clustering, where 0 is the first channel,'
                         ' 1 is the second channel, etc. E.g., if BGR color space is used, "02" '
                         'selects channels B and R. (default "all")')
    ap.add_argument('-n', '--num-clusters', type=int, default=3,
                    help='Number of clusters for K-means clustering (default 3, min 2).')
    ap.add_argument('-o', '--output-file', action='store_true',
                    help='Save output image (side-by-side comparison of original image and clustering result) to disk.')
    ap.add_argument('-f', '--output-format', type=str, default='png',
                    help='File extension for output image (default png)')
    ap.add_argument("-t", "--output-color", type=str, default='gray',
                    help="Output image color type: gray (default), rgb")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Show debbuging info")
    ap.add_argument("--show-results", type=str, default="false",
                    help="Show resulting images in a window: false (default), gui")
    args = vars(ap.parse_args())
    return args


def process_args(args: dict):
    """
    Take the argument list and process them in order to return a dictionary containing the altered image and
    the kmeans clustering result
    :param args: a dictionary, containing at least a key "image" with the path to an image file
    :return: a dictionary with the following keys:
        "filename": (str) the filename of the original image.
        "image": (cv2 image) the converted image in the given color space.
        "orig": (cv2 image) the original image (may have been resized). Same dimensions as the resulting images.
        "width": (int) the width of image, orig and kmeans_image.
        "height": (int) the height of image, orig and kmeans_image.
        "color_space": (str) the color_space used to produce the kmeans clustering.
        "channels": (str) the channels used to do the kmeans clustering.
        "num_clusters": (int) the number of clusters used to do the kmeans clustering (the K parameter).
        "output_file": (bool) if a output file must be produced
        "output_format": (str) the file format which will be used if a output file is produced.
        "output_color_type": (str) used to specify wether the resulting Kmeans output image will be in grayscale or in RGB.
    """
    app_memory = {"filename": str, "image": None, "orig": None, "width": int, "height": int, "color_space": str,
                  "channels": str, "num_clusters": int, "output_file": bool, "output_format": str, "kmeans_res": None,
                  "clustering": None, "labels": list, "adjusted_labels": list, "output_color_type": str, "kmeans_image": None,
                  "show_results": str, "verbose": bool}
    image = cv2.imread(args['image'])
    app_memory["filename"] = copy.copy(args['image'])
    app_memory["image"] = image
    # Resize image and make a copy of the original (resized) image.

    if args['width'] > 0:
        height = int((args['width'] / image.shape[1]) * image.shape[0])
        image = cv2.resize(image, (args["width"], height),
                           interpolation=cv2.INTER_AREA)
        app_memory["width"] = copy.copy(args["width"])
        app_memory["height"] = height
    else:
        app_memory["width"] = copy.copy(image.shape[1])
        app_memory["height"] = copy.copy(image.shape[0])
    orig = image.copy()
    app_memory["orig"] = orig

    # Change image color space, if necessary.
    color_space = args['color_space'].lower()
    if color_space == "hsv":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif color_space == "ycrcb" or color_space == "ycc":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    elif color_space == "lab":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    else:
        color_space = "bgr"  # set for file naming purpose

    app_memory["image"] = image
    app_memory["color_space"] = color_space

    # Keep only the selected channels for K-means clustering.
    if args["channels"] != "all":
        channels = cv2.split(image)
        channelIndices = []
        for char in args["channels"]:
            channelIndices.append(int(char))
        image = image[:, :, channelIndices]
        if len(image.shape) == 2:
            image.reshape(image.shape[0], image.shape[1], 1)
        app_memory["image"] = image
    app_memory["channels"] = copy.copy(args["channels"])

    if args['num_clusters'] < 2:
        if args["verbose"]:
            print('Warning: num-clusters < 2 invalid. Using num-clusters = 2')
    app_memory["num_clusters"] = max(2, args['num_clusters'])

    app_memory["output_color_type"] = copy.copy(args["output_color"].lower())
    app_memory["output_file"] = copy.copy(args["output_file"])
    app_memory["output_format"] = copy.copy(args['output_format'])
    if args["show_results"] != "gui" or args["show_results"] != "false":
        if args["verbose"]:
            print("Warning: show-results value not valid. Not showing results option used per default")
        app_memory["show_results"] = "false"
    else:
        app_memory["show_results"] = args["show_results"]
    app_memory["verbose"] = args["verbose"]
    return app_memory


def kmeans_clustering(app_memory: dict):
    """
    Execute the KMeans method from sklearn with the given parameters,
    modify app_memory by adding results of the kmeans method
    :param app_memory: a dictionary containing the following keys:
        "filename": (str) the filename of the original image.
        "image": (cv2 image) the converted image in the given color space.
        "orig": (cv2 image) the original image (may have been resized). Same dimensions as the resulting images.
        "width": (int) the width of image, orig and kmeans_image.
        "height": (int) the height of image, orig and kmeans_image.
        "color_space": (str) the color_space used to produce the kmeans clustering.
        "channels": (str) the channels used to do the kmeans clustering.
        "num_clusters": (int) the number of clusters used to do the kmeans clustering (the K parameter).
        "output_file": (bool) if a output file must be produced
        "output_format": (str) the file format which will be used if a output file is produced.
        "output_color_type": (str) used to specify wether the resulting Kmeans output image will be in grayscale or in RGB.
    :return: Nothing. Modify the parameter app_memory by adding the following keys:
        "kmeans_res": (KMeans result) the raw result produced by the KMeans method.
        "clustering": (np 2D array) 2D array where each element represent a cluster result.
        "labels": (list(int)) A list of integers representing the clusters produced by the Kmeans clustering method.
    """
    image = app_memory["image"]
    reshaped = image.reshape(image.shape[0] * image.shape[1], image.shape[2])

    num_clusters = app_memory["num_clusters"]
    kmeans = KMeans(n_clusters=num_clusters, n_init=40, max_iter=500).fit(reshaped)

    app_memory["kmeans_res"] = kmeans

    # Reshape result back into a 2D array, where each element represents the
    # corresponding pixel's cluster index (0 to K - 1).
    clustering = np.reshape(np.array(kmeans.labels_, dtype=np.uint8),
                            (image.shape[0], image.shape[1]))

    app_memory["clustering"] = clustering

    # Sort the cluster labels in order of the frequency with which they occur.
    sorted_labels = sorted([n for n in range(num_clusters)],
                          key=lambda x: -np.sum(clustering == x))

    app_memory["labels"] = sorted_labels


def create_kmeans_image(app_memory: dict):
    """
    :param app_memory: a dictionary with the following keys:
        "filename": (str) the filename of the original image.
        "image": (cv2 image) the converted image in the given color space.
        "orig": (cv2 image) the original image (may have been resized). Same dimensions as the resulting images.
        "width": (int) the width of image, orig and kmeans_image.
        "height": (int) the height of image, orig and kmeans_image.
        "color_space": (str) the color_space used to produce the kmeans clustering.
        "channels": (str) the channels used to do the kmeans clustering.
        "num_clusters": (int) the number of clusters used to do the kmeans clustering (the K parameter).
        "output_file": (bool) if a output file must be produced
        "output_format": (str) the file format which will be used if a output file is produced.
        "output_color_type": (str) used to specify wether the resulting Kmeans output image will be in grayscale or in RGB.
        "kmeans_res": (KMeans result) the raw result produced by the KMeans method.
        "clustering": (np 2D array) 2D array where each element represent a cluster result.
        "labels": (list(int)) A list of integers representing the clusters produced by the Kmeans clustering method.
    :return: Nothing. Modify the parameter app_memory by adding the following key:
        "kmeans_image": the image converted into the given color space and then with its colors reorganized into clusters. They are interpreted wether in grayscale or in RGB.
    """
    image = app_memory["image"]
    sorted_labels = app_memory["labels"]
    num_clusters = app_memory["num_clusters"]
    output_color_type = app_memory["output_color_type"]
    kmeans_image_shape : tuple
    if output_color_type == "rgb":
        kmeans_image_shape = image.shape[:]
    else:   # use grayscale
        kmeans_image_shape = image.shape[:2]

    def _init_image(kmeans_image_shape: tuple):
        # Initialize K-means image; set pixel colors based on clustering.
        # If the color type is not set on grayscale, one channel at random will get a cluster value,
        # thus varying between red, green and blue
        gray = True if len(kmeans_image_shape) == 2 else False
        kmeansImage = np.zeros(kmeans_image_shape, dtype=np.uint8)
        app_memory["adjusted_labels"] = []
        for i, label in enumerate(sorted_labels):
            a = int((255) / (num_clusters - 1)) * i
            if gray:
                kmeansImage[app_memory["clustering"] == label] = a
                app_memory["adjusted_labels"].append(a)
            else:
                r = random.randint(0, 2)
                b = [0, 0, 0]
                b[r] = a
                kmeansImage[app_memory["clustering"] == label] = b
                app_memory["adjusted_labels"].append(b)

        app_memory["kmeans_image"] = kmeansImage
    _init_image(kmeans_image_shape)


def show_spots_only_image(app_memory: dict):
    adj_labs = app_memory["adjusted_labels"]
    bin_img = app_memory["kmeans_image"].copy()
    lab_chosen = 4
    mycat = adj_labs[lab_chosen]
    bin_img[np.where(bin_img!=mycat)] = 254
    bin_img[np.where(bin_img==mycat)] = 255
    bin_img[np.where(bin_img==254)] = 0

    # plt.hist(bin_img)
    # plt.plot()
    # plt.show()

    # cv2.imshow("binary", bin_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    output_folder = "output_modified_images"
    cv2.imwrite(os.path.join(output_folder, "test_binaire_"+ str(lab_chosen) + ".jpg"), bin_img)


def output_results(app_memory: dict):
    # Concatenate original image and K-means image, separated by a gray strip.
    orig = app_memory["orig"]
    kmeans_image = app_memory["kmeans_image"]
    concatImage = np.concatenate((orig,
                                  193 * np.ones((orig.shape[0], int(0.0625 * orig.shape[1]), 3), dtype=np.uint8),
                                  cv2.cvtColor(kmeans_image, cv2.COLOR_RGB2BGR)), axis=1)

    if app_memory['output_file']:
        # Construct output filename and write image to disk.
        file_extension = app_memory['output_format']
        filename = (app_memory["filename"].split(os.sep)[-1].split(".")[0] + "__" + "h" + str(app_memory["height"]) + "w" + str(app_memory["width"]) + "_" +
                    app_memory["color_space"] + '_c' + app_memory['channels']
                    + 'n' + str(app_memory["num_clusters"]) + "_" + app_memory["output_color_type"] + '.' + file_extension)
        output_folder = "output_modified_images"
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        final_filename = os.path.join(output_folder, filename)
        cv2.imwrite(final_filename, concatImage)
        if app_memory["verbose"]:
            print(filename, output_folder)
    if app_memory["show_results"] == "gui":
        if app_memory["verbose"]:
            print("showing results")
        cv2.imshow('Original vs clustered', concatImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return app_memory


def main(**kwargs):
    if kwargs["verbose"]:
        print("Color segmentation script - v1.0")
        print(kwargs)
    app_memory = process_args(kwargs)
    kmeans_clustering(app_memory)
    create_kmeans_image(app_memory)
    output_results(app_memory)
    show_spots_only_image(app_memory)

    return app_memory


if __name__ == '__main__':
    args = app_get_arguments()
    app_memory = main(**args)
