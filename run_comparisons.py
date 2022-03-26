from typing import List

import connected_components as cc
import fingerprint as fgp

import os
from tqdm import tqdm


def make_histograms(images_paths) -> List[fgp.PolarHistogram]:
    print("Retrieving CC info...")
    cc_data_set = cc.analyse_cc(images_paths)
    print("Computing histograms...")
    histograms: list = []
    for img_cc_data in tqdm(cc_data_set.cc_data):
        centroids = img_cc_data.centroids
        histogram = fgp.make_polar_histogram(centroids, name=img_cc_data.image_name)
        histograms.append(histogram)

    return histograms


def get_image_paths() -> list:
    source_dir = os.path.join(os.getcwd(), "trainset_color_segmented")
    images_paths = []
    for file_path in tqdm(os.listdir(source_dir)):
        if not os.path.isdir(file_path):
            images_paths.append(os.path.join(source_dir, file_path))
    return images_paths


def run():
    print("Retrieving image paths...")
    image_paths = get_image_paths()
    histograms = make_histograms(image_paths)
    return histograms


if __name__ == "__main__":
    hists = run()
    hists[0].show_img_radius(img_width=1024, img_height=1365)
    print("Done !")
