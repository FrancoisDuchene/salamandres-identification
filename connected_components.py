import json
from json import JSONEncoder
from typing import List, Any

import cv2
import os
import matplotlib.pyplot as plt

import scipy.stats as spstats
import numpy as np
from tqdm import tqdm

import dilatation_erosion

CC_CONNECTIVITY = 8
DILATATION_EROSION_KERNEL_SIZE = 1


class ConnectedComponentsData:
    image_name: str
    num_labels: int
    # labels: np.ndarray    # si on met ce champs lors du dump json, il n'y a plus d'espace mémoire
    stats: np.ndarray
    centroids: np.ndarray
    areas: np.ndarray
    area_avg: float
    area_std: float
    area_median: float

    def __init__(self, image: np.ndarray, image_name: str):
        output = cv2.connectedComponentsWithStats(image, connectivity=CC_CONNECTIVITY, ltype=cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output
        self.num_labels = numLabels
        # self.labels = labels
        self.stats = stats
        self.centroids = centroids
        self.image_name = image_name
        self.areas = self.get_areas()
        self.area_avg = self.get_area_avg()
        self.area_std = self.get_area_std()
        self.area_median = self.get_area_median()

    def get_areas(self) -> np.ndarray:
        areas = []
        for k in range(1, self.num_labels):
            areas.append(self.stats[k, cv2.CC_STAT_AREA])
        area_np = np.array(areas)
        return area_np

    def get_area_avg(self):
        area_np = self.get_areas()
        if len(area_np) == 0:
            return 0
        mean = np.nanmean(area_np)
        if np.isnan(mean):
            return 0
        else:
            return mean

    def get_area_std(self):
        area_np = self.get_areas()
        return area_np.std()

    def get_area_median(self):
        area_np = self.get_areas()
        median = np.median(area_np)
        if isinstance(median, np.ndarray):
            return median[0]
        return median

    def background_stats(self):
        return self.stats[0]

    def component_stats(self, k):
        x = self.stats[k+1, cv2.CC_STAT_LEFT]
        y = self.stats[k+1, cv2.CC_STAT_TOP]
        w = self.stats[k+1, cv2.CC_STAT_WIDTH]
        h = self.stats[k+1, cv2.CC_STAT_HEIGHT]
        area = self.stats[k+1, cv2.CC_STAT_AREA]
        centroids: np.ndarray = self.centroids[k+1]
        return {"x": x, "y": y, "width": w, "height": h, "area": area, "centroids": centroids}

    def nb_labels(self):
        """
        :return: the number of labels (excluding the background)
        """
        return self.num_labels-1

    def __str__(self):
        return "CC for {} (nb_labels: {}, avg_area: {}, std_area: {}, med_area: {})"\
            .format(self.image_name, self.num_labels, self.area_avg, self.area_std, self.area_median)


class ConnectedComponentsEncoder(JSONEncoder):
    # https://stackoverflow.com/questions/36435039/failing-to-convert-numpy-array-to-json
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            # if obj.flags['C_CONTIGUOUS']:
            #     obj_data = obj.data
            # else:
            #     cont_obj = np.ascontiguousarray(obj)
            #     assert (cont_obj.flags['C_CONTIGUOUS'])
            #     obj_data = cont_obj.data
            ## data_b64 = base64.b64encode(obj_data)
            ## converting to base64 and returning a dictionary did not work
            ## return dict(__ndarray__ = data_b64, dtype = str(obj.dtype), shape = obj.shape)
            return obj.tolist()  ## instead, utilize numpy builtin tolist() method
        try:
            my_dict = obj.__dict__  ## <-- ERROR raised here
        except TypeError:
            print(TypeError)
        else:
            return my_dict
        return json.JSONEncoder.default(self, obj)


class CCDataSet:
    cc_data: List[ConnectedComponentsData]
    nb_cc_data: int
    frequency: dict

    def __init__(self):
        self.cc_data = []
        self.nb_cc_data = 0
        self.frequency = {"unique": [], "count": []}

    def add(self, data: ConnectedComponentsData):
        self.cc_data.append(data)
        self.nb_cc_data += 1

    def compute_frequency(self):
        areas = []
        for ccd in self.cc_data:
            for i in range(1,ccd.num_labels):   # 1 because we skip the background
                areas.append(ccd.stats[i, cv2.CC_STAT_AREA])
        areas_np = np.array(areas)
        unique, frequency = np.unique(areas_np, return_counts=True)
        self.frequency["unique"] = unique
        self.frequency["count"] = frequency

        return unique, frequency, areas_np


    def to_json(self) -> json:
        return json.dumps(self, indent=4, cls=ConnectedComponentsEncoder)

    def __len__(self):
        return self.nb_cc_data


def get_biggest_cc(components_stats: np.ndarray, nbCC: int):
    biggest_area: int = 0
    biggest_area_index = -1
    for j in range(1, nbCC):    # we start at 1 to avoid counting the background (always on index 0)
        i_area = components_stats[j, cv2.CC_STAT_AREA]
        if i_area > biggest_area:
            biggest_area = i_area
            biggest_area_index = j
    return biggest_area_index, biggest_area


def loop_over_components(components_stats: np.ndarray, nbCC: int, labs: np.ndarray, centro: np.ndarray, image, masked_path):
    # loop over the number of unique connected component labels
    new_img = image.copy()
    for i in range(0, nbCC):
        # if this is the first component then we examine the
        # *background* (typically we would just ignore this
        # component in our loop)
        if i == 0:
            text = "examining component {}/{} (background)".format(
                i + 1, nbCC)
        # otherwise, we are examining an actual connected component
        else:
            text = "examining component {}/{}".format(i + 1, nbCC)
        # print a status message update for the current connected
        # component
        print("[INFO] {}".format(text))
        # extract the connected component statistics and centroid for
        # the current label
        x = components_stats[i, cv2.CC_STAT_LEFT]
        y = components_stats[i, cv2.CC_STAT_TOP]
        w = components_stats[i, cv2.CC_STAT_WIDTH]
        h = components_stats[i, cv2.CC_STAT_HEIGHT]
        area = components_stats[i, cv2.CC_STAT_AREA]
        (cX, cY) = centro[i]
        # clone our original image (so we can draw on it) and then draw
        # a bounding box surrounding the connected component along with
        # a circle corresponding to the centroid
        cv2.rectangle(new_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.circle(new_img, (int(cX), int(cY)), 4, (0, 0, 255), -1)
        # construct a mask for the current connected component by
        # finding a pixels in the labels array that have the current
        # connected component ID
        componentMask = (labs == i).astype("uint8") * 255
        # show our output image and connected component mask
        output_window_name = "Output"
        cc_window_name = "Connected Component"
        cv2.namedWindow(output_window_name, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(cc_window_name, cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(output_window_name, 300, 300)
        cv2.moveWindow(cc_window_name, 600, 300)
        cv2.imshow(output_window_name, new_img)
        cv2.imshow(cc_window_name, componentMask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    cv2.imwrite(os.path.join(masked_path, "img_test_3_cc.png"), new_img)


def _test():
    # test_file_path = os.path.join(os.getcwd(), "unet_segmentation", "output_pred_images", "masked_images",
    #                               "vidhya_jac_400spe_1ADF1051-3926-4B00-B818-A818B32A08AD (1)_mask.png")
    # masked_path = os.path.join(os.getcwd(), "unet_segmentation", "output_pred_images", "masked_images")
    # test_file_path = os.path.join(masked_path, "test_mask.png")
    masked_path = os.path.join(os.getcwd(), "test_identification")
    test_file_path = os.path.join(masked_path, "img_test_3.png")

    image = cv2.imread(test_file_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[
        1]  # otsu thresholding https://en.wikipedia.org/wiki/Otsu%27s_method
    output = cv2.connectedComponentsWithStats(gray, connectivity=4, ltype=cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output

    # bai, ba = get_biggest_cc(stats, numLabels)
    # componentMask = (labels == bai).astype("uint8") * 255
    # cv2.imshow("Biggest component", componentMask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    loop_over_components(stats, numLabels, labels, centroids, image, masked_path)
    return output


def draw_info_cc(image_bgr: np.ndarray, path, cc_n, x, y, w, h, cx, cy):
    img_cpy = image_bgr.copy()
    cv2.rectangle(img_cpy, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.circle(img_cpy, (int(cx), int(cy)), 4, (0, 0, 255), -1)
    # cv2.imshow("cc_window_name", image_bgr)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite(path[:-4] + "_cc{}_.png".format(cc_n), img_cpy)
    return img_cpy


def analyse_cc(image_paths: list, use_erosion_dilatation_opening: bool = False) -> CCDataSet:
    """
    Analyze the connected components for all the images contained in image_paths
    :param image_paths: a list containing paths to images that needs to be analyzed
    :return: a CCDataset object containing the cc infos for all of the images
    """
    data_set = CCDataSet()
    must_draw = False
    for path in tqdm(image_paths):
        image = cv2.imread(path)
        data = analyse_one_cc(image, path, must_draw=must_draw, use_erosion_dilatation_opening=use_erosion_dilatation_opening)
        data_set.add(data)

    return data_set


def analyse_one_cc(image: np.ndarray, path: str, must_draw: bool = False, use_erosion_dilatation_opening: bool = False) -> ConnectedComponentsData:
    """
    Analyze the connected components for one image
    :param image: a cv2 loaded image
    :param path: the path of the image
    :param must_draw: a boolean, if set to true, the connected components will be drawn one photo at the time and saved
    in the same folder as path
    :return: a connectedComponentData object describing the connected components
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if use_erosion_dilatation_opening:
        dilated: np.ndarray = dilatation_erosion.opening(gray, kernel_size=DILATATION_EROSION_KERNEL_SIZE)
        cc_data = ConnectedComponentsData(image=dilated, image_name=path)
    else:
        cc_data = ConnectedComponentsData(image=gray, image_name=path)

    # if must_draw:
    #     for cc_n in range(0, cc_data.nb_labels()):
    #         stats = cc_data.component_stats(cc_n)
    #         draw_info_cc(image, path, cc_n, stats["x"], stats["y"], stats["width"], stats["height"],
    #                      stats["centroids"][0], stats["centroids"][1])
    return cc_data


def plotting_frequency(unique, count, plot_color='orange', show_plot=True):
    plt.figure(figsize=(10, 7), dpi=200)
    plt.plot(unique, count, color=plot_color)
    plt.xlabel("Unique values", fontsize='x-large')
    plt.ylabel("Frequency", fontsize='x-large')
    plt.xlim(left=-10, right=100)
    if show_plot:
        plt.show()


def plotting_frequency_4(unique_1, count_1, unique_2, count_2, unique_3, count_3, unique_4, count_4):
    plt.figure(figsize=(7, 10), dpi=200)
    plt.plot(unique_1, count_1, color="orange")
    plt.plot(unique_2, count_2, color="green")
    plt.plot(unique_3, count_3, color="red")
    plt.plot(unique_4, count_4, color="blue")
    plt.legend(["norm hist - nb labels = 2", "norm hist - nb labels = 3", "pas norm - nb labels = 2",
                "pas norm - nb labels = 3"])
    plt.xlabel("Aire des taches (pixel)", fontsize='x-large')
    plt.ylabel("Fréquence", fontsize='x-large')
    plt.xlim(left=-5, right=250)
    plt.title("Aire des taches avec {}-connexité\nDataset complet - Avec Érosion-Dilatation".format(CC_CONNECTIVITY),
              size="xx-large")
    plt.show()


def print_stat_moments(areas: np.ndarray):
    print("Stats about area")
    print("mean: ", round(np.mean(areas), ndigits=6))
    print("std: ", round(np.std(areas), ndigits=6))
    print("median: ", int(round(np.median(areas), ndigits=6)))
    print("skewness (corr bias): ", round(spstats.skew(areas, bias=False), ndigits=6))
    print("kurtosis (fisher + corr bias): ", round(spstats.kurtosis(areas, fisher=True, bias=False), ndigits=6))
    print("nb areas: ", len(areas))


def make_image_list(source_dir: str) -> list:
    images_paths = []
    for file_path in tqdm(os.listdir(source_dir)):
        if not os.path.isdir(file_path):
            images_paths.append(os.path.join(source_dir, file_path))
    return images_paths


def aug(elem1: float, elem2: float):
    dif_abs = elem2 - elem1
    div = dif_abs/elem1
    return 100*div


if __name__ == "__main__":
    images_paths_histflat_n2_all = make_image_list(os.path.join(os.getcwd(), "trainset_color_segmented_normalized_histflat_numclusters_2_all_images"))
    images_paths_histflat_n3_all = make_image_list(os.path.join(os.getcwd(), "trainset_color_segmented_normalized_histflat_numclusters_3_all_images"))
    images_paths_nonorm_n2_all = make_image_list(os.path.join(os.getcwd(), "trainset_color_segmented_not_normalized_numclusters_2_all_images"))
    images_paths_nonorm_n3_all = make_image_list(os.path.join(os.getcwd(), "trainset_color_segmented_not_normalized_numclusters_3_all_images"))

    plt.figure(figsize=(10, 7), dpi=200)
    # 1
    cc_data_set = analyse_cc(images_paths_histflat_n2_all, use_erosion_dilatation_opening=True)
    unique_1, count_1, areas_np_1 = cc_data_set.compute_frequency()
    plotting_frequency(unique_1, count_1, show_plot=True)
    print_stat_moments(areas_np_1)
    # 2
    cc_data_set = analyse_cc(images_paths_histflat_n3_all, use_erosion_dilatation_opening=True)
    unique_2, count_2, areas_np_2 = cc_data_set.compute_frequency()
    # plotting_frequency(unique_2, count_2, plot_color="green", show_plot=True)
    print_stat_moments(areas_np_2)
    # 3
    cc_data_set = analyse_cc(images_paths_nonorm_n2_all, use_erosion_dilatation_opening=True)
    unique_3, count_3, areas_np_3 = cc_data_set.compute_frequency()
    # plotting_frequency(unique_3, count_3, plot_color="red", show_plot=True)
    print_stat_moments(areas_np_3)
    # 4
    cc_data_set = analyse_cc(images_paths_nonorm_n3_all, use_erosion_dilatation_opening=True)
    unique_4, count_4, areas_np_4 = cc_data_set.compute_frequency()
    # plotting_frequency(unique_4, count_4, plot_color="blue")
    print_stat_moments(areas_np_4)

    plotting_frequency_4(unique_1, count_1, unique_2, count_2, unique_3, count_3, unique_4, count_4)

    # json = cc_data_set.to_json()
    # jsonFile = open("cc_data_set.json", "w")
    # jsonFile.write(json)
    # jsonFile.close()

