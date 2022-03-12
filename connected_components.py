import json
from json import JSONEncoder
from typing import List, Any

import cv2
import os
import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm


class ConnectedComponentsData:
    image_name: str
    num_labels: int
    # labels: np.ndarray    # si on met ce champs lors du dump json, il n'y a plus d'espace mémoire
    stats: np.ndarray
    centroids: np.ndarray
    avg_area: float


    def __init__(self, image: np.ndarray, image_name: str):
        output = cv2.connectedComponentsWithStats(image, connectivity=8, ltype=cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output
        self.num_labels = numLabels
        # self.labels = labels
        self.stats = stats
        self.centroids = centroids
        self.image_name = image_name

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
    cv2.imwrite(path[:-4] + "_cc{}_.png".format(cc_n), img_cpy)
    return img_cpy


def analyse_cc(image_paths: list) -> CCDataSet:
    """
    Analyze the connected components for all the images contained in image_paths
    :param image_paths: a list containing paths to images that needs to be analyzed
    :return: a CCDataset object containing the cc infos for all of the images
    """
    data_set = CCDataSet()
    for path in tqdm(image_paths):
        image = cv2.imread(path)
        data = analyse_one_cc(image, path)
        data_set.add(data)

    return data_set


def analyse_one_cc(image: np.ndarray, path: str, must_draw: bool = False) -> ConnectedComponentsData:
    """
    Analyze the connected components for one image
    :param image: a cv2 loaded image
    :param path: the path of the image
    :param must_draw: a boolean, if set to true, the connected components will be drawn one photo at the time and saved
    in the same folder as path
    :return: a connectedComponentData object describing the connected components
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cc_data = ConnectedComponentsData(image=gray, image_name=path)

    if must_draw:
        for cc_n in range(0, cc_data.nb_labels()):
            stats = cc_data.component_stats(cc_n)
            draw_info_cc(image, path, cc_n, stats["x"], stats["y"], stats["width"], stats["height"],
                         stats["centroids"][0], stats["centroids"][1])
    return cc_data


def plotting_frequency(unique, count):
    plt.plot(unique, count)
    plt.xlabel("Unique values")
    plt.ylabel("Frequency")
    plt.show()


if __name__ == "__main__":
    source_dir = os.path.join(os.getcwd(), "trainset_color_segmented")
    images_paths = []
    for file_path in tqdm(os.listdir(source_dir)):
        if not os.path.isdir(file_path):
            images_paths.append(os.path.join(source_dir, file_path))

    cc_data_set = analyse_cc(images_paths)
    unique, count, areas_np = cc_data_set.compute_frequency()
    json = cc_data_set.to_json()
    jsonFile = open("cc_data_set.json", "w")
    jsonFile.write(json)
    jsonFile.close()
