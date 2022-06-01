import json
import threading
from typing import List, Optional, Tuple

import time
import pandas
import cv2
import numpy as np

import connected_components as cc
import contours as ct
import fingerprint as fgp

import os
from tqdm import tqdm
import pandas as pd

PRINT_SYSTEM_OUT = True     # if set to true, the script will write on system out

USE_MULTITHREADING = True
SIMILARITY_THRESHOLD = 0.0
#DILATATION_EROSION_KERNEL_SIZE = 1
USE_EROSION_DILATATION = True

PERFORMANCE_TEST_TABLE_FILENAME = os.path.join(os.getcwd(), "performances_tables", "perf_total.csv")
SOURCE_COLOR_SEGMENTED_IMAGES_FOLDER = os.path.join(os.getcwd(), "trainset_color_segmented_normalized_histflat_numclusters_2_all_images")


class ConfusionMatrix:
    true_positive: int
    false_negative: int
    false_positive: int
    true_negative: int
    precision: float
    recall: float
    accuracy: float
    balanced_accuracy: float

    def __init__(self, true_positive: int, false_negative: int, false_positive: int, true_negative: int):
        self.true_positive = true_positive
        self.false_negative = false_negative
        self.false_positive = false_positive
        self.true_negative = true_negative

        self.get_precision()
        self.get_recall()
        self.get_accuracy()
        self.get_balanced_accuracy()

    def get_precision(self) -> float:
        precision = self.true_positive / (self.true_positive + self.false_positive)
        self.precision = precision
        return precision

    def get_recall(self) -> float:
        recall = self.true_positive / (self.true_positive + self.false_negative)
        self.recall = recall
        return recall

    def get_accuracy(self) -> float:
        accuracy = (self.true_positive + self.true_negative) / \
                   (self.true_positive + self.true_negative + self.false_positive + self.false_negative)
        self.accuracy = accuracy
        return accuracy

    def get_balanced_accuracy(self) -> float:
        balanced_accuracy = 0.5 * ((self.true_positive / (self.true_positive + self.false_negative))
                                   + (self.true_negative / (self.true_negative + self.false_positive)))
        self.balanced_accuracy = balanced_accuracy
        return balanced_accuracy

    def to_json_file(self, filepath: str) -> str:
        json_string = json.dumps(self.__dict__)
        with open(filepath, 'w') as outfile:
            outfile.write(json_string)
        return json_string

    def from_json_file(self, filepath: str) -> dict:
        with open(filepath) as json_file:
            data = json.load(json_file)
            self.true_positive = data["true_positive"]
            self.false_negative = data["false_negative"]
            self.false_positive = data["false_positive"]
            self.true_negative = data["true_negative"]
            return data

    def __str__(self):
        return "TP:{},FN:{}\nFP:{},TN:{}\nacc:{},bal_acc:{}" \
            .format(self.true_positive, self.false_negative, self.false_positive, self.true_negative, self.accuracy,
                    self.balanced_accuracy)


class SalamImage:
    individual_id: str
    image_path: str
    cc_data: cc.ConnectedComponentsData
    contours_data: ct.ContoursData
    global_histogram: fgp.PolarHistogram

    def __init__(self, image_path: str, cc_data: cc.ConnectedComponentsData = None,
                 global_histogram: fgp.PolarHistogram = None, contours_data: ct.ContoursData = None) \
            -> None:
        self.image_path = image_path
        self.cc_data = cc_data
        self.global_histogram = global_histogram
        self.contours_data = contours_data

    def __str__(self):
        if self.cc_data is not None:
            return "SalamImage of individual {}, imagepath: {}, cc: {}, hist: {}" \
                .format(self.individual_id, os.sep + "..." + self.image_path.split(os.sep)[-1], self.cc_data,
                        self.global_histogram)
        elif self.contours_data is not None:
            return "SalamImage of individual {}, imagepath: {}, ct: {}, hist: {}" \
                .format(self.individual_id, os.sep + "..." + self.image_path.split(os.sep)[-1], self.contours_data,
                        self.global_histogram)
        else:
            return "SalamImage of individual {}, imagepath: {}, hist: {}" \
                .format(self.individual_id, os.sep + "..." + self.image_path.split(os.sep)[-1], self.global_histogram)


class HistogramComparisonResult:
    salam_image_1: SalamImage
    salam_image_2: SalamImage
    similarity_probability: float
    is_similar: bool

    def __init__(self, salam_image_1: SalamImage, salam_image_2: SalamImage, similarity_probability: float,
                 is_similar: bool):
        self.salam_image_1 = salam_image_1
        self.salam_image_2 = salam_image_2
        self.similarity_probability = similarity_probability
        self.is_similar = is_similar

    def __str__(self):
        return "HistCompRes of Image 1 ({}) and Image 2 ({}), simprob: {}, is it similar? {}" \
            .format(self.salam_image_1, self.salam_image_2, self.similarity_probability, self.is_similar)

    def __repr__(self):
        return "HistCompRes of Img1 ({}) + Img2 ({}), simprob: {}, similar? {}" \
            .format(self.salam_image_1, self.salam_image_2, self.similarity_probability, self.is_similar)


class IndividualComparisonResult:
    id_number: int
    histograms_comparisons_results: List[HistogramComparisonResult]
    salam_images: List[SalamImage]

    # statistics
    avg_probability: float
    percentage_genuine: float
    percentage_impostor: float
    nb_genuine: int
    nb_impostor: int

    def __init__(self, id_number: int, histograms_comparisons_results: List[HistogramComparisonResult],
                 salam_images: List[SalamImage]):
        self.id_number = id_number
        self.histograms_comparisons_results = histograms_comparisons_results
        self.salam_images = salam_images

        self.make_statistics()

    def make_statistics(self):
        avg_probability: float
        probabilities = 0
        for hcr in self.histograms_comparisons_results:
            probabilities += hcr.similarity_probability
        avg_probability = probabilities / len(self.histograms_comparisons_results)

        percentage_genuine: float
        percentage_impostor: float
        nb_genuine: int = 0
        nb_impostor: int = 0
        for hcr in self.histograms_comparisons_results:
            if hcr.is_similar:
                nb_genuine += 1
            else:
                nb_impostor += 1

        self.nb_impostor = nb_impostor
        self.nb_genuine = nb_genuine
        percentage_genuine = nb_genuine / len(self.histograms_comparisons_results)
        percentage_impostor = nb_impostor / len(self.histograms_comparisons_results)

        self.avg_probability = avg_probability
        self.percentage_genuine = percentage_genuine
        self.percentage_impostor = percentage_impostor

    def top_n_similarities(self, n: int) -> List[HistogramComparisonResult]:
        """
        Find the top n best similarities for this individual, returning a list of HistogramComparisonResult of size n
        :param n: an integer bigger than 0
        :return:
        """
        assert n >= 0
        if n == 0:
            return []
        top_n = []
        for i in range(n):
            best_sim = None
            for hist_res in self.histograms_comparisons_results:
                if top_n.__contains__(hist_res):  # we do not take into account the elements already in the list
                    continue
                if best_sim is None:  # we establish the point of comparison
                    best_sim = hist_res
                if hist_res.similarity_probability > best_sim.similarity_probability:
                    best_sim = hist_res
            top_n.append(best_sim)
        return top_n

    def __str__(self):
        return "IndCompRes of id {} (nb histoCompResults: {}, nb salamImage: {}, avg proba: {}, perc genuine: {}, " \
               "perc impostor: {})" \
            .format(self.id_number, len(self.histograms_comparisons_results), len(self.salam_images),
                    self.avg_probability, self.percentage_genuine, self.percentage_impostor)


def make_histograms(images_paths: list, use_cc: bool = True) -> List[SalamImage]:
    """
    computre histograms for the images given in input, it either uses the connected components or the contours border
    to compute the histograms
    :param images_paths:
    :param use_cc: if true, it uses the connected components, else it uses the contours
    :return: a list of SalamImages
    """
    if use_cc:
        if PRINT_SYSTEM_OUT: print("Retrieving CC info...")
        cc_data_set = cc.analyse_cc(images_paths, use_erosion_dilatation_opening=USE_EROSION_DILATATION)
        if PRINT_SYSTEM_OUT: print("Computing histograms...")
        salam_images: List[SalamImage] = []
        for img_cc_data in tqdm(cc_data_set.cc_data, disable=not PRINT_SYSTEM_OUT):
            assert type(img_cc_data) is cc.ConnectedComponentsData
            centroids = img_cc_data.centroids
            histogram = fgp.make_polar_histogram(centroids, name=img_cc_data.image_name)
            salam_images.append(SalamImage(img_cc_data.image_name, img_cc_data, histogram))

        return salam_images
    else:
        if PRINT_SYSTEM_OUT: print("Computing histograms using contours info...")
        salam_images: List[SalamImage] = []
        for image_path in tqdm(images_paths, disable=not PRINT_SYSTEM_OUT):
            image: np.ndarray = cv2.imread(image_path)
            # opened_image = de.opening(image, DILATATION_EROSION_KERNEL_SIZE)
            global_histogram, contour_data = ct.create_histograms_from_contours(image, image_filepath=image_path)
            salam_images.append(SalamImage(image_path, global_histogram=global_histogram, contours_data=contour_data))

        return salam_images


def get_image_paths(source_folder: str = SOURCE_COLOR_SEGMENTED_IMAGES_FOLDER) -> list:
    source_dir = source_folder
    images_paths = []
    for file_path in tqdm(os.listdir(source_dir), disable=not PRINT_SYSTEM_OUT):
        if not os.path.isdir(file_path):
            images_paths.append(os.path.join(source_dir, file_path))
    return images_paths


def find_histogram_by_filename(salam_images: List[SalamImage], filename) -> Optional[SalamImage]:
    for si in salam_images:
        if si.global_histogram.short_name == filename:
            return si
    return None


prefix_name_set = ""


# prefix_name_set = "mandeldemo_"

def make_true_similar_set(salam_images: List[SalamImage], df: pandas.DataFrame) -> List[List[SalamImage]]:
    unique_salams = df["salam_id"].unique()

    salam_images_by_individual: List[List[SalamImage]] = []
    k = 0
    for individual_id in unique_salams:
        salam_images_by_individual.append([])
        df_image_names = df.loc[df["salam_id"] == individual_id]
        for index, row in df_image_names.iterrows():
            salam_image: SalamImage = find_histogram_by_filename(salam_images, prefix_name_set + row[0])
            # We append the individual_id to each image
            salam_image.individual_id = individual_id
            salam_images_by_individual[k].append(salam_image)
        k += 1
    return salam_images_by_individual


def make_true_difference_set(salam_images: List[SalamImage], df: pandas.DataFrame) -> List[List[SalamImage]]:
    salam_images_false_set: List[List[SalamImage]] = []
    for elem in df.index:
        elem_salam_image: SalamImage = find_histogram_by_filename(salam_images,
                                                                  prefix_name_set + df.iloc[elem][0])
        individual_id = df.iloc[elem][1]
        df_image_names = df.loc[df["salam_id"] > individual_id]
        for index, row in df_image_names.iterrows():
            salam_image: SalamImage = find_histogram_by_filename(salam_images, prefix_name_set + row[0])
            salam_image.individual_id = row[1]
            salam_images_false_set.append([elem_salam_image, salam_image])
    return salam_images_false_set


def performance_test(salam_images: List[SalamImage]) -> Tuple[
    List[IndividualComparisonResult], List[HistogramComparisonResult], dict]:
    """
    Do a performance test on the salamanders images given as input, with the true table written on the file
    TabFsmSghImg_lauz.csv. Two sets of results are created, the first is the set containing the results of image
    comparisons of the same individual, the second contains the results of image comparisons of different individuals

    :param salam_images:
    :return: a tuple of a list of IndividualComparisonResult objects (list of results of image comparisons of the same
    individual) and a list of HistogramComparisonResult objects (list of results of images comparisons of different
    individuals)
    """
    sim_data_filename = PERFORMANCE_TEST_TABLE_FILENAME
    # sim_data_filename = "TabFsmSghImg_mandel.csv"
    df = pd.read_csv(sim_data_filename, sep=";", dtype={"filename": str, "salam_id": str})

    # Here, we group the histograms made for the same individual
    salam_images_true_similar_set: List[List[SalamImage]] = make_true_similar_set(salam_images, df)
    # Here, we group the histograms that are not of the same individual by pair, when will come the time to compute the
    # probabilities, it will be necessary to compute them on the pair
    salam_images_true_different_set: List[List[SalamImage]] = make_true_difference_set(salam_images, df)
    # We now compute the probabilities of similarity between each histogram (here each photo) for each individual
    true_similar_set_comparison_results: List[IndividualComparisonResult] = []
    # STATS
    avg_similarity_probability_true_similar = 0.0
    avg_similarity_probability_true_different = 0.0
    true_similar = 0
    true_different = 0
    alpha_error = 0
    beta_error = 0
    # MAIN RUN
    # True similars
    if PRINT_SYSTEM_OUT: print("Computing truth-table positive rate...")
    if USE_MULTITHREADING:
        # the mutltithreading strategy here is to make a new thread for each individual's comparisons
        # thus, some threads will finish sooner than others (which is fine) depending on the number of images
        # per individual
        threads = []
        monitor_threads_compare = ThreadCompareIndividuals()
        for n in range(0, len(salam_images_true_similar_set)):
            images = salam_images_true_similar_set[n]
            if len(images) == 1:  # if there is only one image per individual, we pass
                continue
            t = threading.Thread(target=monitor_threads_compare.compare_individuals, args=(images,))
            threads.append(t)

        time_thread_start = time.time()
        for th in threads:
            th.start()
        for th in tqdm(threads, disable=not PRINT_SYSTEM_OUT):
            th.join()
        time_thread_end = time.time()
        if PRINT_SYSTEM_OUT: print("Time of execution :", time_thread_end - time_thread_start)

        true_similar_set_comparison_results: List[
            IndividualComparisonResult] = monitor_threads_compare.true_similar_set_comparison_results
        for icr in true_similar_set_comparison_results:
            true_similar += icr.nb_genuine
            beta_error += icr.nb_impostor
            avg_similarity_probability_true_similar += icr.avg_probability
        avg_similarity_probability_true_similar /= len(true_similar_set_comparison_results)
    else:
        for images in tqdm(salam_images_true_similar_set, disable=not PRINT_SYSTEM_OUT):
            if len(images) == 1:  # if there is only one image per individual, we pass
                continue

            ress = compare_individuals(images)

            true_similar += ress["true_similar"]
            beta_error += ress["beta_error"]
            avg_similarity_probability_true_similar += ress["avg_similarity_probability_true_similar"]
            individual_comparison_result = ress["individual_comparison_result"]
            true_similar_set_comparison_results.append(individual_comparison_result)

        avg_similarity_probability_true_similar /= len(salam_images_true_similar_set)

    if PRINT_SYSTEM_OUT: print("Computing truth-table negative rate...")
    if USE_MULTITHREADING:
        num_threads = 6
        threads = []
        monitor_threads_compare = ThreadCompareDifferent()

        threads_input_chunks = divide_list(salam_images_true_different_set, num_threads)
        for n in range(0, num_threads):
            if not threads_input_chunks[n]:
                continue
            t = threading.Thread(target=monitor_threads_compare.compare_differents,
                                 args=(threads_input_chunks[n],))
            threads.append(t)
        time_thread_start = time.time()
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        time_thread_end = time.time()
        if PRINT_SYSTEM_OUT: print("Time of execution :", time_thread_end - time_thread_start)

        true_different = monitor_threads_compare.true_different
        alpha_error = monitor_threads_compare.alpha_error
        avg_similarity_probability_true_different = monitor_threads_compare.avg_similarity_probability_true_different
        true_different_set_comparison_results = monitor_threads_compare.true_different_set_comparison_results
    else:
        ress = compare_differents(salam_images_true_different_set)
        true_different = ress["true_different"]
        alpha_error = ress["alpha_error"]
        avg_similarity_probability_true_different = ress["avg_similarity_probability_true_different"]
        true_different_set_comparison_results = ress["true_different_set_comparison_results"]

    avg_similarity_probability_true_different /= len(true_different_set_comparison_results)
    avg_similarity_probability_total = \
        (avg_similarity_probability_true_similar + avg_similarity_probability_true_different) * 0.5

    confusion_matrix = ConfusionMatrix(true_similar, beta_error, alpha_error, true_different)

    return true_similar_set_comparison_results, true_different_set_comparison_results, \
           {"avg_proba_true_similar": avg_similarity_probability_true_similar,
            "avg_proba_true_different": avg_similarity_probability_true_different,
            "avg_proba_total": avg_similarity_probability_total,
            "confusion_matrix": confusion_matrix
            }


class ThreadCompareIndividuals:
    # basically do the same as compare_individuals but with a lock for the critical region
    def __init__(self):
        self.lock = threading.Lock()
        self.true_similar_set_comparison_results: List[IndividualComparisonResult] = []

    def compare_individuals(self, images: List[SalamImage]):
        hcr: List[HistogramComparisonResult] = []
        for i in range(0, len(images)):
            for j in range(i + 1, len(images)):
                similarity_probability = fgp.compare_histograms(images[i].global_histogram, images[j].global_histogram)
                is_similar = True if similarity_probability > SIMILARITY_THRESHOLD else False
                hcr.append(HistogramComparisonResult(images[i], images[j], similarity_probability, is_similar))

        # start of critical zone
        self.lock.acquire()
        self.true_similar_set_comparison_results.append(
            IndividualComparisonResult(images[0].individual_id, hcr, images))
        self.lock.release()
        # end of critical zone


class ThreadCompareDifferent:
    def __init__(self):
        self.lock = threading.Lock()
        self.true_different = 0
        self.alpha_error = 0
        self.avg_similarity_probability_true_different = 0.0
        self.true_different_set_comparison_results: List[HistogramComparisonResult] = []

    def compare_differents(self, salam_images_true_different_set: List[List[SalamImage]]):
        ress = compare_differents(salam_images_true_different_set)

        self.lock.acquire()
        self.true_different += ress["true_different"]
        self.alpha_error += ress["alpha_error"]
        self.avg_similarity_probability_true_different += ress["avg_similarity_probability_true_different"]
        for hcr in ress["true_different_set_comparison_results"]:
            self.true_different_set_comparison_results.append(hcr)
        self.lock.release()


def divide_list(lst, n):
    p = len(lst) // n
    if len(lst) - p > 0:
        return [lst[:p]] + divide_list(lst[p:], n - 1)
    else:
        return [lst]


def compare_individuals(images: List[SalamImage]):
    true_similar = 0.0
    beta_error = 0.0
    avg_similarity_probability_true_similar = 0.0
    hcr: List[HistogramComparisonResult] = []
    for i in range(0, len(images)):
        for j in range(i + 1, len(images)):
            similarity_probability = fgp.compare_histograms(images[i].global_histogram, images[j].global_histogram)
            is_similar = True if similarity_probability > SIMILARITY_THRESHOLD else False
            hcr.append(HistogramComparisonResult(images[i], images[j], similarity_probability, is_similar))
            if is_similar:
                true_similar += 1
            else:
                beta_error += 1
            avg_similarity_probability_true_similar += similarity_probability
    # by construction, each SalamImage in the 'images' array have the same individual_id
    return {"true_similar": true_similar, "beta_error": beta_error,
            "avg_similarity_probability_true_similar": avg_similarity_probability_true_similar,
            "individual_comparison_result": IndividualComparisonResult(images[0].individual_id, hcr, images)}


def compare_differents(salam_images_true_different_set: List[List[SalamImage]]):
    avg_similarity_probability_true_different = 0.0
    true_different = 0.0
    alpha_error = 0.0
    true_different_set_comparison_results: List[HistogramComparisonResult] = []
    for pair in tqdm(salam_images_true_different_set, disable=not PRINT_SYSTEM_OUT):
        similarity_probability = fgp.compare_histograms(pair[0].global_histogram, pair[1].global_histogram)
        is_similar = True if similarity_probability > SIMILARITY_THRESHOLD else False
        true_different_set_comparison_results.append(HistogramComparisonResult(pair[0], pair[1], similarity_probability,
                                                                               is_similar))
        if is_similar:
            alpha_error += 1
        else:
            true_different += 1
        avg_similarity_probability_true_different += similarity_probability
    return {"true_different": true_different, "alpha_error": alpha_error,
            "avg_similarity_probability_true_different": avg_similarity_probability_true_different,
            "true_different_set_comparison_results": true_different_set_comparison_results}


def nb_histograms_true_similar(icr_list: List[IndividualComparisonResult]) -> int:
    nb_histograms = 0
    for icr in icr_list:
        nb_histograms += len(icr.histograms_comparisons_results)
    return nb_histograms


def run() -> List[SalamImage]:
    if PRINT_SYSTEM_OUT: print("Retrieving image paths...")
    image_paths = get_image_paths()
    salam_images = make_histograms(image_paths, True)
    return salam_images


if __name__ == "__main__":
    if USE_MULTITHREADING:
        if PRINT_SYSTEM_OUT: print("Multithreading activated")
    salamImages = run()
    # hists[0].show_img_radius(img_width=1024, img_height=1365)
    if PRINT_SYSTEM_OUT: print("Done !")
    if PRINT_SYSTEM_OUT: print("Performance test")
    individuals_comparison_results, different_comparison_results, stats = performance_test(salamImages)
    print(stats)
    print("balacc ", stats["confusion_matrix"].balanced_accuracy)