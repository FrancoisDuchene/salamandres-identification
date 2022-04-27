import sys
from typing import List, Optional, Tuple

import pandas

import connected_components as cc
import contours as ct
import fingerprint as fgp

import os
from tqdm import tqdm
import pandas as pd


SIMILARITY_THRESHOLD = 0


class SalamImage:
    individual_id: int
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
        return "HistCompRes of Image 1 ({}) and Image 2 ({}), simprob: {}, is it similar? {}"\
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

    def __init__(self, id_number: int, histograms_comparisons_results: List[HistogramComparisonResult], salam_images: List[SalamImage]):
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
                if top_n.__contains__(hist_res):    # we do not take into account the elements already in the list
                    continue
                if best_sim is None:    # we establish the point of comparison
                    best_sim = hist_res
                if hist_res.similarity_probability > best_sim.similarity_probability:
                    best_sim = hist_res
            top_n.append(best_sim)
        return top_n

    def __str__(self):
        return "IndCompRes of id {} (nb histoCompResults: {}, nb salamImage: {}, avg proba: {}, perc genuine: {}, " \
               "perc impostor: {})"\
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
        print("Retrieving CC info...")
        cc_data_set = cc.analyse_cc(images_paths)
        print("Computing histograms...")
        salam_images: List[SalamImage] = []
        for img_cc_data in tqdm(cc_data_set.cc_data):
            assert type(img_cc_data) is cc.ConnectedComponentsData
            centroids = img_cc_data.centroids
            histogram = fgp.make_polar_histogram(centroids, name=img_cc_data.image_name)
            salam_images.append(SalamImage(img_cc_data.image_name, img_cc_data, histogram))

        return salam_images
    else:
        print("Computing histograms using contours info...")
        salam_images: List[SalamImage] = []
        for image_path in tqdm(images_paths):
            global_histogram, contour_data = ct.create_histograms_from_contours(image_path)
            salam_images.append(SalamImage(image_path, global_histogram=global_histogram, contours_data=contour_data))

        return salam_images


def get_image_paths() -> list:
    source_dir = os.path.join(os.getcwd(), "trainset_color_segmented")
    images_paths = []
    for file_path in tqdm(os.listdir(source_dir)):
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
            salam_image: SalamImage = find_histogram_by_filename(salam_images, prefix_name_set + row[0] + ".png")
            # We append the individual_id to each image
            salam_image.individual_id = individual_id
            salam_images_by_individual[k].append(salam_image)
        k += 1
    return salam_images_by_individual


def make_true_difference_set(salam_images: List[SalamImage], df: pandas.DataFrame) -> List[List[SalamImage]]:
    salam_images_false_set: List[List[SalamImage]] = []
    for elem in df.index:
        elem_salam_image: SalamImage = find_histogram_by_filename(salam_images, prefix_name_set + df.iloc[elem][0] + ".png")
        individual_id = df.iloc[elem][1]
        df_image_names = df.loc[df["salam_id"] > individual_id]
        for index, row in df_image_names.iterrows():
            salam_image: SalamImage = find_histogram_by_filename(salam_images, prefix_name_set + row[0] + ".png")
            salam_image.individual_id = row[1]
            salam_images_false_set.append([elem_salam_image, salam_image])
    return salam_images_false_set


def performance_test(salam_images: List[SalamImage]) -> Tuple[List[IndividualComparisonResult], List[HistogramComparisonResult], dict]:
    """
    Do a performance test on the salamanders images given as input, with the true table written on the file
    TabFsmSghImg_lauz.csv. Two sets of results are created, the first is the set containing the results of image
    comparisons of the same individual, the second contains the results of image comparisons of different individuals

    :param salam_images:
    :return: a tuple of a list of IndividualComparisonResult objects (list of results of image comparisons of the same
    individual) and a list of HistogramComparisonResult objects (list of results of images comparisons of different
    individuals)
    """
    sim_data_filename = "TabFsmSghImg_lauz_small.csv"
    # sim_data_filename = "TabFsmSghImg_mandel.csv"
    df = pd.read_csv(sim_data_filename, sep=";", dtype={"filename": str, "salam_id": int})

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
    avg_perc_true_similar = 0.0
    avg_perc_true_different = 0.0
    avg_perc_alpha_error = 0.0
    avg_perc_beta_error = 0.0
    # MAIN RUN
    for images in tqdm(salam_images_true_similar_set):
        if len(images) == 1:    # if there is only one image per individual, we pass
            continue
        hcr: List[HistogramComparisonResult] = []
        for i in range(0, len(images)):
            for j in range(i+1, len(images)):
                similarity_probability = fgp.compare_histograms(images[i].global_histogram, images[j].global_histogram)
                is_similar = True if similarity_probability > SIMILARITY_THRESHOLD else False
                hcr.append(HistogramComparisonResult(images[i], images[j], similarity_probability, is_similar))
                if is_similar:
                    avg_perc_true_similar += 1
                else:
                    avg_perc_beta_error += 1
                avg_similarity_probability_true_similar += similarity_probability
        # by construction, each SalamImage in the 'images' array have the same individual_id
        true_similar_set_comparison_results.append(IndividualComparisonResult(images[0].individual_id, hcr, images))

    true_different_set_comparison_results: List[HistogramComparisonResult] = []
    for pair in tqdm(salam_images_true_different_set):
        similarity_probability = fgp.compare_histograms(pair[0].global_histogram, pair[1].global_histogram)
        is_similar = True if similarity_probability > SIMILARITY_THRESHOLD else False
        true_different_set_comparison_results.append(HistogramComparisonResult(pair[0], pair[1], similarity_probability,
                                                                               is_similar))
        if is_similar:
            avg_perc_alpha_error += 1
        else:
            avg_perc_true_different += 1
        avg_similarity_probability_true_different += similarity_probability

    avg_perc_true_similar /= len(true_similar_set_comparison_results)
    avg_perc_beta_error /= len(true_similar_set_comparison_results)
    avg_perc_true_different /= len(true_different_set_comparison_results)
    avg_perc_alpha_error /= len(true_different_set_comparison_results)
    avg_similarity_probability_true_similar /= len(true_similar_set_comparison_results)
    avg_similarity_probability_true_different /= len(true_different_set_comparison_results)
    avg_similarity_probability_total = \
        (avg_similarity_probability_true_similar + avg_similarity_probability_true_different) * 0.5

    return true_similar_set_comparison_results, true_different_set_comparison_results, \
           {"avg_proba_true_similar": avg_similarity_probability_true_similar,
            "avg_proba_true_different": avg_similarity_probability_true_different,
            "avg_perc_true_similar": avg_perc_true_similar, "avg_perc_true_different": avg_perc_true_different,
            "avg_perc_alpha_error": avg_perc_alpha_error, "avg_perc_beta_error": avg_perc_beta_error,
            "avg_proba_total": avg_similarity_probability_total,
            "efficiency": (avg_perc_true_similar+avg_perc_true_different)/2
            }


def make_statistics(true_similar_set_comparison_results: List[IndividualComparisonResult],
                    true_different_set_comparison_results: List[HistogramComparisonResult])\
        -> dict:
    avg_similarity_probability_true_similar = 0.0
    avg_similarity_probability_true_different = 0.0

    avg_perc_true_similar = 0.0
    avg_perc_true_different = 0.0
    avg_perc_alpha_error = 0.0
    avg_perc_beta_error = 0.0

    for icr in tqdm(true_similar_set_comparison_results):
        avg_perc_true_similar += icr.percentage_genuine
        avg_perc_beta_error += icr.percentage_impostor
        avg_similarity_probability_true_similar += icr.avg_probability

    avg_perc_true_similar /= len(true_similar_set_comparison_results)
    avg_perc_beta_error /= len(true_similar_set_comparison_results)
    avg_similarity_probability_true_similar /= len(true_similar_set_comparison_results)

    counter_td = 0  # counter for true difference
    counter_be = 0  # counter for beta error
    for hcr in tqdm(true_different_set_comparison_results):
        if hcr.is_similar:
            counter_be += 1
        else:
            counter_td += 1
        avg_similarity_probability_true_different += hcr.similarity_probability
    avg_perc_true_different = counter_td / len(true_different_set_comparison_results)
    avg_perc_alpha_error = counter_be / len(true_different_set_comparison_results)
    avg_similarity_probability_true_different /= len(true_different_set_comparison_results)

    avg_similarity_probability_total = \
        (avg_similarity_probability_true_similar + avg_similarity_probability_true_different) * 0.5

    return {"avg_proba_true_similar": avg_similarity_probability_true_similar,
            "avg_proba_true_different": avg_similarity_probability_true_different,
            "avg_perc_true_similar": avg_perc_true_similar, "avg_perc_true_different": avg_perc_true_different,
            "avg_perc_alpha_error": avg_perc_alpha_error, "avg_perc_beta_error": avg_perc_beta_error,
            "avg_proba_total": avg_similarity_probability_total,
            "efficiency": (avg_perc_true_similar+avg_perc_true_different)/2
            }


def nb_histograms_true_similar(icr_list: List[IndividualComparisonResult]) -> int:
    nb_histograms = 0
    for icr in icr_list:
        nb_histograms += len(icr.histograms_comparisons_results)
    return nb_histograms


def run() -> List[SalamImage]:
    print("Retrieving image paths...")
    image_paths = get_image_paths()
    salam_images = make_histograms(image_paths, False)
    return salam_images


if __name__ == "__main__":
    salamImages = run()
    # hists[0].show_img_radius(img_width=1024, img_height=1365)
    print("Done !")
    print("Performance test")
    icr, fscr, stats = performance_test(salamImages)
    # print("Performing statistics")
    # stats = make_statistics(icr, fscr)