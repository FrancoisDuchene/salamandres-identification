from typing import List, Optional, Tuple

import pandas

import connected_components as cc
import fingerprint as fgp

import os
from tqdm import tqdm
import pandas as pd


SIMILARITY_THRESHOLD = 0.2


class SalamImage:
    individual_id: int
    image_path: str
    cc_data: cc.ConnectedComponentsData
    global_histogram: fgp.PolarHistogram

    def __init__(self, image_path: str, cc_data: cc.ConnectedComponentsData, global_histogram: fgp.PolarHistogram) \
            -> None:
        self.image_path = image_path
        self.cc_data = cc_data
        self.global_histogram = global_histogram

    def __str__(self):
        return "SalamImage of individual {}, imagepath: {}, cc: {}, hist: {}"\
            .format(self.individual_id, os.sep + "..." + self.image_path.split(os.sep)[-1], self.cc_data,
                    self.global_histogram)


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
        return "HistCompRes of Image 1 ({}) and Image 2 ({}), simprob: {}, it is similar? {}"\
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

    def __str__(self):
        return "IndCompRes of id {} (nb histoCompResults: {}, nb salamImage: {}, avg proba: {}, perc genuine: {}, " \
               "perc impostor: {})"\
            .format(self.id_number, len(self.histograms_comparisons_results), len(self.salam_images),
                    self.avg_probability, self.percentage_genuine, self.percentage_impostor)



def make_histograms(images_paths) -> List[SalamImage]:
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


def get_image_paths() -> list:
    source_dir = os.path.join(os.getcwd(), "trainset_color_segmented")
    images_paths = []
    for file_path in tqdm(os.listdir(source_dir)):
        if not os.path.isdir(file_path):
            images_paths.append(os.path.join(source_dir, file_path))
    return images_paths


def run() -> List[SalamImage]:
    print("Retrieving image paths...")
    image_paths = get_image_paths()
    salam_images = make_histograms(image_paths)
    return salam_images


def find_histogram_by_filename(salam_images: List[SalamImage], filename) -> Optional[SalamImage]:
    for si in salam_images:
        if si.global_histogram.short_name == filename:
            return si
    return None


def make_true_similar_set(salam_images: List[SalamImage], df: pandas.DataFrame) -> List[List[SalamImage]]:
    unique_salams = df["salam_id"].unique()

    salam_images_by_individual: List[List[SalamImage]] = []
    k = 0
    for individual_id in tqdm(unique_salams):
        salam_images_by_individual.append([])
        df_image_names = df.loc[df["salam_id"] == individual_id]
        for index, row in df_image_names.iterrows():
            salam_image: SalamImage = find_histogram_by_filename(salam_images, "mandeldemo_" + row[0] + ".png")
            # We append the individual_id to each image
            salam_image.individual_id = individual_id
            salam_images_by_individual[k].append(salam_image)
        k += 1
    return salam_images_by_individual


def make_true_difference_set(salam_images: List[SalamImage], df: pandas.DataFrame) -> List[List[SalamImage]]:
    salam_images_false_set: List[List[SalamImage]] = []
    for elem in df.index:
        elem_salam_image: SalamImage = find_histogram_by_filename(salam_images, "mandeldemo_" + df.iloc[elem][0] + ".png")
        individual_id = df.iloc[elem][1]
        df_image_names = df.loc[df["salam_id"] > individual_id]
        for index, row in df_image_names.iterrows():
            salam_image: SalamImage = find_histogram_by_filename(salam_images, "mandeldemo_" + row[0] + ".png")
            salam_image.individual_id = row[1]
            salam_images_false_set.append([elem_salam_image, salam_image])
    return salam_images_false_set


def performance_test(salam_images: List[SalamImage]) -> Tuple[List[IndividualComparisonResult], List[HistogramComparisonResult]]:
    # sim_data_filename = "TabFsmSghImg_lauz.csv"
    sim_data_filename = "TabFsmSghImg_mandel.csv"
    df = pd.read_csv(sim_data_filename, sep=";", dtype={"filename": str, "salam_id": int})

    # Here, we group the histograms made for the same individual
    salam_images_true_similar_set: List[List[SalamImage]] = make_true_similar_set(salam_images, df)
    # Here, we group the histograms that are not of the same individual by pair, when will come the time to compute the
    # probabilities, it will be necessary to compute them on the pair
    salam_images_true_different_set: List[List[SalamImage]] = make_true_difference_set(salam_images, df)
    # We now compute the probabilities of similarity between each histogram (here each photo) for each individual
    true_similar_set_comparison_results: List[IndividualComparisonResult] = []
    for images in tqdm(salam_images_true_similar_set):
        if len(images) == 1:    # if there is only one image per individual, we pass
            continue
        hcr: List[HistogramComparisonResult] = []
        for i in range(0, len(images)):
            for j in range(i+1, len(images)):
                similarity_probability = fgp.compare_histograms(images[i].global_histogram, images[j].global_histogram)
                hcr.append(HistogramComparisonResult(images[i], images[j], similarity_probability,
                                                     True if similarity_probability > SIMILARITY_THRESHOLD else False)
                           )
        # by construction, each SalamImage in the 'images' array have the same individual_id
        true_similar_set_comparison_results.append(IndividualComparisonResult(images[0].individual_id, hcr, images))

    true_different_set_comparison_results: List[HistogramComparisonResult] = []
    for pair in tqdm(salam_images_true_different_set):
        similarity_probability = fgp.compare_histograms(pair[0].global_histogram, pair[1].global_histogram)
        true_different_set_comparison_results.append(
            HistogramComparisonResult(pair[0], pair[1], similarity_probability,
                                      True if similarity_probability > SIMILARITY_THRESHOLD else False)
        )

    return df, true_similar_set_comparison_results, true_different_set_comparison_results


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
        avg_perc_alpha_error += icr.percentage_impostor
        avg_similarity_probability_true_similar += icr.avg_probability

    avg_perc_true_similar /= len(true_similar_set_comparison_results)
    avg_perc_alpha_error /= len(true_similar_set_comparison_results)
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
    avg_perc_beta_error = counter_be / len(true_different_set_comparison_results)
    avg_similarity_probability_true_different /= len(true_different_set_comparison_results)

    avg_similarity_probability_total = \
        (avg_similarity_probability_true_similar + avg_similarity_probability_true_different) * 0.5

    return {"avg_proba_true_similar": avg_similarity_probability_true_similar,
            "avg_proba_true_different": avg_similarity_probability_true_different,
            "avg_perc_true_similar": avg_perc_true_similar, "avg_perc_true_different": avg_perc_true_different,
            "avg_perc_alpha_error": avg_perc_alpha_error, "avg_perc_beta_error": avg_perc_beta_error,
            "avg_proba_total": avg_similarity_probability_total}


if __name__ == "__main__":
    hists = run()
    # hists[0].show_img_radius(img_width=1024, img_height=1365)
    print("Done !")
    print("Performance test")
    df, icr, fscr = performance_test(hists)
    print("Performing statistics")
    stats = make_statistics(icr, fscr)