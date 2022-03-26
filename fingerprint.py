import math

import cv2
import numpy as np
import pandas as pd
from typing import List
import os

HISTOGRAM_ANGLE_REGION = (math.pi / 4)  # 45°
HISTOGRAM_NB_ANGLE_REGION = 8
HISTOGRAM_NB_INTERNAL_CIRCLE = 1

GAMMA = 1e-7


class PolarHistogram:
    radius: float = 0.0
    points_coord: np.ndarray
    neighbors: dict = {}
    centroids: dict = {}
    bins: dict = {}
    name: str

    # bins datastructure: a dictionary with points coordinates indexes serving as keys (0 for first point coordinates,
    # 1 for the second, ...). Each point has a dictionary discribing the bins around them in a circular fashion
    # a virtual circle is created around the point, the circle is divided by angle region (the amount and angles are
    # given by HISTOGRAM_NB_ANGLE_REGION and HISTOGRAM_ANGLE_REGION) and by internal circles (the amount is given by
    # HISTOGRAM_NB_INTERNAL_CIRCLE). Each point has a dictionary with listed as keys all the angle regions as an array.
    # The arrays contains the different parts contained into a angle region, index 0 is always the closest region of
    # the point, with the last item in the array as the furthest region

    #      /|\
    #     / | \
    #    /  | 1\
    #   /  /|\  \
    #  /  /1| \  \
    # ------O------
    #  \  \ |2/  /
    #   \  \|/ 1/
    #    \  |  /
    #     \ | /
    #      \|/
    # there are 4 points in total, point at the top is a, the lowest one is d
    # we divide the histogram into four angular regions and two center regions
    # we start counting regions by starting from the region above baseline (here top right)
    # then going clockwise.

    # gives
    # bins = {
    #   1:{
    #     1 : [0,1],
    #     2 : [2,1],
    #     3 : [0,0],
    #     4 : [1,0]
    #   }
    # ...
    # }
    #
    # Each bin itself is an histogram for a particular point

    def __init__(self, points_coord, name=""):
        self.radius = 0
        self.points_coord = np.array([])
        self.neighbors = {}
        self.centroids = {}
        self.bins = {}
        self.name = name

        self.radius = self.histogram_radius(points_coord)
        self.points_coord = points_coord
        self.neighbors = self.number_neighbors()
        self.centroids = self.compute_centroids()  # contains the centroids for each points, a centroid is a (x,y) coordinates
        self.bins = self.compute_bins()

    def add_neighbor(self, p, q):
        """
        add a neighbor to a point contained in self.points_coord; We work with indexes of that array
        :param p: the index of the first point
        :param q: the index of its neighbor
        :return: nothing
        """
        assert p <= self.points_coord.__len__()
        assert q <= self.points_coord.__len__()
        assert p != q
        if p in self.neighbors:
            self.neighbors[p].append(q)
        else:
            self.neighbors[p] = [q]

        if q in self.neighbors:
            if not self.neighbors[q].__contains__(p):
                self.neighbors[q].append(p)
        else:
            self.neighbors[q] = [p]

    def get_neighbors(self, p):
        """
        returns the neighbor for point p, with p as the point index in the self.points_coord array
        :param p: an index of self.points_coord
        :return: a list of neighbors (in the form of array indexes) or an empty array if there is no neighbors
        """
        if p in self.neighbors:
            return self.neighbors[p]
        return []

    @staticmethod
    def histogram_radius(points: np.ndarray):
        m = points.shape[0]
        gamma = .0
        coef = (1 / (2 * m)) + gamma

        doublesum = 0
        for p in range(0, m):
            for q in range(p + 1, m):
                d = dist(points.__getitem__(p), points.__getitem__(q))
                doublesum += 0.5 * d
        radius = coef * doublesum
        return radius

    def number_neighbors(self):
        points = self.points_coord
        m = points.shape[0]

        for p in range(0, m):
            for q in range(p + 1, m):
                pp = points.__getitem__(p)
                pq = points.__getitem__(q)
                euclidian_dist = dist(pp, pq)
                if euclidian_dist <= self.radius:
                    self.add_neighbor(p, q)
        return self.neighbors

    def compute_centroids(self):
        for i in range(0, self.points_coord.__len__()):
            if i in self.neighbors:
                N = self.neighbors[i].__len__()
                arrays = []
                neigh = self.get_neighbors(i)
                for j in neigh:
                    arrays.append(self.points_coord[j])
                centroid = (1 / N) * np.sum(arrays, 0)
                self.centroids[i] = centroid
            else:  # when there's no neighbors
                self.centroids[i] = self.points_coord[i]
        return self.centroids

    def compute_bins(self):
        bins = {}
        for i in range(0, self.points_coord.__len__()):
            bins[i] = {}
            point_i = self.points_coord[i]
            starting_angle = self.angle_two_points(point_i, self.centroids[i])
            transf_x = 0 - point_i[0]
            transf_y = 0 - point_i[1]
            transf_point = (transf_x, transf_y)
            for j in range(0, HISTOGRAM_NB_ANGLE_REGION):
                internal_counter, internal_points = self.internal_region(i, transf_point, starting_angle)
                external_counter = self.external_region(i, transf_point, starting_angle, internal_points)
                bins[i][j] = [internal_counter, external_counter]
                starting_angle += HISTOGRAM_ANGLE_REGION
        return bins

    def internal_region(self, init_point_index: int, init_point_transf: tuple, starting_angle: float):
        """
        Check if points are part of the histogram internal region
        (hypothesising that histogram have only one internal circle)
        :param init_point_index:
        :param init_point_transf:
        :param starting_angle:
        :return: a counter of the number of points in this region, a list of the points in this region (used later)
        """
        counter = 0
        internal_points = []
        tx = init_point_transf[0]
        ty = init_point_transf[1]
        for k in range(0, self.points_coord.__len__()):
            # we continue if k is the point we are making a histogram for
            if k == init_point_index:
                continue
            pi_t = (self.points_coord[k][0] + tx, self.points_coord[k][1] + ty)
            if self.check_point(pi_t[0], pi_t[1], starting_angle, radius=self.radius/2):
                counter += 1
                internal_points.append(self.points_coord[k])
        return counter, internal_points

    def external_region(self, init_point_index: int, init_point_transf: tuple, starting_angle, internal_points: list):
        counter = 0
        tx = init_point_transf[0]
        ty = init_point_transf[1]
        for k in range(0, self.points_coord.__len__()):
            # we continue if k is the point we are making a histogram for or if the point is already part of
            # the region internal points
            if k == init_point_index or self.numpy_array_contains_point(internal_points, self.points_coord[k]):
                continue
            pi_t = (self.points_coord[k][0] + tx, self.points_coord[k][1] + ty)
            if self.check_point(pi_t[0], pi_t[1], starting_angle):
                counter += 1
        return counter

    @staticmethod
    def numpy_array_contains_point(arr: List[np.ndarray], point: np.ndarray):
        for item in arr:
            test: np.ndarray = item == point    # the test will return a new boolean array
            if test.__contains__(False):        # if the boolean array contains false, that means one part is wrong
                continue
            else:                               # if all is clear, that means that point belongs to arr
                return True
        return False

    @staticmethod
    def angle_two_points(p1: tuple, p2: tuple):
        """
        :param p2:
        :return: the angle between p1 and p2 in radian
        """
        p1_t = (0 - p1[0], 0 - p1[1])
        p2_t = (p2[0] + p1_t[0], p2[1] + p1_t[1])
        return math.atan2(p2_t[1], p2_t[0])

    def check_point(self, x, y, startAngle, radius: float = -1):
        """
        check if a point is inside an angle region. We suppose the referential point is at (0,0)
        startAngle: must be in RADIAN
        """
        if radius == -1:
            radius = self.radius
        # calculate endAngle
        endAngle = startAngle + HISTOGRAM_ANGLE_REGION

        # Calculate polar co-ordinates
        polarradius = math.sqrt(x * x + y * y)
        if x == 0:  # to avoid divide by 0 when its a 90° angle
            Angle = math.radians(90)
        else:
            Angle = math.atan2(y, x)

        # since atan2 gives an answer between -pi/2 and pi/2, if we check a region in between these regions,
        # we'll have a problem. If endingAngle is bigger than pi/2 (180), we need to make a conversion on Angle
        # which might be between -pi/2 and 0 so that it can be compared with starting and ending angles.
        # We have the same problem on the other end, when the starting angle is below the 0 point
        # (something lower than 2pi (360) and that the ending angle is above 0 but since it's counting a full turn,
        # it will be more than 360

        if endAngle >= math.radians(360) and Angle >= 0:     # if the angle made a complete tour
            startAngle = startAngle - math.radians(360)
            endAngle = endAngle - math.radians(360)
        elif endAngle > math.radians(180) and Angle < 0:
            rest = math.radians(180) + Angle
            Angle = math.radians(180) + rest

        # Check whether polarradius is less
        # then radius of circle or not and
        # Angle is between startAngle and
        # endAngle or not
        if (startAngle < Angle <= endAngle
                and polarradius < radius):
            # print("Point (", x, ",", y, ") exist in the circle sector")
            return True
        else:
            # print("Point (", x, ",", y, ") does not exist in the circle sector")
            return False

    def show_img_radius(self, img_width: int = 1920, img_height: int = 1080):
        image = np.zeros(shape=[img_width, img_height, 3], dtype=np.uint8)
        col = 0
        for p in self.points_coord:
            cv2.circle(image, (int(p[0]), int(p[1])), 5, (0, 0, 255), -1)
            cv2.circle(image, (int(p[0]), int(p[1])), int(np.round(self.radius)), (0, math.fabs(255 - col),
                                                                                                255), 1)
            col += 3
        cv2.imshow("radius with points", image)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
        cv2.imwrite("{}_hist_circles.png".format(self.name), image)

    def __str__(self):
        return "PolarHistogram \"{}\" of radius {} and {} points"\
            .format(self.name, round(self.radius, 3), self.points_coord.__len__())

    def __repr__(self):
        return "PolarHistogram \"{}\" (r: {}, nb_points: {})"\
            .format(self.name, round(self.radius, 3), self.points_coord.__len__())


def dist(p: tuple, q: tuple):
    def euclidian_dist(p: tuple, q: tuple):
        part1 = np.square(q[0] - p[0])
        part2 = np.square(q[1] - p[1])
        return np.sqrt(part1 + part2)

    return euclidian_dist(p, q)


def chi_square_test(h1: dict, h2: dict) -> float:
    """
    Perform a chi-square type test on two histograms h1 and h2 and returns the result.
    Concretely, it compare each bin separately and sum up the differences.
    It apply the following formula:

    sum  {h1(k) - h2(k)}²/{h1(k) + h2(k) + GAMMA}
     k

    :param h1: first histogram
    :param h2: second histogram
    :return: a score determining the similarity between h1 and h2 (to get a probability, do : 1 - chi_square_test(h1,h2)
    """
    sum_differences_regions = 0
    for region in h1:
        h1_region_horizontal_bins: list = h1[region]   # represent the horizontal bins of a specific angle region
        h2_region_horizontal_bins: list = h2[region]
        for k in range(0,len(h1_region_horizontal_bins)):
            h1_k = h1_region_horizontal_bins[k]
            h2_k = h2_region_horizontal_bins[k]

            numerator = (h1_k - h2_k) ** 2
            denominator = h1_k + h2_k + GAMMA
            sum_differences_regions += (numerator / denominator)
    return 0.5 * sum_differences_regions


def make_similarity_matrix(global_hist_1: PolarHistogram, global_hist_2: PolarHistogram):
    similarity_matrix = []    # the similarity matrix between global histogram 1 and 2
    for i in range(0, len(global_hist_1.bins)):
        for j in range(0, len(global_hist_2.bins)):
            l_ij = 1 - chi_square_test(global_hist_1.bins[i], global_hist_2.bins[j])
            if j == 0:
                similarity_matrix.append([l_ij])
            else:
                similarity_matrix[i].append(l_ij)

    return similarity_matrix


def analyse_similarity_matrix(similarity_matrix: List[list]) -> float:
    """
    This method analyze a similarity_matrix of two global histograms and
    returns the probability that the two global histograms are similar
    TODO currently using Cui2014Method without RANSAC algorithm
    :param similarity_matrix: a similarity matrix of two global histograms, computed in make_similarity_matrix method
    :return: a probability [0;1] of the similarity between them, with 0 as no similarity at all and with 1 as perfect
    similarity
    """
    nb_rows = len(similarity_matrix)
    nb_col = len(similarity_matrix[0])
    sim_mat_np = np.array(similarity_matrix)
    max_columns = sim_mat_np.max(axis=0)
    max_rows = sim_mat_np.max(axis=1)

    matching_pairs_count = 0

    for i in range(0, min(nb_rows, nb_col)):
        diag_i = similarity_matrix[i][i]
        max_c = max_columns[i]
        max_r = max_rows[i]
        # if the i-th diagonal item is the max value for both the i-th column and row, then it's a matching pair
        if diag_i == max_c and diag_i == max_r:
            matching_pairs_count += 1

    score = matching_pairs_count / min(nb_rows, nb_col)

    return score


def compare_histograms(global_hist_1: PolarHistogram, global_hist_2: PolarHistogram) -> float:
    return analyse_similarity_matrix(make_similarity_matrix(global_hist_1, global_hist_2))


def make_polar_histogram(points: np.ndarray, name: str = ""):
    polar_histogram = PolarHistogram(points_coord=points, name=name)
    return polar_histogram


if __name__ == "__main__":
    R = np.array([[1, 1], [3, 2], [3, 1], [2, 3], [2, 2], [4, 4], [5, 7], [2, 5]])
    T = np.array([[2, 1], [4, 2], [4, 1], [3, 3], [3, 2], [5, 4], [6, 7], [3, 5]])
    Z = np.array([[0, 0], [14, 22], [5, 8], [17, 23], [13, 0.2], [0.5, 0.4], [2, 9], [9, 9]])
    W = np.array([[1, 1], [3, 2], [3, 1], [2, 3], [2, 2], [4, 4], [5, 7], [3, 4]])

    histogram1 = make_polar_histogram(R)
    histogram2 = make_polar_histogram(T)
    histogram3 = make_polar_histogram(Z)
    histogram4 = make_polar_histogram(W)

    sim_matrix_1 = np.array(make_similarity_matrix(histogram1, histogram2))
    sim_matrix_2 = np.array(make_similarity_matrix(histogram1, histogram1))
    sim_matrix_3 = np.array(make_similarity_matrix(histogram1, histogram3))

    proba_sim_1 = compare_histograms(histogram1, histogram2)
    proba_sim_2 = compare_histograms(histogram1, histogram1)
    proba_sim_3 = compare_histograms(histogram1, histogram3)
    proba_sim_4 = compare_histograms(histogram1, histogram4)

    # image = np.zeros(shape=[800, 600, 3], dtype=np.uint8)
    #
    # col = 0
    # for p in R:
    #     cv2.circle(image, (p[0] * 100, p[1] * 100), 5, (0, 0, 255), -1)
    #     cv2.circle(image, (p[0] * 100, p[1] * 100), int(np.round(histogram.radius * 100)), (0, 255 - col, 255), 1)
    #     col += 15
    # cv2.imshow("radius with points", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
