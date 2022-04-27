import unittest
import numpy as np
import math

from fingerprint import PolarHistogram, dist, make_similarity_matrix, analyse_similarity_matrix, compare_histograms


class PolarHistogramTestCase(unittest.TestCase):

    def test_angles(self):
        X = np.array([[0,0],
                      [0,1],
                      [1,1],
                      [1,0],
                      [1,-1],
                      [0,-1],
                      [-1,-1],
                      [-1,0],
                      [-1,1],
                      [4,6]])
        histogram = PolarHistogram(X)

        # tests par rapport à l'origine
        self.assertEqual(histogram.angle_two_points(X[0], X[1]), math.radians(90))
        self.assertEqual(histogram.angle_two_points(X[0], X[2]), math.radians(45))
        self.assertEqual(histogram.angle_two_points(X[0], X[3]), math.radians(0))
        self.assertEqual(histogram.angle_two_points(X[0], X[4]), math.radians(-45))
        self.assertEqual(histogram.angle_two_points(X[0], X[5]), math.radians(-90))
        self.assertEqual(histogram.angle_two_points(X[0], X[6]), math.radians(-135))
        self.assertEqual(histogram.angle_two_points(X[0], X[7]), math.radians(180))
        self.assertEqual(histogram.angle_two_points(X[0], X[8]), math.radians(135))

        self.assertEqual(histogram.angle_two_points(X[1], X[2]), math.radians(0))
        self.assertEqual(histogram.angle_two_points(X[1], X[3]), math.radians(-45))
        self.assertEqual(histogram.angle_two_points(X[2], X[0]), math.radians(-135))
        self.assertEqual(histogram.angle_two_points(X[2], X[6]), math.radians(-135))
        self.assertAlmostEqual(histogram.angle_two_points(X[2], X[9]), math.radians(59.036), places=3)

        self.assertEqual(histogram.angle_two_points(X[1], X[7]), math.radians(-135))
        self.assertEqual(histogram.angle_two_points(X[7], X[1]), math.radians(45))

        self.assertAlmostEqual(histogram.angle_two_points((2, 1), (3.5, 1.25)), math.radians(9.45), places=3)

    def test_distance(self):
        X = np.array([[0, 0],
                      [0, 1],
                      [1, 1],
                      [1, 0],
                      [1, -1],
                      [0, -1],
                      [-1, -1],
                      [-1, 0],
                      [-1, 1]])
        self.assertEqual(dist(X[0],X[1]), 1)
        self.assertEqual(dist(X[1],X[3]), np.sqrt(2))
        self.assertEqual(dist(X[1],X[5]), 2)


    def test_radius(self):
        X1 = np.array([[0, 0],
                      [0, 1],
                      [1, 1]])
        X2 = np.array([[0, 0],
                       [0, 1],
                       [1, 1],
                       [1, 0]])
        X3 = np.array([[2, 1],
                       [3, 1.5],
                       [4, 1]])
        X4 = np.array([[2, 1],
                       [3, 1.5],
                       [4, 1],
                       [7, 7]])
        X5 = np.array([[2, 1],
                       [3, 1.5],
                       [4, 1],
                       [3, 4],
                       [6, 3],
                       [7, 7]])

        self.assertAlmostEqual(PolarHistogram.histogram_radius(X1), 0.1896785)
        self.assertAlmostEqual(PolarHistogram.histogram_radius(X2), 0.2133883)
        self.assertAlmostEqual(PolarHistogram.histogram_radius(X3), 0.35300566479164913)
        self.assertAlmostEqual(PolarHistogram.histogram_radius(X4), 1.597203552517096)
        self.assertAlmostEqual(PolarHistogram.histogram_radius(X5), 2.3883275205163246)

    def test_number_neighbors(self):
        X = np.array([[2, 1],
                       [3, 1.5],
                       [4, 1],
                       [3, 4],
                       [6, 3],
                       [7, 7]])
        histogram = PolarHistogram(X)
        neighbors: dict = histogram.neighbors
        self.assertIsInstance(neighbors, dict)
        self.assertIsInstance(neighbors[0], list)
        self.assertIsInstance(neighbors[1], list)
        self.assertIsInstance(neighbors[2], list)

        self.assertNotIn(3, neighbors)
        self.assertNotIn(4, neighbors)
        self.assertNotIn(5, neighbors)

        self.assertAlmostEqual(histogram.radius, 2.3883275205163246)

        n = 0
        self.assertNotIn(0, neighbors[n])
        self.assertIn(1, neighbors[n])
        self.assertIn(2, neighbors[n])
        self.assertNotIn(3, neighbors[n])
        self.assertNotIn(4, neighbors[n])
        self.assertNotIn(5, neighbors[n])

        n = 1
        self.assertIn(0, neighbors[n])
        self.assertNotIn(1, neighbors[n])
        self.assertIn(2, neighbors[n])
        self.assertNotIn(3, neighbors[n])
        self.assertNotIn(4, neighbors[n])
        self.assertNotIn(5, neighbors[n])

        n = 2
        self.assertIn(0, neighbors[n])
        self.assertIn(1, neighbors[n])
        self.assertNotIn(2, neighbors[n])
        self.assertNotIn(3, neighbors[n])
        self.assertNotIn(4, neighbors[n])
        self.assertNotIn(5, neighbors[n])

        # test of get_neighbors function
        for j in range(0, len(X)):
            self.assertIsInstance(histogram.get_neighbors(j), list)

        n = 0
        neighbors = histogram.get_neighbors(n)
        xn = [1, 2]
        self.assertCountEqual(neighbors, xn)

        n = 1
        neighbors = histogram.get_neighbors(n)
        xn = [0, 2]
        self.assertCountEqual(neighbors, xn)

        n = 2
        neighbors = histogram.get_neighbors(n)
        xn = [0, 1]
        self.assertCountEqual(neighbors, xn)

        n = 3
        neighbors = histogram.get_neighbors(n)
        self.assertEqual(len(neighbors), 0)

        n = 4
        neighbors = histogram.get_neighbors(n)
        self.assertEqual(len(neighbors), 0)

        n = 5
        neighbors = histogram.get_neighbors(n)
        self.assertEqual(len(neighbors), 0)

    def test_centroids(self):
        X = np.array([[2, 1],
                      [3, 1.5],
                      [4, 1],
                      [3, 4],
                      [6, 3],
                      [7, 7]])
        histogram = PolarHistogram(X)
        centroids = histogram.centroids

        self.assertIsInstance(centroids, dict)
        self.assertEqual(centroids[0][0], 2)
        self.assertEqual(centroids[0][1], 1)
        self.assertEqual(centroids[1][0], 3)
        self.assertEqual(centroids[1][1], 1.5)
        self.assertEqual(centroids[2][0], 4)
        self.assertEqual(centroids[2][1], 1)
        self.assertEqual(centroids[3][0], 3)
        self.assertEqual(centroids[3][1], 4)
        self.assertEqual(centroids[4][0], 6)
        self.assertEqual(centroids[4][1], 3)
        self.assertEqual(centroids[5][0], 7)
        self.assertEqual(centroids[5][1], 7)

    def test_check_point(self):
        # reference point is always on (0,0)
        histogram = PolarHistogram(np.array([[0, 0]]))

        #
        #
        #   \   |   /
        #    \3 | 2/
        #   4 \ | / 1
        # ------O------
        #   5 / | \ 8
        #    /  |  \
        #   / 6 | 7 \
        #
        #


        # Situation 1) first quartier
        self.assertFalse(histogram.check_point(0, 0, 0, radius=3))
        self.assertTrue(histogram.check_point(0.1, 0.01, 0, radius=3))
        self.assertFalse(histogram.check_point(1, -1, 0, radius=3))
        self.assertFalse(histogram.check_point(-1, -1, 0, radius=3))
        self.assertFalse(histogram.check_point(-1, 1, 0, radius=3))

        # A point at the edge of a quadrant is considered to be part of a bin if it is the ending edge,
        # but not the starting edge
        self.assertTrue(histogram.check_point(1, 1, 0, radius=3))
        self.assertFalse(histogram.check_point(1, 0, 0, radius=3))
        self.assertFalse(histogram.check_point(0, 1, 0, radius=3))
        self.assertTrue(histogram.check_point(0.5, 0.2, 0, radius=3))

        # Situation 2)
        self.assertFalse(histogram.check_point(1, 1, math.radians(45), radius=3))
        self.assertFalse(histogram.check_point(1, -1, math.radians(45), radius=3))
        self.assertFalse(histogram.check_point(-1, -1, math.radians(45), radius=3))
        self.assertFalse(histogram.check_point(-1, 1, math.radians(45), radius=3))
        self.assertFalse(histogram.check_point(1, 0, math.radians(45), radius=3))
        self.assertTrue(histogram.check_point(0, 1, math.radians(45), radius=3))

        # Situation 4)
        self.assertFalse(histogram.check_point(1, 1, math.radians(135), radius=3))
        self.assertFalse(histogram.check_point(1, -1, math.radians(135), radius=3))
        self.assertFalse(histogram.check_point(-1, -1, math.radians(135), radius=3))
        self.assertFalse(histogram.check_point(-1, 1, math.radians(135), radius=3))
        self.assertFalse(histogram.check_point(1, 0, math.radians(135), radius=3))
        self.assertFalse(histogram.check_point(0, 1, math.radians(135), radius=3))
        self.assertTrue(histogram.check_point(-1, 0, math.radians(135), radius=3))

        # Situation 5)
        self.assertFalse(histogram.check_point(1, 1, math.radians(-180), radius=3))
        self.assertFalse(histogram.check_point(1, -1, math.radians(-180), radius=3))
        self.assertTrue(histogram.check_point(-1, -1, math.radians(-180), radius=3))
        self.assertFalse(histogram.check_point(-1, 1, math.radians(-180), radius=3))
        self.assertFalse(histogram.check_point(1, 0, math.radians(-180), radius=3))
        self.assertFalse(histogram.check_point(0, 1, math.radians(-180), radius=3))
        self.assertFalse(histogram.check_point(-1, 0, math.radians(-180), radius=3))

        # Situation 7)
        self.assertFalse(histogram.check_point(1, 1, math.radians(-90), radius=3))
        self.assertTrue(histogram.check_point(1, -1, math.radians(-90), radius=3))
        self.assertFalse(histogram.check_point(-1, -1, math.radians(-90), radius=3))
        self.assertFalse(histogram.check_point(-1, 1, math.radians(-90), radius=3))
        self.assertFalse(histogram.check_point(1, 0, math.radians(-90), radius=3))
        self.assertFalse(histogram.check_point(0, 1, math.radians(-90), radius=3))
        self.assertFalse(histogram.check_point(0, -1, math.radians(-90), radius=3))

        # Situation 8)
        self.assertFalse(histogram.check_point(1, 1, math.radians(-45), radius=3))
        self.assertFalse(histogram.check_point(1, -1, math.radians(-45), radius=3))
        self.assertFalse(histogram.check_point(-1, -1, math.radians(-45), radius=3))
        self.assertFalse(histogram.check_point(-1, 1, math.radians(-45), radius=3))
        self.assertTrue(histogram.check_point(1, 0, math.radians(-45), radius=3))
        self.assertFalse(histogram.check_point(0, 1, math.radians(-45), radius=3))

        # Situation X1) starting angle : 170, ending angle : 215 (-145)
        self.assertTrue(histogram.check_point(-1, 0.01, math.radians(170), radius=3))
        self.assertTrue(histogram.check_point(-1, -0.01, math.radians(170), radius=3))
        # Situation X2) starting angle : 345 (-15), ending angle 390 (30)
        self.assertTrue(histogram.check_point(1, -0.1, math.radians(345), radius=3))
        self.assertTrue(histogram.check_point(1, 0.1, math.radians(345), radius=3))
        # Situation X3) starting angle : 365 (5), ending angle 410 (50),
        # this can happen when the centroid is placed weirdly. However, the ending angle will never be higher than 4 pi
        self.assertTrue(histogram.check_point(1, 0.5, math.radians(360), radius=3))
        self.assertTrue(histogram.check_point(0.5, 0.5, math.radians(360), radius=3))
        self.assertTrue(histogram.check_point(1, 0.1, math.radians(360), radius=3))


    def test_compute_bins(self):
        X = np.array([[2, 1],
                       [3, 1.5],
                       [4, 1],
                       [3, 4],
                       [6, 3],
                       [7, 7]])
        histogram_1 = PolarHistogram(X)
        # r = 2.388
        # r_int = 1.194

        # Angle Regions
        #
        #   \   |   /
        #    \3 | 2/
        #   4 \ | / 1
        # ------O------
        #   5 / | \ 8
        #    /  |  \
        #   / 6 | 7 \

        bins_1 = histogram_1.bins
        # bins[0] -> bins for point 0 (in histogram.points_coord[0])
        # bins[0][0] -> bins for point 0 with the angle region 0 (which is from 0 to pi/4)
        # bins[0][0][0] -> counter for the internal region of point 0 at angle region 0
        # test for first point (a)
        self.assertEqual(bins_1[0][0][0], 0)
        self.assertEqual(bins_1[0][0][1], 0)
        self.assertEqual(bins_1[0][1][0], 0)
        self.assertEqual(bins_1[0][1][1], 0)
        self.assertEqual(bins_1[0][2][0], 0)
        self.assertEqual(bins_1[0][2][1], 0)
        self.assertEqual(bins_1[0][3][0], 0)
        self.assertEqual(bins_1[0][3][1], 0)
        self.assertEqual(bins_1[0][4][0], 0)
        self.assertEqual(bins_1[0][4][1], 0)
        self.assertEqual(bins_1[0][5][0], 0)
        self.assertEqual(bins_1[0][5][1], 0)
        self.assertEqual(bins_1[0][6][0], 0)
        self.assertEqual(bins_1[0][6][1], 0)
        self.assertEqual(bins_1[0][7][0], 0)
        self.assertEqual(bins_1[0][7][1], 0)
        # test for second point (b)
        self.assertEqual(bins_1[1][0][0], 0)
        self.assertEqual(bins_1[1][0][1], 0)
        self.assertEqual(bins_1[1][1][0], 0)
        self.assertEqual(bins_1[1][1][1], 0)
        self.assertEqual(bins_1[1][2][0], 0)
        self.assertEqual(bins_1[1][2][1], 0)
        self.assertEqual(bins_1[1][3][0], 0)
        self.assertEqual(bins_1[1][3][1], 0)
        self.assertEqual(bins_1[1][4][0], 0)
        self.assertEqual(bins_1[1][4][1], 0)
        self.assertEqual(bins_1[1][5][0], 0)
        self.assertEqual(bins_1[1][5][1], 0)
        self.assertEqual(bins_1[1][6][0], 1)
        self.assertEqual(bins_1[1][6][1], 0)
        self.assertEqual(bins_1[1][7][0], 0)
        self.assertEqual(bins_1[1][7][1], 0)
        # test for third point (c)
        self.assertEqual(bins_1[2][0][0], 0)
        self.assertEqual(bins_1[2][0][1], 1)
        self.assertEqual(bins_1[2][1][0], 0)
        self.assertEqual(bins_1[2][1][1], 0)
        self.assertEqual(bins_1[2][2][0], 0)
        self.assertEqual(bins_1[2][2][1], 0)
        self.assertEqual(bins_1[2][3][0], 0)
        self.assertEqual(bins_1[2][3][1], 0)
        self.assertEqual(bins_1[2][4][0], 0)
        self.assertEqual(bins_1[2][4][1], 0)
        self.assertEqual(bins_1[2][5][0], 0)
        self.assertEqual(bins_1[2][5][1], 0)
        self.assertEqual(bins_1[2][6][0], 0)
        self.assertEqual(bins_1[2][6][1], 0)
        self.assertEqual(bins_1[2][7][0], 1)
        self.assertEqual(bins_1[2][7][1], 0)
        # test for fourth point (d)
        self.assertEqual(bins_1[3][0][0], 0)
        self.assertEqual(bins_1[3][0][1], 0)
        self.assertEqual(bins_1[3][1][0], 0)
        self.assertEqual(bins_1[3][1][1], 0)
        self.assertEqual(bins_1[3][2][0], 0)
        self.assertEqual(bins_1[3][2][1], 0)
        self.assertEqual(bins_1[3][3][0], 0)
        self.assertEqual(bins_1[3][3][1], 0)
        self.assertEqual(bins_1[3][4][0], 0)
        self.assertEqual(bins_1[3][4][1], 0)
        self.assertEqual(bins_1[3][5][0], 0)
        self.assertEqual(bins_1[3][5][1], 0)
        self.assertEqual(bins_1[3][6][0], 0)
        self.assertEqual(bins_1[3][6][1], 0)
        self.assertEqual(bins_1[3][7][0], 0)
        self.assertEqual(bins_1[3][7][1], 0)

    def test_histogram_comparisons(self):
        # Reference image
        X1 = np.array([[2, 1],
                      [3, 1.5],
                      [4, 1],
                      [3, 4],
                      [6, 3],
                      [7, 7]])
        # Translated on X (each point)
        X2 = np.array([[3, 1],
                       [4, 1.5],
                       [5, 1],
                       [4, 4],
                       [7, 3],
                       [8, 7]])
        # Rotate of -90 degrees
        X3 = np.array([[1, 6],
                       [1.5, 5],
                       [1, 4],
                       [4, 5],
                       [3, 2],
                       [7, 1]])
        # Reference image minus the farthest point
        X4 = np.array([[2, 1],
                       [3, 1.5],
                       [4, 1],
                       [3, 4],
                       [6, 3]])
        # Reference image minus one point
        X5 = np.array([[3, 1.5],
                       [4, 1],
                       [3, 4],
                       [6, 3],
                       [7, 7]])
        # Translation X (with variation of delta +- 0.2)
        X6 = np.array([[2.9, 1],
                       [4, 1.5],
                       [5.2, 1],
                       [3.8, 4],
                       [7, 3],
                       [8.1, 7]])
        # Translation X (with variation of delta +- 0.4)
        X7 = np.array([[2.6, 1],
                       [4, 1.5],
                       [5.4, 1],
                       [3.6, 4],
                       [7, 3],
                       [8.2, 7]])
        # Translation XY (with variation of delta +- 0.2)
        X8 = np.array([[2.9, 2.9],
                       [4, 2.5],
                       [5.2, 2.2],
                       [3.8, 4.8],
                       [7, 4],
                       [8.1, 8.1]])
        histogram_1 = PolarHistogram(X1)
        histogram_2 = PolarHistogram(X2)
        histogram_3 = PolarHistogram(X3)
        histogram_4 = PolarHistogram(X4)
        histogram_5 = PolarHistogram(X5)
        histogram_6 = PolarHistogram(X6)
        histogram_7 = PolarHistogram(X7)
        histogram_8 = PolarHistogram(X8)

        sim_matrix_h1_h1 = make_similarity_matrix(histogram_1, histogram_1)

        for i in range(0, len(sim_matrix_h1_h1)):
            self.assertEqual(sim_matrix_h1_h1[i][i], 1)

        self.assertEqual(compare_histograms(histogram_1, histogram_2), 1)
        self.assertEqual(compare_histograms(histogram_1, histogram_3), 1)
        self.assertEqual(compare_histograms(histogram_2, histogram_3), 1)
        # We remove points
        self.assertEqual(compare_histograms(histogram_1, histogram_4), 1)
        self.assertEqual(compare_histograms(histogram_2, histogram_4), 1)
        self.assertEqual(compare_histograms(histogram_3, histogram_4), 1)

        self.assertEqual(compare_histograms(histogram_1, histogram_5), 0.6)
        self.assertEqual(compare_histograms(histogram_2, histogram_5), 0.6)
        self.assertEqual(compare_histograms(histogram_3, histogram_5), 0.4)

        self.assertEqual(compare_histograms(histogram_1, histogram_6), 1)
        self.assertGreaterEqual(compare_histograms(histogram_1, histogram_7), 0.66)
        self.assertGreaterEqual(compare_histograms(histogram_6, histogram_7), 0.83)
        self.assertGreaterEqual(compare_histograms(histogram_1, histogram_8), 0.33)
        self.assertGreaterEqual(compare_histograms(histogram_6, histogram_8), 0.33)
        self.assertGreaterEqual(compare_histograms(histogram_7, histogram_8), 0.33)


if __name__ == '__main__':
    unittest.main()
