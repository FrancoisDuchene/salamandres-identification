import threading


import fingerprint as fgp


def threadfun_sim_matrix_build(hist_1_bin_i: dict, hist_2_bin_j: dict):
    return fgp.chi_square_test(hist_1_bin_i, hist_2_bin_j)

