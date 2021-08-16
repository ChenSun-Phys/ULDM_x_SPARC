"""This is a module that performs an even-grid scan
...Module author: Chen Sun
...Year: 2021
...Email: chensun@mail.tau.ac.il

"""

import numpy as np
import pickle
import mcmc
import chi2
from tqdm import tqdm


class Result(object):
    """The container that stores the results for a single m
    """

    def __init__(self, log_m):
        self.m = 10**log_m
        self.chi2_arr = None
        self.M_arr = None
        self.sane = False

    def check_chi_min(self):
        chi2_min = min(self.chi2_arr)
        if chi2_min == 0:
            self.sane = True


class Results(object):
    """The container for the array of results
    """

    def __init__(self, arr):
        self.storage = arr


def analyze(res_arr, sigma_lvl):
    for result in res_arr:
        if not result.sane:
            continue

        m = result.m
        M_arr = result.M_arr
        chi2_arr = result.chi2_arr
        M_contours = []
        for sigma in sigma_lvl:
            M_contour = np.interp(sigma, chi2_arr, M_arr)
            M_contours.append(M_contour)

        # update result
        result.sigma_lvl = sigma_lvl
        result.M_contours = M_contours


class Scanner():

    def scan(self,
             gal,
             ups_low=0,
             ups_high=5,
             num_of_ups=20,
             log_M_low=5,
             log_M_high=14,
             num_of_log_M=30,
             log_m_low=-25,
             log_m_high=-19,
             num_of_log_m=20,
             sig_levels=[1, 2, 3, 4, 5, 10],
             dir_name='results'):

        ups_disk_arr = np.linspace(ups_low, ups_high, num_of_ups)
        ups_bulg_arr = np.linspace(ups_low, ups_high, num_of_ups)
        log_M_arr = np.linspace(log_M_low, log_M_high, num_of_log_M)
        log_m_arr = np.linspace(log_m_low, log_m_high, num_of_log_m)

        ups_disk_mesh, ups_bulg_mesh = np.meshgrid(ups_disk_arr, ups_bulg_arr)

        ups_disk_flat = ups_disk_mesh.reshape(-1)
        ups_bulg_flat = ups_bulg_mesh.reshape(-1)

        res_arr = []
        for log_m in tqdm(log_m_arr):

            m = 10**log_m
            chi2_arr = np.asarray([1e10]*len(log_M_arr))

            # for each M value, minimizes over Ups
            for j in range(len(log_M_arr)):
                M = 10**log_M_arr[j]
                for i in range(len(ups_disk_flat)):
                    ups_disk = ups_disk_flat[i]
                    ups_bulg = ups_bulg_flat[i]
                    chi2_val = chi2.chi2_single_gal_overshooting(
                        m=m, M=M, ups_disk=ups_disk, ups_bulg=ups_bulg, gal=gal)
                    chi2_arr[j] = min(chi2_arr[j], chi2_val)

            result = Result(log_m)
            result.chi2_arr = chi2_arr
            result.M_arr = 10**log_M_arr

            # sanity check
            result.check_chi_min()

            # save the galaxy as well
            result.gal = gal

            # save it
            res_arr.append(result)

        # find M contours
        analyze(res_arr, sig_levels)

        # pickle the result
        mcmc.dir_init('../%s' % dir_name)
        # uid = np.random.randint(1e10)
        # path = '../%s/result-%d.dat' % (dir_name, uid)
        path = '../%s/result-%s.dat' % (dir_name, gal.name)

        with open(path, 'w') as f:
            # pickle.dump(result, f)
            results = Results(res_arr)
            pickle.dump(results, f)
        return


def lower_array(y1_arr, y2_arr):
    """function to find the lower bound of two curves represented by two arrays. Note: y1_arr and y2_arr need to have the same length. Otherwise, it simply returns y1_arr

    : param y1_arr: the first array
    : param y2_arr: the second array

    """
    # special use: for the first comparison, if y1_arr is [], then y2_arr is passed to it
    if len(y1_arr) == 0:
        y1_arr = np.copy(y2_arr)

    if len(y1_arr) != len(y2_arr):
        # print(len(y2_arr))
        # print(len(y1_arr))
        # raise Exception(
        #     'The two arrays need to be of the same length. Quitting')
        return np.asarray(y1_arr)
    y_arr = []
    for i in range(len(y1_arr)):
        y_arr.append(min(y1_arr[i], y2_arr[i]))
    return np.asarray(y_arr)


def lower_array(x1_arr, y1_arr, x2_arr, y2_arr):
    """function to find the lower bound of two curves(x1, y1), and (x2, y2) represented by two arrays.

    """
    # special use: for the first comparison, if y1_arr is [], then y2_arr is passed to it
    x1_arr = np.asarray(x1_arr)
    x2_arr = np.asarray(x2_arr)
    y1_arr = np.asarray(y1_arr)
    y2_arr = np.asarray(y2_arr)

    x_arr = np.concatenate((x1_arr, x2_arr))
    x_arr = np.unique(x_arr)
    x_arr = np.sort(x_arr)

    y_arr = []

    UNDEF = 1.e100
    for x in x_arr:
        try:
            y1 = np.interp(x, x1_arr, y1_arr, right=UNDEF, left=UNDEF)
        except:
            y1 = None
        try:
            y2 = np.interp(x, x2_arr, y2_arr, right=UNDEF, left=UNDEF)
        except:
            y2 = None

        flg_y1_valid = False
        flg_y2_valid = False

        if y1 is not None:
            flg_y1_valid = True
        if y2 is not None:
            flg_y2_valid = True

        if not flg_y1_valid and not flg_y2_valid:
            # print(x_arr)
            print(x1_arr)
            print(y1_arr)
            print(x2_arr)
            print(y2_arr)
            print(x)
            raise Exception('Error in interpolating.')

        if flg_y1_valid and not flg_y2_valid:
            y = y1
        if not flg_y1_valid and flg_y2_valid:
            y = y2
        if flg_y1_valid and flg_y2_valid:
            y = min(y1, y2)

        y_arr.append(y)
    return x_arr, np.asarray(y_arr)

    # if len(y1_arr) == 0:
    #     y1_arr = np.copy(y2_arr)

    # if len(y1_arr) != len(y2_arr):
    #     # print(len(y2_arr))
    #     # print(len(y1_arr))
    #     # raise Exception(
    #     #     'The two arrays need to be of the same length. Quitting')
    #     return np.asarray(y1_arr)
    # y_arr = []
    # for i in range(len(y1_arr)):
    #     y_arr.append(min(y1_arr[i], y2_arr[i]))
    # return np.asarray(y_arr)
