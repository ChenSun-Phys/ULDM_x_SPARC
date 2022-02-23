"""This is a module that performs an bisection scan
...Module author: Chen Sun
...Year: 2022
...Email: chensun@mail.tau.ac.il

"""
import numpy as np
import pickle
import mcmc
import chi2
from scan import Result, Results, Scanner
from tqdm import tqdm
from scipy.interpolate import interp1d


class Scanner_bis():
    """ A scanner class that utilizes bisection instead of even grid. 

    """

    def scan(self,
             gal,
             log_M_low=5,
             log_M_high=14,
             num_of_log_M=None,
             log_m_low=-25,
             log_m_high=-19,
             num_of_log_m=20,
             sig_levels=[1, 2, 3, 4, 5, 10],
             dir_name='results',
             tolerance=1.e-2,
             max_step=100,
             debug=False,
             log_m_to_include=[]):

        # log_M_arr = np.linspace(log_M_low, log_M_high, num_of_log_M)
        log_m_arr_raw = np.linspace(log_m_low, log_m_high, num_of_log_m)
        log_m_arr = np.concatenate((log_m_arr_raw, log_m_to_include))
        log_m_arr = np.unique(log_m_arr)

        # ups_disk_mesh, ups_bulg_mesh = np.meshgrid(ups_disk_arr, ups_bulg_arr)

        # ups_disk_flat = ups_disk_mesh.reshape(-1)
        # ups_bulg_flat = ups_bulg_mesh.reshape(-1)

        res_arr = []
        for log_m in tqdm(log_m_arr):

            m = 10**log_m
            chi2val = 0
            result = Result(log_m)
            result.chi2_arr = sig_levels
            result.M_arr = []
            result.sane = True

            for sig in sig_levels:
                # prepare for bisection
                low = log_M_low
                high = log_M_high
                logM = low
                counter = 0
                while (np.abs(chi2val - sig**2) > tolerance):
                    logM = 0.5*(high + low)

                    chi2val = chi2.chi2_single_gal_overshooting(
                        m=m, M=10**logM, ups_disk=0, ups_bulg=0, gal=gal)

                    chi2lowval = chi2.chi2_single_gal_overshooting(
                        m=m, M=10**low, ups_disk=0, ups_bulg=0, gal=gal)

                    chi2highval = chi2.chi2_single_gal_overshooting(
                        m=m, M=10**high, ups_disk=0, ups_bulg=0, gal=gal)

                    if chi2val > sig**2:
                        if debug:
                            print('too high')
                            print("logM=%f, chi2val=%f" % (logM, chi2val))
                            print("low=%f, chi2val=%f" % (low, chi2lowval))
                            print("high=%f, chi2val=%f" % (high, chi2highval))
                        high = logM
                    else:
                        if debug:
                            print('too low')
                            print("logM=%f, chi2val=%f" % (logM, chi2val))
                            print("low=%f, chi2val=%f" % (low, chi2lowval))
                            print("high=%f, chi2val=%f" % (high, chi2highval))
                        low = logM
                    if counter > max_step:
                        print("fail to converge within %d steps!" % counter)
                        result.sane = False
                        break
                    counter += 1
                if debug:
                    print("it took %d steps to converge!\n\n\n" % counter)
                result.M_arr.append(10**logM)

            # save the galaxy as well
            result.gal = gal

            # save it
            res_arr.append(result)

        # pickle the result
        mcmc.dir_init('../../%s' % dir_name)
        # uid = np.random.randint(1e10)
        # path = '../%s/result-%d.dat' % (dir_name, uid)
        path = '../../%s/result-%s.dat' % (dir_name, gal.name)

        with open(path, 'w') as f:
            # pickle.dump(result, f)
            results = Results(res_arr)
            pickle.dump(results, f)
        return
