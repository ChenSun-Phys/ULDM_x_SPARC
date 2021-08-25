""" This file automates the emcee runs with fixed m slices.
The same can be done manually to fix m in the param card
...Module author: Chen Sun
...Year: 2021
...Email: chensun@mail.tau.ac.il

"""
import sys
import os
import getopt
import mcmc
import warnings
import numpy as np
import random
import multiprocessing as mp
# import emcee
# from contextlib import closing

if __name__ == '__main__':
    # read runtime options
    warnings.filterwarnings('error', 'overflow encountered')
    warnings.filterwarnings('error', 'invalid value encountered')
    argv = sys.argv[1:]
    help_msg = "python %s -N <number_of_steps> -o <output_folder> -L <likelihood_directory> -i <param_file> -w <number_of_walkers> -m <'logm_min logm_max number_of_slicing'> -G <'galaxies'>" % (
        sys.argv[0])
    try:
        opts, args = getopt.getopt(argv, 'hN:o:L:i:w:m:G:')
    except getopt.GetoptError:
        raise Exception(help_msg)
    flgN = False
    flgo = False
    flgL = False
    flgi = False
    flgw = False
    flgm = False
    flgG = False
    for opt, arg in opts:
        if opt == '-h':
            raise Exception(help_msg)
        elif opt == '-N':
            chainslength = arg
            flgN = True
        elif opt == '-o':
            dir_output = arg
            flgo = True
        elif opt == '-L':
            dir_lkl = arg
            flgL = True
        elif opt == '-i':
            path_of_param = arg
            flgi = True
        elif opt == '-w':
            number_of_walkers = int(arg)
            flgw = True
        elif opt == '-m':
            logmrange = np.asarray(arg.split()).astype(np.int)
            logm_min = logmrange[0]
            logm_max = logmrange[1]
            logm_num = logmrange[2]
            flgm = True
        elif opt == '-G':
            galaxies = np.asarray(arg.split())
            flgG = True

    if not (flgN and flgo and flgL and flgi and flgw and flgm and flgG):
        raise Exception(help_msg)

    # modify param card: m, galaxies
    lst_logm = np.linspace(logm_min, logm_max, logm_num)
    # for logm in lst_logm:

    def run(logm):
        for galaxy in galaxies:
            path, file_name = os.path.split(path_of_param)
            path_of_new_param = os.path.join(
                path, 'sample_mslicing_gen_%d.param' % random.randint(1, 10000000000))
            with open(path_of_param, 'r') as f_old:
                with open(path_of_new_param, 'w') as f_new:
                    for line_old in f_old:
                        if line_old == 'logm = _TBD_\n':
                            line_new = 'logm = [%.1f, %.1f, %.1f, 0]\n' % (
                                logm, logm, logm)
                            f_new.write(line_new)
                        elif line_old == "use_galaxies = ('_TBD_')\n":
                            line_new = "use_galaxies = ('%s')\n" % (galaxy)
                            f_new.write(line_new)
                        else:
                            f_new.write(line_old)

            # modify output folder
            dir_root, dir_old = os.path.split(dir_output)
            dir_new = dir_old + '_%s' % (galaxy) + '_ma_%.1f' % (np.abs(logm))
            # chain_name = 'chain_%s' % (galaxy) + '_ma_%.1f.h5' % (np.abs(logm))
            dir_output_new = (os.path.join(dir_root, dir_new))
            print('results being saved under %s' % dir_output_new)

            mcmc.main(chainslength=chainslength,
                      dir_output=dir_output_new,
                      dir_lkl=dir_lkl,
                      path_of_param=path_of_new_param,  # path_of_param,
                      number_of_walkers=number_of_walkers)

            # clean up
            os.remove(path_of_new_param)
    pool = mp.Pool()
    pool.map(run, lst_logm)
    pool.close()
    pool.join()

  # (p0, nwalkers, ndim, lnprob, backend, use_multithreading) = mcmc.main(chainslength=chainslength,
  #                                                                        dir_output=dir_output_new,
  #                                                                        dir_lkl=dir_lkl,
  #                                                                        path_of_param=path_of_new_param,  # path_of_param,
  #                                                                        number_of_walkers=number_of_walkers)
  #  if use_multithreading:
  #       from multiprocessing import Pool
  #       with closing(Pool()) as pool:
  #                   # initialize sampler
  #           sampler = emcee.EnsembleSampler(nwalkers,
  #                                           ndim,
  #                                           lnprob,
  #                                           backend=backend,
  #                                           pool=pool)
  #           sampler.reset()

  #           try:
  #               result = sampler.run_mcmc(
  #                   p0, chainslength, progress=True)

  #           except Warning:
  #               print('p0=%s, chainslength=%s' % (p0, chainslength))
  #               raise

  #           pool.terminate()
  #   else:
  #       # initialize sampler
  #       sampler = emcee.EnsembleSampler(nwalkers,
  #                                       ndim,
  #                                       lnprob,
  #                                       backend=backend)
  #       sampler.reset()

  #       result = sampler.run_mcmc(p0, chainslength, progress=True)

  #   print("Mean acceptance fraction: {0:.3f}".format(
  #       np.mean(sampler.acceptance_fraction)))
