""" This module analyzes the chains from the emcee runs and generates corner plots using corner.py.
Module author: Chen Sun
Year: 2020
Email: chensun@mail.tau.ac.il

"""
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from datetime import datetime
except:
    pass
import os
import errno
import numpy as np
import emcee
from emcee.autocorr import AutocorrError
import corner
# import h5py
import sys
import getopt

from mcmc import fill_mcmc_parameters, update_specific_vals


def pltpath(dir):
    path = os.path.join(dir, 'plots')

    run_name = str(dir).rstrip('/')
    run_name = run_name.split('/')[-1]

    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    return os.path.join(path,
                        # 'corner_' + datetime.now().strftime("%Y%m%d_%H%M%S%f") + '.pdf')
                        'corner_' + run_name + '.pdf')


if __name__ == '__main__':
    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv, 'hi:')
    except getopt.GetoptError:
        raise Exception('python %s -i <folder_of_chains>' %
                        (sys.argv[0]))
    flgi = False
    for opt, arg in opts:
        if opt == '-h':
            raise Exception('python %s -i <folder_of_chains>' %
                            (sys.argv[0]))
        elif opt == '-i':
            directory = arg
            flgi = True
    if not flgi:
        raise Exception('python %s -i <folder_of_chains>' %
                        (sys.argv[0]))

    for filename in os.listdir(directory):
        if filename.endswith(".h5"):
            path = os.path.join(directory, filename)

            reader = emcee.backends.HDFBackend(path, read_only=True)
            # tau = reader.get_autocorr_time()
            try:
                tau = reader.get_autocorr_time()
                print('auto correlation time = %s' % tau)
            except AutocorrError as e:
                # this is the case the chain is shorter than 50*(autocorr time)
                print('%s' % e)
                # tau = [410., 100., 140, 140]
                tau = e.tau
                print('setting correlation time to the current estimate.')

            # use auto-correlation time to estimate burnin here
            # works only for long chains
            burnin = int(2*np.max(tau))
            thin = int(0.5*np.min(tau))
            samples = reader.get_chain(
                discard=burnin, flat=True, thin=thin)
            print("burn-in: {0}".format(burnin))
            print("thin: {0}".format(thin))
            print("flat chain shape: {0}".format(samples.shape))
            try:
                all_samples = np.append(all_samples, samples, axis=0)
            except:
                all_samples = samples
        else:
            continue

    # load log.param
    params, keys, keys_fixed = fill_mcmc_parameters(
        os.path.join(directory, 'log.param'))
    params, keys = update_specific_vals(params, keys)

    # test data authenticity
    if len(keys) != len(samples[0]):
        raise Exception(
            'log.param and h5 files are not consistent. \
            Data is compromised. Quit analyzing.')

    # compute mean
    dim_of_param = len(samples[0])
    mean = np.mean(samples, axis=0)
    print('mean = %s' % mean)

    # corner plot
    plt.figure(0)
    labels = keys
    # labels = [r"$\Omega_\Lambda$", r"$h$", r"$\log\ m_a$", r"$\log\ g_a$"]
    figure = corner.corner(samples, labels=labels, quantiles=[
                           0.16, 0.5, 0.84], show_titles=True,
                           title_kwargs={"fontsize": 12})
    axes = np.array(figure.axes).reshape((dim_of_param, dim_of_param))

    plt.savefig(pltpath(directory))
