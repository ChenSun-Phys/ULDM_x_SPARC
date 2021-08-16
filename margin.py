""" This module marginalizes all but two parameters
and perform log likelihood ratio test
Module author: Chen Sun
Year: 2021
Email: chensun@mail.tau.ac.il

"""
import h5py
import glob
import re
import os
import numpy as np
import model


class Result(object):
    """class of result"""

    def __init__(self, galaxy):
        self.name = galaxy
        self.m = []
        self.Mupper = []
        self.Mlower = []
        self.bestfit = None
        self.path = []


def load_chain(path):
    return


def binning(path, ind_x, ind_y, num_of_bins=20, verbose=0):
    # load file. input: path. output: raw_chain
    f = h5py.File(path, 'r')
    raw_chain = f.get('mcmc')
    if verbose > 2:
        print('--margin.py:file %s has keys: %s' % (path, f.keys()))
        print('--margin.py:raw_chain has keys: %s' % (raw_chain.keys()))
        print("--margin.py:log_prob has shape: %s" %
              (raw_chain['log_prob'].shape,))
        print("--margin.py:chains have shape: %s" %
              (raw_chain['chain'].shape,))

    # first parsing. input: raw_chain. output: flat chain and chi2.
    chain = np.array(raw_chain['chain'])
    dim_of_param = (chain.shape)[-1]
    chain = chain.reshape(-1, dim_of_param)  # flatten the walkers

    chi2 = np.array(raw_chain['log_prob']) * (-2.)
    chi2 = chi2.reshape(-1)  # flatten the walkers

    # then make blocks. input: flat chain. output: edges of the blocks
    chain_x = chain[:, ind_x]
    # chain_neg_x = chain_x[np.where(chain_x < 0)]
    # TODO: verify the above is not needed
    _, edges_x = np.histogram(chain_x, bins=num_of_bins)

    chain_y = chain[:, ind_y]
    # chain_neg_y = chain_y[np.where(chain_y < 0)]
    # TODO: verify
    _, edges_y = np.histogram(chain_y, bins=num_of_bins)

    # now in each block marginalizing all but x and y by finding chi2_min
    # input: chi2, edges_x, edges_y
    # output: point_count (scanning density), and chi2_min of the block
    point_count = []
    chi2_min = []
    for i in range(len(edges_y)-1):
        for j in range(len(edges_x)-1):
            # select those chi2
            chi2_block = chi2[np.where((chain_y > edges_y[i]) & (chain_y < edges_y[i+1])
                                       & (chain_x > edges_x[j]) & (chain_x < edges_x[j+1]))]
            point_count.append(len(chi2_block))
            if len(chi2_block) > 0:
                chi2_min.append(min(chi2_block))
            else:
                chi2_min.append(np.nan)  # TODO: this might cause problems
    chi2_min = np.array(chi2_min)

    # output meshgrid
    block_y = (edges_y[:-1] + edges_y[1:])/2.
    block_x = (edges_x[:-1] + edges_x[1:])/2.
    mesh_y, mesh_x = np.meshgrid(block_y, block_x, indexing='ij')

    return (mesh_x, mesh_y, chi2_min, point_count)


def post_1d(path, ind_x, num_of_bins=20, verbose=0, flg_save_bf=False):
    """ Generate one dimensional posterior. 

    :param path: the path of the chain files
    :param ind_x: the index of the parameter to be checked
    :param num_of_bins: number of bins
    :param verbose: control of the verbosity
    :param flg_save_bf: whether to output the bestfit

    """
    # load file. input: path. output: raw_chain
    f = h5py.File(path, 'r')
    raw_chain = f.get('mcmc')
    if verbose > 2:
        print('--margin.py:file %s has keys: %s' % (path, f.keys()))
        print('--margin.py:raw_chain has keys: %s' % (raw_chain.keys()))
        print("--margin.py:log_prob has shape: %s" %
              (raw_chain['log_prob'].shape,))
        print("--margin.py:chains have shape: %s" %
              (raw_chain['chain'].shape,))

    # first parsing. input: raw_chain. output: flat chain and chi2.
    chain = np.array(raw_chain['chain'])
    dim_of_param = (chain.shape)[-1]
    chain = chain.reshape(-1, dim_of_param)  # flatten the walkers

    chi2 = np.array(raw_chain['log_prob']) * (-2.)
    chi2 = chi2.reshape(-1)  # flatten the walkers

    # then make blocks. input: flat chain. output: edges of the blocks
    # TODO: get rid of the binning here.
    chain_x = chain[:, ind_x]
    _, edges_x = np.histogram(chain_x, bins=num_of_bins)

    # now in each block marginalizing all but x by finding chi2_min
    # input: chi2, edges_x
    # output: point_count (scanning density), and chi2_min of the block
    point_count = []
    chi2_min = []
    if flg_save_bf:
        bestfit = np.empty((0, dim_of_param))
    for j in range(len(edges_x)-1):
        # select those chi2, projected onto x intervals
        chi2_block = chi2[np.where(
            (chain_x > edges_x[j]) & (chain_x < edges_x[j+1]))]
        point_count.append(len(chi2_block))
        if len(chi2_block) > 0:
            chi2_min_this_block = min(chi2_block)
            chi2_min.append(chi2_min_this_block)
            if flg_save_bf:
                index_of_bf = np.where(chi2 == chi2_min_this_block)
                bf_of_this_block = chain[index_of_bf[0][0]]
                bestfit = np.concatenate((bestfit, [bf_of_this_block]))
                # print('index_of_bf=%s' % index_of_bf)
                # print('bf_of_this_block=%s' % bf_of_this_block)
                # print('chi2_of_bf=%s' % chi2_min_this_block)
        else:
            chi2_min.append(np.nan)  # TODO: this might cause problems
            if flg_save_bf:
                bestfit = np.concatenate((bestfit, [[np.nan]*dim_of_param]))
    chi2_min = np.array(chi2_min)

    # output meshgrid
    block_x = (edges_x[:-1] + edges_x[1:])/2.
    if flg_save_bf:
        return (block_x, chi2_min, point_count, bestfit)
    else:
        return (block_x, chi2_min, point_count)


def SH_bound(m, gal):
    """The soliton mass predicted by the soliton halo relation

    :param m: soliton mass [eV]
    :param gal: galaxy instance

    """

    Vmax = max(gal.Vobs)
    E_over_M = (Vmax / model._c)**2
    Msol = 4.3 * np.sqrt(np.abs(E_over_M)) * model._Mpl2_over_eV_Msun/m
    return Msol


def m_slicing_single_gal(gal, n_sig, dim_of_param, verbose=1.,):
    lst_M95_upper = []
    lst_M95_lower = []
    lst_bestfit = np.empty((0, dim_of_param))
    for i in range(len(gal.m)):
        m = gal.m[i]
        path = gal.path[i]
        path_new = os.path.join(gal.path[i], 'chain_1.h5')

        # get 1D posterior
        (lst_M, lst_chi2, point_count, lst_bf) = post_1d(path_new,
                                                         0,
                                                         num_of_bins=30,
                                                         verbose=verbose,
                                                         flg_save_bf=True)

        # finding contours in the poterior
        lst_chi2 = lst_chi2 - min(lst_chi2)

        # account for upper and lower
        index_of_chi2min = np.where(lst_chi2 == 0.)
        index_of_chi2min = np.asarray(list(index_of_chi2min)).reshape(-1)

        if len(index_of_chi2min) == 1:
            index_of_chi2min = np.squeeze(index_of_chi2min.reshape(-1))
        else:
            # just get the last zero
            index_of_chi2min = index_of_chi2min[-1]

        M95_upper = np.interp(
            n_sig**2, lst_chi2[index_of_chi2min:], lst_M[index_of_chi2min:], left=np.nan, right=np.nan)
        M95_lower = np.interp(
            n_sig**2., lst_chi2[index_of_chi2min::-1], lst_M[index_of_chi2min::-1], left=np.nan, right=np.nan)
        if verbose > 1:
            print("-margin.py: 95%% CL M upper=10^%.2f Msun" % (M95_upper))
            print("-margin.py: 95%% CL M lower=10^%.2f Msun" % (M95_lower))

        # find the block that contains the global chi2_min
        index_bf = np.interp(n_sig**2,
                             lst_chi2,
                             range(len(lst_M)),
                             left=np.nan,
                             right=np.nan)

        # output
        # lst_m_slice = np.append(lst_m_slice, 10**logm)
        lst_M95_upper = np.append(lst_M95_upper, 10**M95_upper)
        lst_M95_lower = np.append(lst_M95_lower, 10**M95_lower)

        if not np.isnan(index_bf):
            lst_bestfit = np.concatenate(
                (lst_bestfit, [lst_bf[int(round(index_bf))]]))
        else:
            lst_bestfit = np.concatenate(
                (lst_bestfit, [[np.nan]*dim_of_param]))
        if verbose > 1:
            print('\n\n')
    gal.Mupper = lst_M95_upper
    gal.Mlower = lst_M95_lower
    gal.bestfit = lst_bestfit
    # return (lst_m_slice, lst_M95_upper, lst_M95_lower, lst_bestfit)


def m_slicing(runid, n_sig, dim_of_param, verbose=1., multiprocessing=False, path='/a/home/cc/students/physics/chensun/Code/BEC_dynamics/chains/'):
    """Does the m slicing chi2 analysis

    :param runid: the id of the run
    :param n_sig: number of sigmas
    :param dim_of_param: the number of parameters
    :param verbose: level of verbosity

    """
    dct_gal = collect_galaxies(runid=runid, path=path)

    if multiprocessing:
        # FIXME: doesn't work due to pickling error
        import multiprocessing as mp

        pool = mp.Pool()

        def wrapper(gal):
            m_slicing_single_gal(
                gal, n_sig=n_sig, dim_of_param=dim_of_param, verbose=verbose)
        pool.map(wrapper, dct_gal.values())
        pool.close()
        pool.join()

    else:
        for galname, gal in dct_gal.items():
            if verbose > 0:
                print(galname)
            # print(gal.Mupper)
            m_slicing_single_gal(
                gal, n_sig=n_sig, dim_of_param=dim_of_param, verbose=verbose)
            # print(gal.Mupper)
            # print('\n\n')
    return dct_gal


def collect_galaxies(runid, verbose=1., path='/a/home/cc/students/physics/chensun/Code/BEC_dynamics/chains/'):
    """Parse the result and find all the galaxies that were studied

    :param runid: run id 
    :param dct_parse: the dictionary that will contain the result
    :param verbose: verbosity
    :param path: path to the chains

    """
    dct_parse = {}
    for folder in glob.glob(os.path.join(path, 'run_%s*' % runid)):
        search = re.search(
            "_([a-zA-Z\-]+\d*)_ma_(\d*.\d*)$", folder)
        if search:
            galaxy = search.group(1)
            m = 10**(-1.*float(search.group(2)))
            if verbose > 1:
                print('-margin.py: galaxy=%s  m=%.1f' % (galaxy, m))

            try:
                dct_parse['%s' % galaxy]
            except KeyError:
                dct_parse['%s' % galaxy] = Result('galaxy')
            dct_parse['%s' % galaxy].m.append(m)
            dct_parse['%s' % galaxy].path.append(folder)

    return dct_parse
