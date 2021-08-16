""" this module calls emcee and use it to propose a smart grid 
to quickly find the (local) mininum of -log(lkl). 
...Module author: Chen Sun
...Year: 2020, 2021
...Email: chensun@mail.tau.ac.il

"""
import os
import errno
from collections import OrderedDict as od
from contextlib import closing
import numpy as np
import emcee

import spc
import chi2


def dir_init(path):
    """Create a dictionary at path, if it doesn't exist yet. Otherwise pass.

    :param path: path to be created.
    :returns: None

    """

    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    return


def fill_mcmc_parameters(path):
    """Function to store all the input parameters from path to a dictionary

    :param path: path to the .param file
    :returns: (param, keys, fixed_keys)
    :rtype: (od, list, list)

    """

    res = od()
    keys = []
    fixed_keys = []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith("#"):
                pass
            elif (line.startswith('\n')) or (line.startswith('\r')):
                pass
            else:
                words = line.split("=")
                key = (words[0]).strip()
                try:
                    res[key] = float(words[1])
                except:
                    # print line, words, key
                    res[key] = (words[1]).strip()
                    # not a number, start parsing
                    if res[key][0] == '(' and res[key][-1] == ')':
                        res[key] = eval(res[key])
                        # if it's a tuple of one element, python converts it to scalar
                        # so we add a layer of array to prevent this
                        # yet flatten it if it's not a tuple of one.
                        res[key] = np.array([res[key]]).reshape(-1)
                    elif res[key][0] == '[' and res[key][-1] == ']':
                        # make sure the string is safe to eval()
                        res[key] = eval(res[key])
                        if res[key][3] != 0.:
                            res[key+' mean'] = res[key][0]
                            res[key+' low'] = res[key][1]
                            res[key+' up'] = res[key][2]
                            res[key+' sig'] = res[key][3]
                            keys.append(str(key))
                        else:
                            res[key+' fixed'] = res[key][0]
                            fixed_keys.append(str(key))
                    elif res[key] == 'TRUE' or res[key] == 'True' or res[key] == 'true' or res[key] == 'T' or res[key] == 'yes' or res[key] == 'Y' or res[key] == 'Yes' or res[key] == 'YES':
                        res[key] = True

                    elif res[key] == 'FALSE' or res[key] == 'False' or res[key] == 'false' or res[key] == 'F' or res[key] == 'NO' or res[key] == 'No' or res[key] == 'no' or res[key] == 'N':
                        res[key] = False
    return (res, keys, fixed_keys)


def update_specific_vals(params, keys):
    """Update the param dictionary to deal with the galaxy-specific variables.
    Galaxy-spcific variables are those mcmc parameters specific to each galaxy,
    such as Rs and c. These are specified in the param card by spec_val option.
    The nomenclature we adopt here is 'varname galname mean/up/low/sig'. For
    example, params['Rs UGC04325 up'] gives the upper bound of Rs for UGC04325.

    Warning: the content of params is loosely maintained: the old content is not deleted. What matters is the keys (and keys_fixed). In this sense, params can well be replaced with a regular dictionary instead of od().

    :param params: The dictionary that contains all the input params
    :returns: (an updated param variable, updated keys)
    :rtype: (od, list)

    """
    try:
        params['spec_val']
    except KeyError:
        return (params, keys)

    for gal_name in params['use_galaxies']:
        for spec_val in params['spec_val']:
            # print('gal_name=%s' % gal_name)
            if spec_val in keys:
                params[spec_val+' '+gal_name +
                       ' mean'] = params[spec_val+' mean']
                params[spec_val+' '+gal_name+' up'] = params[spec_val+' up']
                params[spec_val+' '+gal_name+' low'] = params[spec_val+' low']
                params[spec_val+' '+gal_name+' sig'] = params[spec_val+' sig']
                keys.append(spec_val+' '+gal_name)
    # remove the old keys
    for spec_val in params['spec_val']:
        if spec_val in keys:
            keys.remove(spec_val)
    return (params, keys)


def is_Out_of_Range(x, keys, params):
    """
    Returns a Boolean type indicating whether the current
    point is within the range

    Parameters
    ----------
    x : tuple
        the current point in the hyperspace to be checked
    keys: list
        each correspond to a dimension in the hyperspace,
        i.e. all the variables to be scanned
    """
    res = False

    for i in range(len(x)):
        if x[i] > params[keys[i]+' up'] or x[i] < params[keys[i]+' low']:
            res = True
            break
    return res


##########################
# initialize
##########################
def main(chainslength,
         dir_output,
         dir_lkl,
         path_of_param,
         number_of_walkers):
    """The main routine

    :param chainslength: chain length
    :param dir_output: output directory
    :param dir_lkl: likelihood directory
    :param path_of_param: location of the param card
    :param number_of_walkers: number of walkers

    """

    # init the dir
    dir_init(dir_output)

    # check if there's a preexisting param file from a previous run
    if os.path.exists(os.path.join(dir_output, 'log.param')):
        path_of_param = os.path.join(dir_output, 'log.param')
        # get the mcmc params from existing file
        params, keys, keys_fixed = fill_mcmc_parameters(
            path_of_param)
        params, keys = update_specific_vals(params, keys)
        # the keys order has changed, and we are going to stick to this order to feed x
    else:
        # get the mcmc params
        params, keys, keys_fixed = fill_mcmc_parameters(
            path_of_param)
        params, keys = update_specific_vals(params, keys)
        # save the input file only after the params are filled, i.e. legit
        from shutil import copyfile
        copyfile(path_of_param, os.path.join(dir_output, 'log.param'))

    # fill up defaults
    try:
        params['debug']
    except KeyError:
        params['debug'] = False

    if params['debug']:
        debug = True
    else:
        debug = False

    try:
        params['verbose']
    except:
        params['verbose'] = 0

    verbose = params['verbose']

    if verbose > 4:
        print('----params=%s' % params)

   #  # determine if soliton-halo relation is imposed
   #  flg_sol_halo_relation = False
   #  try:
   #      params['logM']
   # except KeyError:
   #      # this means logM is not specified
   #      flg_sol_halo_relation = True
   #      # TODO: implement soliton-halo relation in sampling
   #      # DONE

    # determing if multiple galaxies are fitted together
    # later need to extend Rs, c, and M for each galaxies
    if len(params['use_galaxies']) > 1:
        flg_multigal_run = True
    else:
        flg_multigal_run = False

    # multi-threading
    try:
        params['use_multithreading']
    except:
        params['use_multithreading'] = 'True'
        # print('hit')

    # note that in fill_mcmc_parameters: eval(res[key]) will convert
    # 'True' To boolean True automatically
    if params['use_multithreading'] is True:
        use_multithreading = True
        print('running in multiprocessing mode.')
    else:
        use_multithreading = False
        if verbose > 0:
            print('running in linear mode as use_multithreading is set to %s' %
                  (params['use_multithreading']))


##########################
# Sanity checks
##########################
    try:
        from multiprocessing import Pool
    except:
        if use_multithreading:
            raise Exception(
                'You asked for multiprocessing, but it is not properly setup on this machine')


##########################
# load up likelihoods
# that are read from a file
##########################
    # load SPARC
    # TODO: specify path
    # DONE
    path = os.path.join(dir_lkl, params['SPARC_lkl'])
    data = spc.readSPARC(path, verbose)

    # update with extra info
    # TODO: change func
    # DONE
    path = os.path.join(dir_lkl, params['SPARC_aux'])
    spc.readSPARC_ext(data, path, verbose)
    if verbose > 2:
        print('--len(data)=%d' % (len(data)))

    # TODO: select the subgroup of galaxies based on param card
    # DONE
    data_sub = np.array([])
    for gal_name in params['use_galaxies']:
        gal = spc.findGalaxyByName(gal_name, data)
        data_sub = np.append(data_sub, [gal])

    if verbose > 2:
        print('--%d galaxies are selected: %s' %
              (len(data_sub), [gal.name for gal in data_sub]))


##########################
# emcee related deployment
##########################


    def lnprob(x):
        """Total log-likelihood. Runtime wrapper.
        This is the only thing that emcee cares about.
        This default is \chi^2 = \infty.
        Note that all evaluation of model.py and chi2.py modules
        should be wrapped with is_Out_of_Range(), since they do not
        know the boundary. The user is responsible to set sensible
        boundaries in the param card, which is enforced by is_Out_of_Range().

        :param x: the current sampling point
        :returns: ln(likelihood)

        """
        # loglkl = -np.inf
        # now we save all the chi2 components
        loglkl = [-np.inf]
        for _ in data_sub:
            loglkl.append(-np.inf)
        if not is_Out_of_Range(x, keys, params):  # to avoid overflow
            # loglkl = -1./2 * \
            #     chi2.chi2_gals(x, keys, keys_fixed, data_sub, params)
            loglkl = -1./2 * chi2.chi2_gals(
                x, keys, keys_fixed, data_sub, params, verbose)
            # print('keys=%s' % keys)
            # print('x=%s' % x)
            # print('chi2=%s' % (-2.*loglkl))
            # print('loglkl=%s' % (loglkl))

        return loglkl

#
# first step guess
#
    p0mean = []
    for key in keys:
        p0mean.append(params[key+' mean'])
    if verbose > 0:
        print('keys=%s' % keys)
        print('p0mean=%s' % p0mean)
        print('keys_fixed=%s' % keys_fixed)
        print('fixed at=%s' % [params[key+' fixed'] for key in keys_fixed])

    # for verification
    p0low = []
    p0up = []
    for key in keys:
        p0low.append(params[key+' low'])
        p0up.append(params[key+' up'])

    # initial one sigma
    p0sigma = []
    for key in keys:
        p0sigma.append(params[key+' sig'])

    ndim = len(p0mean)
    nwalkers = number_of_walkers

    # initial point, following Gaussian/ uniform
    p0 = []
    for i in range(len(p0mean)):
        # FIXME
        # p0component = np.random.normal(p0mean[i], p0sigma[i], nwalkers)
        p0component = np.random.uniform(
            low=p0low[i], high=p0up[i], size=(nwalkers,))
        p0.append(p0component)
    p0 = np.array(p0).T

    # Set up the backend
    counter = 0
    for filename in os.listdir(dir_output):
        if filename.endswith(".h5"):
            counter += 1
    filename = "chain_%s.h5" % (counter + 1)
    path = os.path.join(dir_output, filename)
    backend = emcee.backends.HDFBackend(path)
    backend.reset(nwalkers, ndim)

    # use_multithreading = True

    if use_multithreading:
        if verbose > 3:
            print('---use_multithreading=%s' % use_multithreading)
            print('---using multiprocessing')
        with closing(Pool()) as pool:
            # initialize sampler
            sampler = emcee.EnsembleSampler(nwalkers,
                                            ndim,
                                            lnprob,
                                            backend=backend,
                                            pool=pool)
            sampler.reset()

            try:
                result = sampler.run_mcmc(p0, chainslength, progress=True)

            except Warning:
                print('p0=%s, chainslength=%s' % (p0, chainslength))
                raise

            pool.terminate()
    else:
        print('---use_multithreading=%s' % use_multithreading)
        print('---not using multiprocessing')
        # initialize sampler
        sampler = emcee.EnsembleSampler(nwalkers,
                                        ndim,
                                        lnprob,
                                        backend=backend)
        sampler.reset()

        result = sampler.run_mcmc(p0, chainslength, progress=True)

    print("Mean acceptance fraction: {0:.3f}".format(
        np.mean(sampler.acceptance_fraction)))

    # if verbose > 3:
    #     if use_multithreading:
    #         print('---use_multithreading=%s' % use_multithreading)
    #         print('---using multiprocessing')
    #     else:
    #         print('---use_multithreading=%s' % use_multithreading)
    #         print('---not using multiprocessing')

    # return (p0, nwalkers, ndim, lnprob, backend, use_multithreading)
