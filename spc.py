#########################################
# This is the code related to loading and
# processing the SPARC data set
#########################################
import os
import numpy as np
import model as mdl


class Galaxy(object):
    """Class of galaxies. Each galaxy of the SPARC dataset
    is one instance of this class

    """
    # the class of

    def __init__(self):
        self.name = ''
        self.D = 0.  # [Mpc]
        self.R = np.array([])  # [kpc]
        self.Vobs = np.array([])  # [km/s]
        self.dVobs = np.array([])  # [km/s]
        self.Vgas = np.array([])  # [km/s]
        self.Vdisk = np.array([])  # [km/s]
        self.Vbul = np.array([])  # [km/s]
        self.SBdisk = np.array([])  # [solLum/pc2]
        self.SBbul = np.array([])  # [solLum/pc2]

        # extended meta data from Lelli2016.c
        self.Vflat = 0.  # [km/s]
        self.dVflat = 0.  # [km/s]

    def sanity_check(self):
        # TODO: add some sanity checks
        return

    def get_M(self):
        return max(self.Vobs * self.R / mdl._c / mdl._Msun_over_kpc_Mpl2)

    # def get_Msol(self):
    #     sqrt_ctilde_Phi = 1. # 1.1 to 0.9
    #     return 2.1 * sqrt_ctilde_Phi *
    def get_Vmax(self):
        return max(self.Vobs)


def readSPARC(path, verbose=0):
    """Function that load the SPARC dataset

    :param path: SPARC dataset location
    :param verbose: parameter to control the verbosity

    """
    # path = '../Data/SPARC.txt'
    # cwd = os.getcwd()
    # path = os.path.join(cwd, path)
    if verbose > 3:
        print('---%s: loading %s' % (os.path.basename(__file__), path))

    # the assumed baryonic contribution to the rotation velocity, depends on
    # mass-to-light ratio UpsStar. We take typical value from
    # http://iopscience.iop.org/article/10.3847/0004-6256/152/6/157/pdf
    # UpsStar = 0.5

    # initialize containers
    res = np.array([])
    Ngal = 0
    my_ex = ''  # the memory of last gal's name
    with open(path, 'r') as fid:
        for i in range(25):
            next(fid)
        for line in fid:
            words = line.split()
            current_gal_name = words[0]
            # determine if this is a new gal
            if current_gal_name != my_ex:
                # save the last gal
                if Ngal > 0:
                    res = np.append(res, Gal)
                Gal = Galaxy()
                Ngal += 1
                my_ex = words[0]  # reset memory

            Gal.name = words[0]
            Gal.D = float(words[1])
            Gal.R = np.append(Gal.R, [float(words[2])])
            Gal.Vobs = np.append(Gal.Vobs, [float(words[3])])
            Gal.dVobs = np.append(Gal.dVobs, [float(words[4])])
            Gal.Vgas = np.append(Gal.Vgas,  [float(words[5])])
            Gal.Vdisk = np.append(Gal.Vdisk,  [float(words[6])])
            Gal.Vbul = np.append(Gal.Vbul,  [float(words[7])])
            Gal.SBdisk = np.append(Gal.SBdisk,  [float(words[8])])
            Gal.SBbul = np.append(Gal.SBbul,  [float(words[9])])
    res = np.append(res, Gal)
    return res


def readSPARC_ext(lst_galaxies, path, verbose=0):
    """function that updates lst_galaxies with the
    extended information (scalar value for each gal,) such as
    Hubble type, inclination, vflat, etc.

    :param lst_galaxies: the list of galaxies to be updated
    :param path: SPARC_Lelli2016c.txt dataset location
    :param verbose: parameter to control the verbosity
    :returns: galaxies after update
    :rtype: numpy array

    """
    # path ='../Data/SPARC_Lelli2016c.txt'
    # cwd = os.getcwd()
    # path = os.path.join(cwd, path)
    if verbose > 3:
        print('---%s: loading %s' % (os.path.basename(__file__), path))

    # CS: for now I'm only updating vflat,
    # since that's the one quantity we are going to use
    line_counter = 0
    with open(path, 'r') as fid:
        for i in range(98):
            next(fid)
        for line in fid:
            words = line.split()
            current_gal_name = words[0]
            gal = lst_galaxies[line_counter]
            if gal.name == current_gal_name:
                gal.Vflat = float(words[15])
                gal.dVflat = float(words[16])
            else:
                # double-check the orders of the two files are the same
                print('%s' % (gal.name))
            line_counter += 1


def findGalaxyByName(name, lst_galaxies):
    """find the galaxy instance by its name

    :param name: string of the galaxy to be found
    :param lst_galaxies: list of galxy instances
    :returns: the galaxy that bears this name
    :rtype: Galaxy instance

    """
    is_found = False
    for gal in lst_galaxies:
        if gal.name == name:
            is_found = True
            break

    if is_found:
        res = gal
    else:
        res = -1
    return res
