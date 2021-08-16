########################################
# this is the module to define the chi2
# and related functions
########################################
import numpy as np
import model


def chi2_single_gal(m, M, c, Rs, ups_disk, ups_bulg, gal, flg_Vtot=False, DM_profile="NFW"):
    """chi2 for a fixed theory point (m, M, c, Rs; ups_disk, ups_bulg). Runs over a single galaxy, gal
up
    :param m: scalar mass [eV]
    :param M: soliton mass [Msun]
    :param c: concentration param of the NFW profile, and delc for Burkert
    :param Rs: critical radius of the NFW profile
    :param ups_disk: disk surface brightness
    :param ups_bulg: bulge surface brightness
    :param gal: a Galaxy instance
    :param flg_Vtot: flg to signal returning of the total rot velocity
                 defined as Vtot**2 = Vb**2 + VNFW**2 + Vsol**2 [km/s]
    :param DM_profile: flag to choose between NFW and Burkert. (Default: NFW)
    :returns: chi2 value

    """
    chi2 = 0
    if flg_Vtot:
        Vtot = np.array([])
    for i, r in enumerate(gal.R):
        # treat the i-th bin of the rot curve
        #
        # TODO: move this part out to a dedicated function
        # V thory due to DM
        if DM_profile == "NFW":
            M_enclosed = model.M_NFW(r, Rs, c) + model.M_sol(r, m, M)
        elif DM_profile == "Burkert":
            M_enclosed = model.M_Burkert(
                r, delc=c, Rs=Rs) + model.M_sol(r, m, M)
        else:
            raise Exception(
                "Only NFW and Burkert are implemented at the moment.")
        VDM2 = model._G_Msun_over_kpc * model._c**2 * (M_enclosed/1.) * (1./r)
        # combine DM with the baryon mass model (from SB data)
        Vb2 = (ups_bulg*np.abs(gal.Vbul[i])*gal.Vbul[i]
               + ups_disk*np.abs(gal.Vdisk[i])*gal.Vdisk[i]
               + np.abs(gal.Vgas[i])*gal.Vgas[i]
               )
        Vth2 = VDM2 + Vb2
        if Vth2 > 0:
            Vth = np.sqrt(Vth2)
        else:
            Vth = 0.
        # ODOT

        Vobs = gal.Vobs[i]
        dVobs = gal.dVobs[i]

        # compute chi2 for this bin
        chi2 += (Vth - Vobs)**2/dVobs**2

        # construct Vtot for visual/sanity checks
        # construct Vtot
        if flg_Vtot:
            Vtot = np.append(Vtot, Vth)
    if flg_Vtot:
        return (chi2, Vtot)
    else:
        return chi2


def chi2_single_gal_overshooting(m, M, ups_disk, ups_bulg, gal, flg_Vtot=False):
    """chi2 for a fixed theory point (m, M; ups_disk, ups_bulg)
    run over a single galaxy, gal. It only has soliton core in it and only counts
    the overshooting of the data

    :param m: scalar mass [eV]
    :param M: soliton mass [Msun]
    :param ups_disk: disk surface brightness
    :param ups_bulg: bulge surface brightness
    :param gal: a Galaxy instance
    :param flg_Vtot: flg to signal returning of the total rot velocity
                 defined as Vtot**2 = Vb**2 + VNFW**2 + Vsol**2 [km/s]
    :returns: chi2 value

    """
    chi2 = 0
    if flg_Vtot:
        Vtot = np.array([])
    for i, r in enumerate(gal.R):
        # treat the i-th bin of the rot curve
        #
        # TODO: move this part out to a dedicated function
        # V thory due to DM
        M_enclosed = model.M_sol(r, m, M)  # now only with soliton, no NFW
        VDM2 = model._G_Msun_over_kpc * model._c**2 * (M_enclosed/1.) * (1./r)
        # combine DM with the baryon mass model (from SB data)
        Vb2 = (ups_bulg*np.abs(gal.Vbul[i])*gal.Vbul[i]
               + ups_disk*np.abs(gal.Vdisk[i])*gal.Vdisk[i]
               + np.abs(gal.Vgas[i])*gal.Vgas[i]
               )
        Vth2 = VDM2 + Vb2
        if Vth2 > 0:
            Vth = np.sqrt(Vth2)
        else:
            Vth = 0.
        # ODOT

        Vobs = gal.Vobs[i]
        dVobs = gal.dVobs[i]

        # compute chi2 for this bin
        if Vth > Vobs:
            chi2 += (Vth - Vobs)**2/dVobs**2

        # construct Vtot for visual/sanity checks
        # construct Vtot
        if flg_Vtot:
            Vtot = np.append(Vtot, Vth)
    if flg_Vtot:
        return (chi2, Vtot)
    else:
        return chi2


def chi2_gals(x, keys, keys_fixed, data, params, verbose=0):
    """This function mainly does the parsing of x from lnprob(x) and feeds each galaxy with parameters relevant to it.

    :param x: The theory point whose chi2 is being computed. n-tuple
    :param keys: The description of each entry of x. list of length n
    :param keys_fixed: The description of each fixed entry
    :param data: the galaxies to be run over
    :param params: param card
    :returns: chi2 value

    """
    chi2_tot = 0
    chi2_comp = []  # the list to save all the chi2 components
    for gal in data:
        # for each galaxy, we pack a dict
        param_for_gal = {}

        # deal with free params
        for i, key in enumerate(keys):
            words = key.split()
            if (len(words) == 2) and (words[1] == gal.name):
                # dealing with galaxy-specific variables
                param_for_gal[words[0]] = x[i]
            if len(words) == 1:
                # dealing with universal variables
                param_for_gal[words[0]] = x[i]

        # now deal with fixed params
        for i, key in enumerate(keys_fixed):
            param_for_gal[key] = params[key+' fixed']

        # from now on only read param_for_gal param card
        m = 10**param_for_gal['logm']
        try:
            DM_profile = params['DM_profile']
        except KeyError:
            if verbose > 0:
                print('chi2.py:DM_profile is not specified, default to NFW')
            DM_profile = "NFW"
        try:
            Rs = param_for_gal['Rs']
        except:
            if verbose > 5:
                print('-----param_for_gal= %s' % param_for_gal)
                print('-----Rs is not set, so soliton is the only component')
            if DM_profile != "Soliton":
                raise Exception(
                    "You specify the DM profile to be either NFW or Burkert, but didn't supply Rs")

        flg_catch_c = False
        flg_catch_logc = False

        try:
            # in NFW use c for concentration
            c = param_for_gal['c']
            flg_catch_c = True
        except:
            pass

        try:
            # in Burkert profile try to catch logc for log(del_c)
            c = 10**param_for_gal['logc']
            flg_catch_logc = True
        except:
            pass

        # I'm being sloppy here by postponing the raise until here when it is actually used
        # let's call it a just-in-time warning :-)
        if flg_catch_c and flg_catch_logc:
            raise Exception('You gave both c and logc. Only one can be kept.')

        if (not flg_catch_c) and (not flg_catch_logc) and (DM_profile != "Soliton"):
            raise Exception(
                "You specify the DM profile to be either NFW or Burkert, but didn't supply c or logc")

        try:
            ups_disk = param_for_gal['ups_disk']
        except KeyError:
            raise Exception('no ups_disk is specified.')

        try:
            ups_bulg = param_for_gal['ups_bulg']
        except KeyError:
            raise Exception('no ups_bulg is specified')

        # try:
        #     h = param_for_gal['h']
        # except KeyError:
        #     raise Exception('h is not specified.')

        # now deal with derived params, if any, such as M
        try:
            M = 10**param_for_gal['logM']
        except KeyError:
            raise Exception('logM is not specified.')

        # sum over
        if (DM_profile == "NFW") or (DM_profile == "Burkert"):
            chi2_i = chi2_single_gal(m=m,
                                     M=M,
                                     c=c,
                                     Rs=Rs,
                                     ups_disk=ups_disk,
                                     ups_bulg=ups_bulg,
                                     gal=gal,
                                     DM_profile=DM_profile)

        elif DM_profile == "Soliton":
            chi2_i = chi2_single_gal_overshooting(m=m,
                                                  M=M,
                                                  ups_disk=ups_disk,
                                                  ups_bulg=ups_bulg,
                                                  gal=gal,
                                                  flg_Vtot=False)
        chi2_tot += chi2_i
        # chi2_comp.append(chi2_i)
        chi2_comp.append(M)  # use chi2_comp to contain M temporarily
    chi2_comp.insert(0, chi2_tot)
    if verbose > 5:
        print('-----chi2.py:chi2_comp=%s' % chi2_comp)
    # return chi2_tot
    return np.array(chi2_comp)
