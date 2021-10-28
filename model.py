""" This module builds up the mass profile for model A, B, and C, as shown in the paper.
...Module author: Chen Sun
...Year: 2020, 2021
...Email: chensun@mail.tau.ac.il

"""

#######################################
# This is the code related to
# the model part, including
# 1) NFW mass profile
# 2) Burkert mass profile
# 3) soliton mass profile
#######################################
from __future__ import division

import numpy as np
from scipy.integrate import quad
import chi2
import tools as tl

_rhoc_ref = 3.967061568e-11  # [eV^4] rho_crit w/ H0=70 km/s/Mpc
_h_ref = 0.7  # H0/(100 km/s/Mpc)
_M_ref = 1.360577e2  # _rhoc_ref * kpc**3 [Msun]
_Msun_over_kpc_Mpl2 = 4.78556286411784e-17  # Msun/(kpc*Mpl**2) [1]
_c = 2.99792458e5  # speed of light [km/s]
_Msun_over_kpc3_eV4 = 2.917197883135e-13  # Msun/kpc^3/eV^4
_G_Msun_over_kpc = 4.78556056645064e-17  # G*Msun/kpc
_Mpl2_over_eV_Msun = 1.33628703752223e-10  # Mpl^2/eV/Msun
_G_rhocref_kpc2 = 6.50782503239317e-15  # G*rho_crit*kpc^2
_eV4_kpc3_over_Msun = 3.427947e+12  # eV^4*kpc*3/Msun
_eV4_pc3_over_Msun = 3.427947e+3  # eV^4*pc*3/Msun
_tau_U_ = 13.  # [Gyr]
_Mpl2_km2_over_s2_kpc_over_Msun_ = 232501.397985234
_kpc_eV_ = 1.5637e26
_coulomb_log_threadhold_ = np.e

######################
# NFW:
# input param (c, Rs)
# output rho(r), M(r)
######################


def delta_c(c):
    """the multiplier of the critical density that goes into
    eq. 37 of BBBS2018 (1805.00122)

    :param c: the concentration parameter [1]
    :returns: delta_c [1]

    """
    res = 200./3*c**3/(np.log(1.+c) - c/(1.+c))
    return res


def rho_crit(h):
    """returns the critical density given h

    :param h: H/(100 km/s/Mpc)
    :returns: critical density [eV^4]

    """
    res = _rhoc_ref * (h/_h_ref)**2
    return res


def rho_NFW(r, Rs=1., c=0.5, h=_h_ref):
    """The NFW density profile

    :param r: radius [kpc]
    :param Rs: critical radius of NFW [kpc], default: 1
    :param h: Hubble const H/(100 km/s/Mpc) [1], default: 0.7
    :param c: the concentration parameter [1], default: 0.5
    :returns: density of the halo [Msun/kpc**3]

    """

    rhoc = rho_crit(h)
    delc = delta_c(c)
    rhos = rhoc * delc
    res = rhos / (r/Rs * (1 + r/Rs)**2) * _eV4_kpc3_over_Msun
    return res


def M_NFW(r, Rs=1., c=0.5, h=_h_ref):
    """The NFW mass profile

    :param r: radius [kpc]
    :param Rs: critical radius of NFW [kpc], default: 1
    :param c: the concentration parameter [1], default: 0.5
    :param h: Hubble const H/(100 km/s/Mpc) [1], default: 0.7
    :returns: mass within the radius r [Msun]

    """
    delc = delta_c(c)
    try:
        res = 4. * np.pi * delc * \
            (- r/Rs/(1. + r/Rs) + np.log(1 + r/Rs)) * _M_ref
    except RuntimeWarning as e:
        print('r=%e' % r)
        print('Rs=%e' % Rs)
        print('r/Rs=%e' % (r/Rs))
        raise e
    scaling = (h/_h_ref)**2 * (Rs/1)**3
    return res * scaling


######################
# isolated soliton
# input param (m, M)
# output rho(r), M(r)
######################


def lam(m, M):
    """the scaling variable, a.k.a sqrt(chi(0))

    :param m: scalar mass [eV]
    :param M: solition mass [Msun]
    :returns: lambda, the scaling variable [1]

    """
    res = 3.6e-4*(m/(1.e-22))*(M/1.e9)
    return res


def rc(m, M):
    """The half density radius. 1.9 times it is the radius where density
drops by a factor of 10.

    :param m: scalar mass [eV]
    :param M: solition mass [Msun]
    :returns: rc, the radius at which density drops by a half [kpc]

    """

    scaling = lam(m, M)
    res = 8.2e-5 / (m/1.e-22) / scaling
    return res


def rho_sol(r, m, M):
    """Soliton density profile. The form is taken as rho(0)/(1+0.091(x/rc)^2)^8, rho(0) is expressed in terms of M, and rc by \int_0^\infty rho_sol 4pi r^2 dr = 11.5865 rc^3 rho(0)

    :param r: radius [kpc]
    :param m: scalar mass [eV]
    :param M: soliton mass [Msun]
    :returns: density at r [Msun/kpc^3]

    """
    rcval = rc(m, M)
    rho0 = 1./11.5865 * (M/1.) * (1./rcval)**3  # rho(0) [Msun/kpc^3]
    rho_shape = 1./(1. + 0.091 * (r/rcval)**2)**8
    return rho0 * rho_shape


def M_sol(r, m, M):
    """Soliton mass profile. The form is achieved by analytically integrating rho_sol(r) * 4pi*r^2 dr

    :param r: radius [kpc]
    :param m: scalar mass [eV]
    :param M: soliton mass [Msun]
    :returns: enclosed mass within r [Msun]

    """
    rcval = rc(m, M)
    M_shape = 7.37618/(rcval**2 + 0.091*r**2)**7 * (
        0.384872*rcval**11*r**3 + 0.0665595*rcval**9*r**5 + 0.00665084*rcval**7*r**7
        + 0.000390285*rcval**5*r**9 + 0.0000125498*rcval**3*r**11
        + (0.637*rcval**12*r**2 + 0.173901*rcval**10*r**4 + 0.026375*rcval**8*r**6
           + 0.00240012*rcval**6*r**8 + 0.000131047*rcval**4*r**10
           + 3.97508e-6*rcval**2*r**12 + rcval**14 + 5.16761e-8*r**14
           )*np.arctan(0.301662*r/rcval)
        - 0.301662*rcval**13*r + 1.71305e-7*rcval*r**13
    )

    # TODO: check with a numerical integral of rho for typos
    # DONE

    # note that the rc**3 in M_shape cancels with the 1/rc**3 in the M_norm
    # M_norm is actually rho0 * rc^3 instead of rho0, to make the unit consistent
    M_norm = M / 11.5865
    return M_norm * M_shape


###############
# Burkert
###############
def delta_c_Burkert(c):
    """the multiplier of the critical density that goes into
    rho_Burkert

    :param c: the concentration parameter
    :returns: delta_c
    """
    res = -800.*c**3/(-3.*np.log(c**2 + 1) - 6.*np.log(c+1) + 6.*np.arctan(c))
    return res


def rho_Burkert(r, delc, Rs=1., h=_h_ref):
    """The Burkert density profile

    :param r: radius [kpc]
    :param delc: delta_c that goes into the numerator
    :param Rs: critical radius of Burkert [kpc], default: 1
    :param h: Hubble const H/(100 km/s/Mpc) [1], default: 0.7
    :returns: density of the halo [Msun/kpc**3]

    """
    rhoc = rho_crit(h)
    # delc = delta_c_Burkert(c)
    rho0 = rhoc * delc
    # DEPRECATED
    # res = rho0 / (1. + r/Rs) / (1 + (r/Rs)**2)
    res = rho0 / (1. + r/Rs) / (1 + (r/Rs)**2) * _eV4_kpc3_over_Msun
    return res


def M_Burkert(r, delc, Rs=1., h=_h_ref):
    """The Burkert mass profile

    :param r: radius [kpc]
    :param delc: delta_c
    :param Rs: critical radius of Burkert [kpc], default: 1
    :param c: the concentration parameter [1], default: 0.5
    :param h: Hubble const H/(100 km/s/Mpc) [1], default: 0.7
    :returns: mass within the radius r [Msun]

    """
    # delc = delta_c_Burkert(c)
    try:
        res = np.pi * delc * (np.log(1.+(r/Rs)**2) + 2. *
                              np.log(1+r/Rs) - 2.*np.arctan(r/Rs)) * _M_ref
    except RuntimeWarning as e:
        print('r=%e' % r)
        print('Rs=%e' % Rs)
        print('r/Rs=%e' % (r/Rs))
        raise e
    scaling = (h/_h_ref)**2 * (Rs/1)**3
    return res * scaling


###########################
# model fit or independent
###########################
def reconstruct_density_total(gal, flg_give_R=False, interpol_method="linear", interpol_precision=300, flg_errorbar=False, Vth=None):
    """ reconstruct the total density based on the rotaion curve, purely numerically [Msun/pc**3]

    :param gal: galaxy instance
    :param flg_give_R: whether to output R
    :param interpol_method: how to interpolate the neighboring points
    :param interpol_precision: number of data points for the interpolation
    :param flg_errorbar: provide an error bar for rho. Note that this is a rough estimate by translating sigma(Vobs) to sigma(rho). 
    :param Vth: supply velocity directly from model. Vth will be used for the reconstruction instead of gal.Vobs

    """
    if interpol_method == 'nearest':
        if Vth is None:
            V = gal.Vobs
        else:
            V = Vth
        r = gal.R
        M_unit = 232501.397985234  # [Msun] computed with km/s, kpc
        M = V**2 * r * M_unit
        Rmid = (r[1:] + r[:-1]) / 2.
        dr = r[1:] - r[:-1]
        rho = (M[1:] - M[:-1]) / 4./np.pi/Rmid**2 / dr / 1e9  # [Msun/pc**3]
        if flg_give_R:
            if flg_errorbar:
                # estimate of error bar
                Vmid = (V[1:] + V[: -1]) / 2.
                dVmid = np.sqrt((gal.dVobs[1:]**2 + gal.dVobs[: -1])**2)
                factor = 3.7e-5  # [Mpl**2*2*(km/s)**2*kpc/(4pi)]
                sigma_rho = factor * Vmid / Rmid / dr * dVmid
                return (Rmid, rho, sigma_rho)
            else:
                return (Rmid, rho)
        else:
            return rho
    elif interpol_method == 'linear':
        r = np.logspace(np.log10(gal.R[0]), np.log10(
            gal.R[-1]), interpol_precision)
        V = 10**np.interp(np.log10(r), np.log10(gal.R), np.log10(gal.Vobs))
        M_unit = 232501.397985234  # [Msun] computed with km/s, kpc
        M = V**2 * r * M_unit
        Rmid = (r[1:] + r[:-1]) / 2.
        dr = r[1:] - r[:-1]
        rho = (M[1:] - M[:-1]) / 4./np.pi/Rmid**2 / dr / 1e9  # [Msun/pc**3]
        if flg_give_R:
            return (Rmid, rho)
        else:
            return rho
    else:
        raise Exception("Only 'linear' and 'nearest' are implemented.")


def reconstruct_density_DM(gal, DM_profile='NFW'):
    """ reconstruct the DM density based on the rotaion curve, fit with NFW [Msun/pc3]

    """
    # fit with NFW/Burkert
    try:
        c = gal.c
        rs = gal.rs
    except AttributeError:
        chi2.fit_rot_curve(gal, DM_profile=DM_profile)
        c = gal.c
        rs = gal.rs

    def rho(r):
        # return rho_NFW(r, Rs=rs, c=c, h=_h_ref) * _eV4_pc3_over_Msun
        # [Msun/kpc3] to [Msun/pc3]
        return rho_NFW(r, Rs=rs, c=c, h=_h_ref) / 1e9

    return (rho, rs, c)


def reconstruct_mass_total(gal):
    """Numerically reconstruct total M(r) based on v**2(r) purely numerically

    :param gal: galaxy instance

    """
    M_arr = gal.Vobs**2 * gal.R * _Mpl2_km2_over_s2_kpc_over_Msun_
    return M_arr


def reconstruct_mass_DM(gal, DM_profile='NFW'):
    """Reconstruct DM component M(r) based on v**2(r), fit, output mass function [Msun]

    :param gal: galaxy instance

    """
    try:
        c = gal.c
        rs = gal.rs
    except AttributeError:
        chi2.fit_rot_curve(gal, DM_profile=DM_profile)
        c = gal.c
        rs = gal.rs
    if DM_profile == "NFW":
        def mass(r):
            return M_NFW(r, Rs=rs, c=c, h=_h_ref)
    elif DM_profile == "Burkert":
        def mass(r):
            return M_Burkert(r, Rs=rs, delc=c, h=_h_ref)
    return (mass, rs, c)


def v2_rot(gal, c, Rs, ups_bulg, ups_disk, DM_profile, m=None, M=None, flg_debug=False, flg_baryon=True):
    """Constructs the rotation curve [km**2/s**2]. It can produce sol alone, NFW alone, Burkert alone, sol+NFW, and sol+Burkert.

    :param gal: galaxy instance
    :param c: concentration
    :param Rs: transition rs of NFW profile [kpc]
    :param ups_bulg: Upsilon of bulge
    :param ups_disk: Upsilon of disk
    :param DM_profile: dark matter profile ('NFW'|'Burkert')
    :param m: scalar mass [eV]. Soliton component will be neglected if set to None.
    :param M: soliton mass [Msun]. Soliton component will be neglected if set to None.
    :param flg_debug: whether to output the mask for truncating NFW at small radius
    :param flg_baryon: whether to include baryon component in constructing the rotation curve (default: True)

    """
    Vth2_arr = []
    mask = None
    truncated_mass = 0.
    mask = [1.] * len(gal.R)
    # TODO: check if gal.R is already sorted

    # NFW case, determine if soliton is bigger than NFW anywhere
    # if yes, cut out the NFW at small r to prevent NFW beating soliton again
    if (DM_profile == "NFW") and (m is not None) and (M is not None):
        rho_NFW_arr = rho_NFW(gal.R, Rs=Rs, c=c)  # [Msun/kpc**3] unit fixed
        rho_sol_arr = rho_sol(gal.R, m=m, M=M)  # [Msun/kpc**3]
        flg_truncating = False
        truncated_mass = 0.
        for i in range(1, len(gal.R)+1):
            if rho_NFW_arr[-i] < rho_sol_arr[-i]:
                # One-way switch. Once it's on it's never off
                flg_truncating = True

            if flg_truncating is True:
                # rho_NFW_arr[-i] = 0.
                mask[-i] = 0.
                # record the truncated mass
                truncated_mass = max(
                    truncated_mass, M_NFW(gal.R[-i], Rs, c))
    # print("mask: %s" % mask)
    for i, r in enumerate(gal.R):
        # treat the i-th bin of the rot curve
        #
        # TODO: move this part out to a dedicated function
        # V thory due to DM
        M_enclosed = 0.
        if (m is not None) and (M is not None):
            M_enclosed += M_sol(r, m, M)
        if DM_profile == "NFW":
            M_enclosed += (M_NFW(r, Rs, c) - truncated_mass) * mask[i]
        elif DM_profile == "Burkert":
            M_enclosed += M_Burkert(r, delc=c, Rs=Rs)
        else:
            raise Exception(
                "Only NFW and Burkert are implemented at the moment.")
        VDM2 = _G_Msun_over_kpc * _c**2 * (M_enclosed/1.) * (1./r)
        # combine DM with the baryon mass model (from SB data)
        Vb2 = (ups_bulg*np.abs(gal.Vbul[i])*gal.Vbul[i]
               + ups_disk*np.abs(gal.Vdisk[i])*gal.Vdisk[i]
               + np.abs(gal.Vgas[i])*gal.Vgas[i]
               )
        if flg_baryon:
            Vth2 = VDM2 + Vb2
        else:
            Vth2 = VDM2
        Vth2_arr.append(Vth2)
    Vth2_arr = np.array(Vth2_arr)
    if not flg_debug:
        return Vth2_arr
    else:
        return (Vth2_arr, mask)


def sigma_over_vcirc(x):
    """ empirical formula derived from Jeans equation assuming NFW. It is not intended to be called directly. 

    :param x: r/rs

    """
    factor = 0.55 + 0.6 * np.exp(-2*x)**4 + 0.2 * \
        np.exp(-1*x)**2 + 0.2*np.exp(-0.5*x)
    return factor


def sigma_disp_over_vcirc(gal, R):
    """The velocity dispersion computed at r=x*Rs. [km/s]

    :param R: radius [kpc]
    :param gal: galaxy object

    """
    # get Rs
    (rho, rs, c) = reconstruct_density_DM(gal, DM_profile='NFW')

    # make array of r, preferably with gal.R
    x_arr = np.array(gal.R / rs)
    ratio_arr = sigma_over_vcirc(x_arr)

    R, is_scalar = tl.treat_as_arr(R)
    fn = tl.interp_fn(np.stack((gal.R, ratio_arr), axis=-1))
    res = fn(R)
    if is_scalar:
        res = np.squeeze(res)
    return res


def sigma_disp(gal,
               R=None,
               get_ratio=False,
               get_array=False,
               debug=False):
    """The velocity dispersion computed at r=x*Rs. [km/s]

    :param R: radius [kpc]
    :param gal: galaxy object

    """
    # # empirical formula derived from Jeans equation
    # # assuming NFW
    # def sigma_over_vcirc(x):
    #     factor = 0.55 + 0.6 * np.exp(-2*x)**4 + 0.2 * \
    #         np.exp(-1*x)**2 + 0.2*np.exp(-0.5*x)
    #     return factor

    # get Rs
    (rho, rs, c) = reconstruct_density_DM(gal, DM_profile='NFW')

    # make array of r, preferably with gal.R
    x_arr = np.array(gal.R / rs)

    # FIXME: compute the velocity
    ratio_arr = sigma_over_vcirc(x_arr)
    sigma_arr = np.array(gal.Vobs * ratio_arr)

    if get_array:
        res = (gal.R, sigma_arr, ratio_arr)
    else:
        R, is_scalar = tl.treat_as_arr(R)
        # print("x_arr: %s" % x_arr)
        # print("sigma_arr: %s\n" % sigma_arr)

        fn = tl.interp_fn(np.stack((gal.R, sigma_arr), axis=-1), debug=False)
        # for interpolation debugging
        if debug:
            res = np.interp(R, gal.R, sigma_arr)
        else:
            res = fn(R)
        if is_scalar:
            res = np.squeeze(res)
    return res


#######################
# soliton halo relation
#######################
# def M_SH(m, gal, ctilde=0.5):
#     """This is based on eq. 45 (or eq. 25 and 44 in the 2018 paper)

#     :param m: soliton mass [eV]
#     :param Vmax: the maximal velocity [km/s]
#     :param ctilde: the ctilde param defined in 43, varies between 0.35 and 0.55

#     """
#     # Vmax_over_rootPhi = 1.37e5 / _c
#     Vmax_over_rootPhi = 1.37e5  # [km/s]
#     Vmax = gal.get_Vmax()
#     return 2.1 * Vmax / Vmax_over_rootPhi * np.sqrt(ctilde) / m * _Mpl2_over_eV_Msun
#     # E_over_M = (Vmax / _c)**2
#     # return 4.3 * np.sqrt(np.abs(E_over_M)) * _Mpl2_over_eV_Msun/m


def M_SH(m, gal):
    """This is based on eq. 10 of the 2019 SMBH paper. Gives soliton mass [Msun]

    :param m: scalar mass [eV]
    :param gal: the galaxy instance

    """
    Vmax = gal.get_Vmax()
    K_over_M = (Vmax / _c)**2 / 2.
    # virial theorem Ep = -2 Ek = 2 E
    E_over_M = K_over_M
    return 4.3 * np.sqrt(np.abs(E_over_M)) * _Mpl2_over_eV_Msun/m


def bar_ratio_at_peak(gal, M,  Ups_bulg=0.5, Ups_disk=0.5):
    Vobs_arr = gal.Vobs
    Vbar2_arr = Ups_bulg * np.abs(gal.Vbul) * gal.Vbul + Ups_disk * \
        np.abs(gal.Vdisk) * gal.Vdisk + np.abs(gal.Vgas) * gal.Vgas
    max_idx = np.argmax(Vobs_arr)

    Vobs2_max = (Vobs_arr[max_idx]) * (Vobs_arr[max_idx])
    Vbar2_max = Vbar2_arr[max_idx]
    return Vbar2_max / Vobs2_max


# compute the raius predicted by the SH relation
def rc_SH(m, gal):
    """ the radius predicted by the SH relation [kpc]
    """
    Msol = M_SH(m, gal)  # the soliton mass predicted by SH relation
    rc = 2.29e-3 * (Msol/1e11)**(-1) * (m/1e-22)**(-2)
    return rc


########################
# relaxation time scales
########################

def tau(f, m, sigma=57., rho=0.003):
    """ relaxation time computation [Gyr]
    :param f: fraction
    :param m: scalar mass [eV]
    :param sigma: dispersion [km/s]
    :param rho: DM density [Msun/pc**3]

    """
    # fixed: 0.6 -> 2.
    return 2. * 1./f**2 * (m/(1.e-22))**3 * (sigma/100)**6 * (rho/0.1)**(-2)


def relaxation_at_rc(m, gal, f, verbose=0, multiplier=1.):
    """ relaxation time at rc*multiplier, where rc is predicted by the SH relation [Gyr]
    """
    rc_eff = rc_SH(m, gal)*multiplier  # this is rc*multiplier
    if verbose > 1:
        print('rc_SH=%s' % rc_eff)

    # note this should be total density
    r_arr, rho_arr = reconstruct_density_total(gal, flg_give_R=True)
    rho_at_rc = np.interp(rc_eff, r_arr, rho_arr)
    if verbose > 1:
        print('rho_at_rc=%s' % rho_at_rc)

    v_at_rc = np.interp(rc_eff, gal.R, gal.Vobs)
    v_inside_rc_arr = gal.Vobs[gal.R < rc_eff]
    # make sure to use the max of v inside rc, to be conservative
    if len(np.array(v_inside_rc_arr)) > 0:
        v_disp = max(v_at_rc, max(v_inside_rc_arr))
    else:
        v_disp = v_at_rc
    if verbose > 1:
        print('v_at_rc=%s' % v_at_rc)

    # fixed: 0.6 -> 2
    # relax_time = 0.6 * 1/f**2 * (m/1e-22)**3 * (v_at_rc/100)**6 * (rho_at_rc/0.1)**(-2)
    relax_time = 2. * 1/f**2 * (m/1e-22)**3 * \
        (v_disp/100)**6 * (rho_at_rc/0.1)**(-2)

    return relax_time


def relax_radius(f, m, gal, method='num', interpol_method='linear'):
    """Computes the radius within which the relaxation time is smaller

    :param f: fraction of total ULDM
    :param m: mass of ULDM [eV]
    :param gal: galaxy instance
    :param method: method of reconstructing the density profile
    :param interpol_method: method of interpolating the density profile, when method is 'num'

    """
    if method == 'num':
        r_arr = gal.R
        v_arr = gal.Vobs
        r_mid_arr, rho_arr = reconstruct_density_total(
            gal, flg_give_R=True, interpol_method=interpol_method)
        # rho_arr = np.insert(rho_arr, -1, rho_arr[-1])
        v_mid_arr = 10**np.interp(np.log10(r_mid_arr),
                                  np.log10(r_arr), np.log10(v_arr))
        r_arr = r_mid_arr
        v_arr = v_mid_arr     # only use the mid points

    elif method == 'fit':
        # TODO: need to add the baryon component
        print("Using DM component only. This is technically not correct.")
        r_arr = np.logspace(np.log10(gal.R[0]), np.log10(gal.R[-1]), 200)
        rho_fn, _, _ = reconstruct_density_DM(gal)
        rho_arr = rho_fn(r_arr)
        mass_fn, _, _ = reconstruct_mass_DM(gal)
        v_arr = np.sqrt(mass_fn(r_arr) / r_arr /
                        _Mpl2_km2_over_s2_kpc_over_Msun_)

    else:
        raise Exception("method must be either 'num' or 'fit'.")

    sigma_arr = sigma_disp_over_vcirc(gal, r_arr) * v_arr
    # tau_arr = tau(f, m, v_arr, rho_arr)
    # now use dispersion velocity instead of the circ vel
    tau_arr = tau(f, m, sigma_arr, rho_arr)

    # check coulomb log:
    _, mask2 = coulomb_log(gal, m)
    if mask2 is False:
        # make sure relaxation breaks down
        tau_arr = tau_arr * np.inf

    # mask = np.array(np.where(tau_arr - _tau_U_ < 0.)).reshape(-1)
    mask = np.where(tau_arr - _tau_U_ < 0., True, False)

    if len(tau_arr[mask]) == len(r_arr):
        # everything is relaxed for the whole data range
        return r_arr[-1]
    elif len(tau_arr[mask]) == 0:
        # nothing is relaxed for the whole data range
        return -1
    else:
        # return the point it flips from relaxed to unrelaxed
        return r_arr[mask][-1]


def supply_radius(f, m, gal, method='fit'):
    """This is the radius within which there's enough mass to collect to
make the soliton predicted by SH relation.

    """
    # get the soliton mass
    M_SH_val = M_SH(m, gal)

    if method == 'num':
        print("You chose to use total mass to supply the growth of BEC core. This is\
              technically incorrect. Need to subtract the baryonic component,\
              then multiply by the fraction of the correct species. ")
        # TODO: find a way to subtract the baryonic components
        r_arr = gal.R
        M_arr = reconstruct_mass_total(gal)
    elif method == 'fit':
        mass_fn, _, _ = reconstruct_mass_DM(gal)
        r_arr = np.logspace(np.log10(gal.R[0]), np.log10(gal.R[-1]), 200)
        M_arr = mass_fn(r_arr)
    else:
        raise Exception("method must be either 'num' or 'fit'.")

    # find the radius that gives enough mass
    mask = (np.array(np.where(M_arr > M_SH_val))).reshape(-1)

    if len(mask) == 0:
        # this is the soliton takes more than the whole galaxy
        # don't say anything about it
        return -1
    elif len(mask) == len(r_arr):
        # when there's enough mass even within the first point
        return r_arr[0] / np.sqrt(f)
    else:
        r_supply = r_arr[mask][0]
        return r_supply / np.sqrt(f)  # rescale according to isothermal of ULDM


def f_critical(m, gal, factor=1.):
    """It solves the f such that relax_radius meets supply_radius

    """
    f_arr = np.logspace(-3, 0., 100)
    r_relax_arr = np.array(
        [relax_radius(f, m, gal, method='num') for f in f_arr])
    r_supply_arr = np.array(
        [supply_radius(f, m, gal, method='fit') for f in f_arr])

    mask1 = np.where(r_relax_arr != -1, True, False)
    mask2 = np.where(r_supply_arr != -1, True, False)
    mask = mask1 * mask2

    if sum(mask) > 0:
        r_relax_common_arr = r_relax_arr[mask]
        r_supply_common_arr = r_supply_arr[mask]
        solve = np.where(r_supply_common_arr * factor <
                         r_relax_common_arr, True, False)
        if sum(solve) > 0:
            f_critical = min(f_arr[mask][solve])
        else:
            f_critical = 1.5
    else:
        f_critical = 1.1

    # TODO: sanity check to make sure r_core is contained

    return f_critical


def f_critical_two_species(m1, m2, f2, gal):
    """It solves the f1 such that relax_radius meets supply_radius

    """

    f1_arr = np.logspace(-3, np.log10(1-f2), 100)
    r_relax_arr = np.array(
        [max(relax_radius(f1, m1, gal, method='num'), relax_radius(f2, m2, gal, method='num'))
         for f1 in f1_arr])

    r_supply_arr = np.array(
        [supply_radius(f1, m1, gal, method='fit') for f1 in f1_arr])

    mask1 = np.where(r_relax_arr != -1, True, False)
    mask2 = np.where(r_supply_arr != -1, True, False)
    mask = mask1 * mask2

    if sum(mask) > 0:
        r_relax_common_arr = r_relax_arr[mask]
        r_supply_common_arr = r_supply_arr[mask]
        solve = np.where(r_supply_common_arr < r_relax_common_arr, True, False)
        if sum(solve) > 0:
            f_critical = f1_arr[mask][solve][0]
        else:
            f_critical = 1.5
    else:
        f_critical = 1.1
    # TODO: sanity check to make sure r_core is contained

    return f_critical


###########################
# BH absorption time scales
###########################

def tau_BH_smallzeta(m, gal, MBH):
    """The BH absorption time scale [Gyr] for small zeta. Eq. B15 in Bar et al. 2018

    """
    # get halo mass
    mass_fn, _, _ = reconstruct_mass_DM(gal)
    Mh = mass_fn(gal.R[-1])

    # compute time scale
    res = 2.4e8 * (m / 1.e-22)**(-2) * \
        (MBH / 4.e6)**(-1) * (Mh / 1e12)**(-4./3)
    return res


def tau_BH_largezeta(m, gal, MBH):
    """The BH absorption time scale [Gyr] for large zeta. Eq. B16 in Bar et al. 2018

    """
    # get halo mass
    mass_fn, _, _ = reconstruct_mass_DM(gal)
    Mh = mass_fn(gal.R[-1])

    # compute time scale
    res = 1.5e9 * (m / 1.e-22)**(-3) * (MBH / 4.e6)**(-2) * (Mh / 1e12)**(-1)
    return res


def tau_BH(m, gal, MBH, debug_factor=1.):
    """The BH absorption time scale [Gyr]

    :param m: scalar mass [eV]
    :param gal: galaxy instance
    :param MBH: BH mass [Msun]. If it is None type, compute the BH mass using the first data point of the rotation curve (conservative.)
    :param debug_factor: factor for debugging

    """
    if MBH is None:
        MBH = reconstruct_mass_total(gal)[0] * debug_factor
    is_scalar = False
    m_arr = np.array(m)
    if m_arr.ndim == 0:
        is_scalar = True
        m_arr = m_arr[None]
    res_arr = []
    for mi in m_arr:
        # get halo mass
        tau_small_val = tau_BH_smallzeta(mi, gal, MBH)
        tau_large_val = tau_BH_largezeta(mi, gal, MBH)
        tau_min = min(tau_small_val, tau_large_val)
        res_arr.append(tau_min)
    res_arr = np.array(res_arr)

    if is_scalar:
        res = res_arr.squeeze()
    else:
        res = res_arr
    return res


def coulomb_log(gal, m):
    """check if the coulomb_log breaks down the relaxation estimate

    :param gal: galaxy instance
    :param m: scalar mass

    """

    m, is_scalar = tl.treat_as_arr(m)
    R = gal.R[-1]
    sigma = sigma_disp(gal, R) / _c
    res = m * R * _kpc_eV_ * sigma
    flag = np.where(res > _coulomb_log_threadhold_, True, False)
    if is_scalar:
        res = np.squeeze(res)
        flag = np.squeeze(flag)
    return res, flag
