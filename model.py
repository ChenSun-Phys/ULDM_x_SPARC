#######################################
# This is the code related to
# the model part, including
# 1) NFW mass profile
# 2) Burkert mass profile
# 3) soliton mass profile
#######################################
import numpy as np
from scipy.integrate import quad

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
    :returns: density of the halo [eV^4]

    """

    rhoc = rho_crit(h)
    delc = delta_c(c)
    rhos = rhoc * delc
    res = rhos / (r/Rs * (1 + r/Rs)**2)
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
    """the half density radius

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
    """Soliton mass profile. The form is achieved by integrating rho_sol(r) * 4pi*r^2 dr

    :param r: radius [kpc]
    :param m: scalar mass [eV]
    :param M: soliton mass [Msun]
    :returns: density at r [Msun/kpc^3]

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

    M_norm = 1./11.5865 * (M/1.)  # rho0 * rc^3
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
    :returns: density of the halo [eV^4]

    """
    rhoc = rho_crit(h)
    # delc = delta_c_Burkert(c)
    rho0 = rhoc * delc
    res = rho0 / (1. + r/Rs) / (1 + (r/Rs)**2)
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
    """This is based on eq. 10 of the 2019 SMBH paper

    :param m: soliton mass [eV]
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
