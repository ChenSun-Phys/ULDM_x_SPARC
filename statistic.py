""" This module computes some basic statistics. 
Module author: Chen Sun
Year: 2020, 2021
Email: chensun@mail.tau.ac.il

"""

from __future__ import division
from scipy.integrate import quad
from scipy.optimize import newton, bisect
import numpy as np
import mpmath as mp  # for arbitrary precision

# set precision
mp.mp.dps = 100.


def gauss1d(x, mu=0., sig=1.):
    """The 1D Gaussian distribution. Note that it is defined using mpmath, meaning it's slow but with high precision

    :param x: random variable x
    :param mu: expectation value
    :param sig: standard deviation

    """

    pdf = 1./sig/mp.sqrt(2.*mp.pi) * mp.exp(-1./2*(x - mu)**2/sig**2)
    return pdf


def Omega(d):
    if d % 2 == 0:
        # return 1./np.math.factorial(d/2 - 1) * 2.*np.pi**(d/2)
        return 1./mp.factorial(d/2 - 1) * 2.*mp.pi**(d/2)
    elif d % 2 == 1:
        # return np.math.factorial(1./2*(d-1))/(np.math.factorial(d-1)) * 2.**d * np.pi**((d-1)/2.)
        return mp.factorial(1./2*(d-1))/(mp.factorial(d-1)) * 2.**d * mp.pi**((d-1)/2.)


def gausscdf(chi, d):
    """CDF of d-dimensional Gaussian distribution, with mu_i = 0., sig_i = 1.

    :param chi: the localtion of r = sqrt(x_1^2 + x_2^2 + ... + x_d^2)
    :param d: number of the random variables

    """

    def integrand(r):
        return Omega(d) / mp.sqrt((2.*mp.pi)**d) * r**(d-1.) * mp.exp(-1./2*r**2)

    # cdf = quad(integrand, 0., float(chi))
    cdf = mp.quad(integrand, [0., float(chi)])
    return cdf

# findroot


def chi1d_to_chind(chi1d, d):
    x = chi1d
    # cdf1d = quad(gauss1d, -x, x, (0., 1.))[0]
    cdf1d = mp.quad(gauss1d, [-x, x])

    def f(x1):
        cdfnd = gausscdf(x1, d)
        return cdfnd - cdf1d
    # res = newton(f, 1.)
    res = bisect(f, a=0., b=13.)
    return res


# test
def delta_c_Burkert(c):
    """the multiplier of the critical density that goes into
    rho_Burkert

    :param c: the concentration parameter 
    :returns: delta_c 
    """
    res = -800.*mp.power(c, 3)/(-3.*mp.log(mp.power(c, 2) +
                                           1) - 6.*mp.log(c+1) + 6.*mp.atan(c))
    return res
