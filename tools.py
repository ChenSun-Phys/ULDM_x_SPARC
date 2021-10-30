import numpy as np
from scipy.interpolate import interp1d


def interp_fn(array, debug=False):
    """
    An interpolator for log-arrays spanning many orders of magnitude.

    Parameters
    ----------
    array : An array of shape (N, 2) from which to interpolate.
    debug : debugging switch
    """

    array[array < 1.e-300] = 1.e-300  # regularizing small numbers

    if debug:
        print(array)

    def fn(x): return 10**interp1d(np.log10(array[:, 0]),
                                   np.log10(array[:, 1]), fill_value=np.nan, bounds_error=False)(np.log10(x))

    return fn


def treat_as_arr(arg):
    """
    A routine to cleverly return scalars as (temporary and fake) arrays. True arrays are returned unharmed. Thanks to Chen!
    """

    arr = np.asarray(arg)
    is_scalar = False

    # making sure scalars are treated properly
    if arr.ndim == 0:  # it is really a scalar!
        arr = arr[None]  # turning scalar into temporary fake array
        is_scalar = True  # keeping track of its scalar nature

    return arr, is_scalar


def scientific(val, output='string'):
    """Convert a number to the scientific form

    :param val: number(s) to be converted
    :param output: LaTeX "string" form or "number" form. (Default: 'string')

    """

    val, is_scalar = treat_as_arr(val)
    exponent, factor = [], []
    string = []

    for vali in val:
        expi = np.floor(np.log10(vali))
        print(expi)
        faci = vali / 10**expi
        # save it
        exponent.append(expi)
        factor.append(faci)
        if round(faci) == 1.:
            string.append(r"$10^{{{:.0f}}}$".format(expi))
        else:
            string.append(
                r"${{{:.0f}}} \times 10^{{{:.0f}}}$".format(faci, expi))
    exponent = np.array(exponent)
    factor = np.array(factor)
    string = np.array(string)

    if is_scalar:
        exponent = np.squeeze(exponent)
        factor = np.squeeze(factor)
        string = np.squeeze(string)
    if output == 'string':
        res = string
    elif output == 'number':
        res = (factor, exponent)
    return res