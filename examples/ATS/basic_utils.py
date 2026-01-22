"""
Created on Thu Dec  9 11:57:29 2021

@author: marer, vethaak
"""

from typing import Tuple

# TODO: move to lib.fitting?
import numpy as np
from scipy.stats import poisson as poissonFunc


def lorentzian(x, x0, a, gamma, bias=0):
    return a * gamma**2 / (gamma**2 + (x - x0) ** 2) + bias


def gaussian(x, x0, a, gamma, bias=0):
    return a * np.exp(-((x - x0) ** 2) / (2 * gamma**2)) + bias


def gaussian_periodic_2pi(x, x0, a, gamma, bias=0):

    return np.array(
        [
            (
                a * np.exp(-(((xval - x0 + np.pi) % (2 * np.pi) - np.pi) ** 2) / (2 * gamma**2)) + bias
                if xval - x0 >= -np.pi and xval - x0 <= np.pi
                else bias
            )
            for xval in x
        ]
    )


def multi_gaussian(x, params):
    off = params[0]
    params_rest = params[1:]
    assert not (len(params_rest) % 3)
    return off + sum([gaussian(x, *params_rest[i : i + 3]) for i in range(0, len(params_rest), 3)])


def poisson(k, lamb):
    return poissonFunc.pmf(k, lamb)


def linear(x, k, bias=0):
    return k * x + bias


def two_d_gaussian(xy, amplitude, x0, y0, sigma_x, sigma_y, offset):
    """
    For use in scipy.optimize.curve_fit, hence the tuple as first argument.
    Inputs:
    * xy = (x,y): each a np.array (2D), the repeated coordinates of the grid,
     that together make up the 'xdata' argument of scipy.optimize.curve_fit().
      E.g. for an x,y square grid between 0 and 2, x = [[0,1,2],[0,1,2],[0,1,2]]
    * amplitude: fitting parameter, height of the Gaussian
    * x0, y0: fitting parameter, coordinates of Gaussian peak,
    * sigma_x, sigma_y: fitting parameters, width of the peak
    * offset: fitting parameter, offset of the Gaussian.
    Return a ravel()-ed 2D array of
    offset + amplitude * np.exp(-((x-x0) / sigma_x)**2 - ((y-y0) / sigma_y)**2)."""
    x, y = xy
    # xy is a tuple of arrays:
    # x = array([[x0,x1,...], [x0,x1,...]]) repeated len(yvalues) times
    # y = array([[y0,y1,...], [y0,y1,...]]) repeated len(xvalues) times
    gaussian_value = offset + amplitude * np.exp(-(((x - x0) / sigma_x) ** 2) - ((y - y0) / sigma_y) ** 2)
    # scipy.optimize requires 1D data, so we use ravel() to flatten the
    # array
    return gaussian_value.ravel()


def sigmoid(x, amp, width, offset_x, offset_y):
    return amp / (1 + np.exp(-(x - offset_x) / width)) + offset_y


def parabolic_fit_function(
    indices: Tuple[float, float], A: float, B: float, C: float, D: float, E: float, F: float
) -> float:
    """
    Calculate the value of a 2D parabolic function at a given point.

    This function computes the value of a 2D parabolic function defined by the coefficients A, B, C, D, E, and F at a specified point (x, y).

    Args:
        indices (Tuple[float, float]): A tuple containing the x and y coordinates where the function is evaluated.
        A (float): Coefficient of the x^2 term.
        B (float): Coefficient of the y^2 term.
        C (float): Coefficient of the x*y term.
        D (float): Coefficient of the x term.
        E (float): Coefficient of the y term.
        F (float): Constant term.

    Returns:
        float: The value of the 2D parabolic function at the given point.

    Examples:
        >>> parabolic_fit_function((1, 2), 1, 1, 1, 1, 1, 1)
        9.0
    """
    x, y = indices
    return A * x**2 + B * y**2 + C * x * y + D * x + E * y + F

