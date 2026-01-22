"""
Created on Thu Nov 12 12:08:01 2020

@author: kervinen
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from numpy import *
from scipy.special import erf
from qutip import wigner, destroy, basis
import scipy.fftpack
from scipy import fftpack
import scipy.integrate as integrate
import cmath as cm
import scipy
from scipy import optimize
from basic_utils import gaussian, gaussian_periodic_2pi
from typing import Literal, Callable, Optional
from itertools import groupby
from operator import itemgetter

import simple as simplefit
from scipy.optimize import curve_fit
def funcD(x, n, scale, Ascale):
    return scale * np.exp(-((Ascale * x) ** 2)) * (Ascale * x) ** (2 * n) / (math.factorial(n))


def disp_calib(xdata, ydata):
    """
    Args:
        xdata (list): drive amplitudes
        ydata (list): containing two lists for measured amplitudes at ground state and fock 1 (shape doesn't matter)
    """
    ydata = ydata / np.max(ydata)
    N = np.min(ydata.shape)

    x = xdata

    if ydata.shape[1] > ydata.shape[0]:
        ydata = ydata.T

    y = np.matrix.flatten(ydata, order="F")

    def lsq_func(params, *args):
        scale = params[0]
        Ascale = params[1]
        n = int(args[2])
        x = args[0]
        y = args[1]

        yfit = np.empty(y.shape)
        for i in range(n):
            yfit[i * len(x) : (i + 1) * len(x)] = funcD(x, i, scale, Ascale)
        return y - yfit

    args = (x, y, N)

    scale0 = 1
    Ascale0 = 10
    params0 = np.array([scale0, Ascale0])
    bounds = [0, np.inf]
    fit = least_squares(lsq_func, params0, args=args, bounds=bounds)

    return fit


def disp_calib_plot(xdata, ydata, fit, title_extension=""):
    """
    Args:
        xdata (list): drive amplitudes
        ydata (list): containing two lists for measured amplitudes at ground state and fock 1 (shape doesn't matter)
        fit (OptimizeResult): fitted results by disp_calib
    """
    ydata = ydata / np.max(ydata)
    if ydata.shape[1] > ydata.shape[0]:
        ydata = ydata.T

    yplot = np.zeros(ydata.shape)
    N = np.min(ydata.shape)
    for i in range(N):
        yplot[:, i] = funcD(xdata, i, *fit.x)

    plt.plot(xdata, ydata)
    plt.gca().set_prop_cycle(None)
    plt.plot(xdata, yplot, lw=4, alpha=0.5)
    plt.xlabel("Voltage [a.u]")
    plt.ylabel("Population")
    plt.title(title_extension + f"Displacement calibration scale: {fit.x[1]:.2f} ") 
    print("Displacement calibration scale:", fit.x[1])

def fit_wigner_trisqueezed_state(wigner_data, alphax, alphay, cavity_dim=40, plot=False, save_path=None):
    """Fit trisqueezed wigner state
    Args: 
        wigner_data (ndarray): 2d wigner pixels
        alphax (ndarray): alpha values at x-pixels
        alphay (ndarray): alpha values at y-pixels
        cavity_dim (int): Hilbert space dimension of the fitted cavity state
        plot (bool): shows plot if true
        save_path (str): saves to path if not None
    """
    a = destroy(cavity_dim)
    a_dag = a.dag()
    psi0 = basis(cavity_dim,0)

    def get_wigner_trisqueezed_state(triplicity): # Function to determine the unknown parameter 
        T = (-1j*((triplicity[0]+1j*triplicity[1])*a**3 + (triplicity[0]-1j*triplicity[1])*a_dag**3)).expm()
        state = (T*psi0).unit()
        return np.pi/2 * wigner(state, alphax, alphay, g=2)
    
    def cost_func(triplicity):# Introduce the function needs to be minimized
        tot = sum(((get_wigner_trisqueezed_state(triplicity)-wigner_data).flatten())**2)
        return tot

    guesses = [[0.01, 0]]
    opt_results = [optimize.minimize(cost_func, triplic_guess).x for triplic_guess in guesses]
    triplicity_fit = guesses[0]
    for res in opt_results:
        if cost_func(res) < cost_func(triplicity_fit):
            triplicity_fit = res


    trisqueezing_angle = np.arctan2(triplicity_fit[1], triplicity_fit[0]) / 3  # divided by 3 due to triplicity
    trisqueezing_db = abs(triplicity_fit[0]+1j*triplicity_fit[1]) * 20/np.log(10)  # r to dB conversion

    disp_min_x = alphax[-1]
    disp_max_x = alphax[0]
    disp_min_y = alphay[-1]
    disp_max_y = alphay[0]

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
    # imshow_params = {'interpolation': 'none',
    #                     'origin': 'lower',
    #                     'extent': (disp_min_x, disp_max_x, disp_min_y, disp_max_y),
    #                     'cmap': 'RdBu',
    #                     'vmin': -1,
    #                     'vmax': 1}

    # # plot Wigner tomography data
    # ax1.imshow(wigner_data, **imshow_params)
    # ax1.set_xlabel(r'Re($\alpha$)')
    # ax1.set_ylabel(r'Im($\alpha$)')
    # ax1.tick_params(axis='both', which='major', labelsize=18)

    # # plot fitted Wigner function
    # x, y = np.meshgrid(alphax, alphay)
    # c = ax2.imshow(get_wigner_trisqueezed_state(triplicity_fit), **imshow_params)
    # ax2.tick_params('y', labelleft=False)
    # # ax2.text(0.05, 0.05, rf"${trisqueezing_db:.3f}dB,{trisqueezing_angle:.3f} rad$", transform=ax2.transAxes)
    # ax2.set_xlabel(r'Re($\alpha$)')
    # ax2.tick_params(axis='both', which='major', labelsize=18)
    # if plot:
    #     plt.show()

    # ax1.set_title('Wigner tomography')
    # ax2.set_title('Fitted Wigner function')
    # # fig.colorbar(c, ax=[ax1, ax2], shrink=0.75)
    # # if save:
    # if save_path != None:
    #     plt.savefig(save_path / f'{trisqueezing_db:.3f}dB_{trisqueezing_angle:.3f}_fitted.png')
    # # if show:
    # #     plt.show()
    # # plt.close()

    return trisqueezing_db, trisqueezing_angle

def generate_trisqueeze(a, tcx):
    """
    Generate unitary operator for tri-squeezing.
    Args:
        a (Qobj): Annihilation operator
        tcx (complex): Complex trisqueezing parameter (r * e^{i*theta})
    Returns:
        Qobj: Unitary operator D3(tcx)
    """
    # Interaction Hamiltonian is typically proportional to (a^3 - a_dag^3) or similar.
    # Based on user's previous code: T = (-1j*((triplicity[0]+1j*triplicity[1])*a**3 + (triplicity[0]-1j*triplicity[1])*a_dag**3)).expm()
    # If tcx = triplicity[0] + 1j*triplicity[1]
    # Then term = tcx*a^3 + tcx.conj()*a_dag^3
    # Exponent is -1j * term.
    
    a_dag = a.dag()
    # Ensure tcx is complex
    if isinstance(tcx, (float, int)):
        tcx = complex(tcx)
        
    op = -1j * (tcx * a**3 + np.conj(tcx) * a_dag**3)
    return op.expm()

def fit_trisqueeze (W, x, y, N_fock) : 
    """
    arguments : 
        W : 2d array, wigner function
        x : 1d array, x coordinate of the wigner function
        y : 1d array, y coordinate of the wigner function
        N_fock : truncation level of fock space. 
    returns :
    t : complex trisqueezing. -pi < angle(t) < pi are all distinct states.
    fid : fidelity
    """
    a = destroy(N_fock)
    O = basis(N_fock, 0)

    dx = (x[1] - x[0])
    dy = (y[1] - y[0])
    
    def cost_func(t, W, x, y):
        tcx = t[0] * np.exp(1j*t[1])
        # Note: 'generate_trisqueeze' helper needed.
        unitary = generate_trisqueeze(a, tcx)
        test_state =  unitary * O
        test_W = wigner(test_state, x, y)
        test_W /= np.sum(test_W) * dx*dy #normalisation  
        fid = np.sum(W*test_W) * dx*dy *2*np.pi # 2pi factor from Wigner definition?
        # User snippet had * 2*pi. Typically overlap is integral(W1 W2) * 2pi for normalization definition
        return 1-fid.real

    bounds = ((-1, 1), (-np.pi, np.pi))

    # Initial guess
    res = optimize.minimize(cost_func, (0.01, 0), args = (W, x, y), bounds=bounds)
    t = res.x[0] * np.exp(1j*res.x[1])
    fid = 1-cost_func((np.abs(t), np.angle(t)), W, x, y)
    return (t, fid)

import numpy as np

def calculate_phase_space_metrics(W, xvec, yvec, epsilon=0.01):
    """
    Calculate phase space support metrics (Radius and Negativity) for a Wigner function.
    
    Args:
        W (ndarray): Wigner function data (2D).
        xvec (ndarray): x coordinates.
        yvec (ndarray): p coordinates (y).
        epsilon (float): Threshold for radial support (default 0.01 for 99%).
        
    Returns:
        dict: Dictionary containing:
            - 'R_epsilon': Radius containing (1-epsilon) of the probability.
            - 'V_neg': Negativity volume.
            - 'Trace': Total integrated probability (should be ~1.0).
    """
    dx = xvec[1] - xvec[0]
    dy = yvec[1] - yvec[0]
    dA = dx * dy
    
    # 1. Negativity Volume (V_neg)
    # This metric is robust and calculation is correct
    V_neg = np.sum(np.abs(W[W < 0])) * dA
    
    # 2. Radial Support (R_epsilon)
    X, Y = np.meshgrid(xvec, yvec)
    R_grid = np.sqrt(X**2 + Y**2)
    
    # Flatten and sort by radius
    r_flat = R_grid.flatten()
    w_flat = W.flatten()
    
    sort_idx = np.argsort(r_flat)
    r_sorted = r_flat[sort_idx]
    w_sorted = w_flat[sort_idx]
    
    # Cumulative sum of probability
    # Note: Wigner functions oscillate, so this curve isn't strictly monotonic 
    # at small radii, but stabilizes in the tails.
    cum_prob = np.cumsum(w_sorted) * dA
    
    # normalization: use actual trace from grid to handle slight numerical deviations
    total_trace = cum_prob[-1] 
    
    # Safety Check: Warn if grid is too small
    if abs(total_trace - 1.0) > 0.1:
        print(f"Warning: Wigner function trace is {total_trace:.4f}. "
              "Grid might be too small or resolution too low.")

    # Target probability threshold
    target = (1 - epsilon) * total_trace    
    
    # Find first index where cumulative probability exceeds target
    # Since we look for 99%, we are likely in the 'tail' where curve is monotonic.
    idx_cross = np.argmax(cum_prob >= target)
    R_epsilon = r_sorted[idx_cross]
    
    return R_epsilon, V_neg, total_trace

def get_peak_direction(ydata: np.ndarray, peak_direction: Literal["up", "down", "auto"] = "auto"):
    """Validate and select peak direction from options and data."""
    assert peak_direction in ["up", "down", "auto"]
    if peak_direction == "auto":
        peak_direction = "up" if abs(max(ydata) - np.mean(ydata)) > abs(min(ydata) - np.mean(ydata)) else "down"
    return peak_direction

def fit_peak(
    xdata: np.ndarray,
    ydata: np.ndarray,
    peak_func: Callable = gaussian,
    fit_peak_at_x_value: Optional[float] = None,
    peak_direction: Literal["up", "down", "auto"] = "auto",
    peak_found_threshold: int = 4,
    save: bool = False,
    save_plot_path: Optional[str] = None,
    figname: str = "fit_peak",
    plot: bool = False,
) -> dict:
    """
    Fit data to a peaked function.

    Args:
        xdata: Ordinates of the data (Should be real valued).
        ydata: Abscissae of the data (Should be real valued).
        peak_func: function to be fitted taking the arguments (data, peak_x_val, amp, gamma, bias)
        fit_peak_at_x_value: x-value of the peak to be fitted (useful if several exists)
        peak_direction: Whether the peak points up or down ('up' or 'down' or 'auto').
        peak_found_threshold: The minimum number of sigmas above floor noise to decide a peak was found.
        save: it True, save to the figure.
        save_plot_path: where to save the figure.
        figname: figure name when saved.

    Returns:
        A dictionary containing "popt", "pcov", "errors", "initial_guess" and "peak_found".
    """
    xdata = np.asarray(xdata)
    assert np.all(np.isreal(xdata)), "xdata should not be complex"
    assert np.all(np.isreal(ydata)), "ydata should not be complex"
    peak_direction = get_peak_direction(ydata, peak_direction)

    if fit_peak_at_x_value is None:
        peak_x_val = xdata[np.argmax(ydata)] if peak_direction == "up" else xdata[np.argmin(ydata)]
    else:
        peak_x_val = fit_peak_at_x_value

    if peak_direction == "up":
        amplitude = max(ydata) - np.mean(ydata)
        idx_peaks = np.where(ydata > np.mean(ydata) + amplitude / 4)[0]
    else:
        amplitude = min(ydata) - np.mean(ydata)
        idx_peaks = np.where(ydata < np.mean(ydata) + amplitude / 4)[0]

    gamma_estimate = (max(xdata) - min(xdata)) / 5
    for _, g in groupby(enumerate(idx_peaks), lambda i_x: i_x[0] - i_x[1]):
        group_idxs = list(map(itemgetter(1), g))
        group_xdata = xdata[group_idxs]
        if peak_x_val >= min(group_xdata) and peak_x_val < max(group_xdata):
            gamma_estimate = max(max(group_xdata) - min(group_xdata), 2 * np.diff(xdata)[0])
            break

    initial_guess = [
        peak_x_val,
        amplitude,
        gamma_estimate / 2,  # the interval over which the fitting happens depends on the data
        np.mean(ydata),
    ]
    bounds = (
        [peak_x_val - np.abs(gamma_estimate), -np.inf, 0, -np.inf],
        [peak_x_val + np.abs(gamma_estimate), np.inf, np.inf, np.inf],
    )

    if peak_func == gaussian_periodic_2pi:
        bounds[0][0] = max(peak_x_val - gamma_estimate, 0)
        bounds[1][0] = min(peak_x_val + gamma_estimate, 2 * np.pi)

    popt, pcov = curve_fit(peak_func, xdata, ydata, p0=initial_guess, maxfev=3000000, bounds=bounds)
    errors = simplefit.get_fit_errors(popt, pcov)
    peak_found = max(ydata) > np.mean(ydata) + peak_found_threshold * np.std(ydata)

    # Plotting the fit
    if plot:
        plt.figure(figsize=(8, 6))
        plt.plot(xdata, ydata, "b-", label="Data")
        plt.plot(xdata, peak_func(xdata, *popt), "r-", label="Fit: " + peak_func.__name__)
        plt.axvline(peak_x_val, ls="--", color="black", label="initial x0 estimate and bounds")
        plt.axvline(peak_x_val - gamma_estimate, ls="--", color="black")
        plt.axvline(peak_x_val + gamma_estimate, ls="--", color="black")
        plt.axvline(popt[0])
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.title("Peak Fitting")
        plt.legend()
        plt.grid(True)
        if save and save_plot_path is not None:
            plt.savefig(save_plot_path / f"{figname}_{peak_func.__name__}.png")

    return {
        "popt": popt,
        "pcov": pcov,
        "errors": errors,
        "initial_guess": initial_guess,
        "peak_found": peak_found,
    }
