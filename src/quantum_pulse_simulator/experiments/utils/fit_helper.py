import numpy as np
from typing import Literal, Callable, Optional
from scipy.optimize import curve_fit, least_squares
from itertools import groupby
from operator import itemgetter
import math
import qutip as qt
import matplotlib.pyplot as plt
from scipy import fftpack
import scipy.integrate as integrate
import cmath as cm
import scipy
from scipy import optimize

from typing import Literal, Callable, Optional
from itertools import groupby
from operator import itemgetter

import quantum_pulse_simulator.experiments.utils.simple as simplefit
from quantum_pulse_simulator.experiments.utils.basic_utils import gaussian, gaussian_periodic_2pi


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
        # For periodic function, we allow the peak to be anywhere within the data range plus a buffer
        bounds[0][0] = peak_x_val - np.pi
        bounds[1][0] = peak_x_val + np.pi


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

def fit_coherent_from_rho(rho):
    """
    Fits a coherent state to a density matrix by calculating the mean field.
    
    Parameters:
        rho (Qobj): The density matrix or ket state.
        
    Returns:
        dict: Dictionary containing alpha, amplitude, phase, and fidelity.
    """
    # 1. Calculate expectation value of annihilation operator (a)
    # This gives the center of mass of the Wigner function: alpha = <a_hat>
    # Note: If rho is mixed (e.g. thermal), this gives the displacement of the center.
    N = rho.dims[0][0]
    a_op = qt.destroy(N)
    
    alpha_fit = qt.expect(a_op, rho)
    
    # 2. Extract Amplitude and Phase
    amplitude = np.abs(alpha_fit)
    phase = np.angle(alpha_fit)
    
    # 3. Calculate Fidelity with the perfect coherent state
    # This tells you "how coherent" the state actually is.
    rho_fit = qt.coherent_dm(N, alpha_fit)
    fid = qt.fidelity(rho, rho_fit)**2  # Squared fidelity (probability overlap)
    
    return {
        "alpha": alpha_fit,
        "amplitude": amplitude,
        "phase": phase,
        "fidelity": fid
    }