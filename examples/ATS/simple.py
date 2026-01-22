import numpy as np


def lorentzian_abs(x, a0, a1, Ql, x0):
    return np.abs((a0 - a1 / (np.pi * x0 / Ql + 1j * 2 * np.pi * (x - x0))))


def lorentzian(x, x0, a, gamma, bias=0):
    return a * gamma**2 / (gamma**2 + (x - x0) ** 2) + bias


def rabi_fit(t, a, off, t1, phi):
    return off - a * np.cos(2 * np.pi / t1 * t + phi)


def rabi_fit_decay(t, a, off, t1, t2):
    return off - a * np.cos(2 * np.pi / t1 * t) * np.exp(-t / t2)


def rabi_fit_var_phase(t, a, off, t1, phi):
    return off + a * np.sin(2 * np.pi / t1 * t + phi)


def T1_fit(t, a, t1, off):
    return off + a * np.exp(-t / t1)


def T2Echo_fit(t, a, t1, off):
    return T1_fit(t, a, t1, off)


def ramsey_fit(t, a, t2, off, frequency, phi0):
    return off + a * np.exp(-t / t2) * np.cos(2 * np.pi * frequency * t + phi0)


def ramsey_fit_cut(t, a, t2, off, frequency, phi0):
    return off + a * np.cos(2 * np.pi * frequency * t + phi0)


def ramsey_fit2(t, a, t2, f, phi, b):
    return a / 2 * np.cos(2 * np.pi * f * t + phi) + b


def ramsey_fit3(x, a, b, phi, t2, d):
    return a * np.cos(2 * np.pi * b * x + phi) * np.exp(-x / t2) + d


def gauss_fit(f, mu, sigma, off, a):
    return off + np.array(a / (sigma * (2 * np.pi) ** (1 / 2)) * np.exp(-1 / 2 * ((f - mu) / sigma) ** 2))

def gauss_simple_fit(f, mu, sigma, A):
    return A * np.exp(-((f - mu) ** 2) / (2 * sigma ** 2))

def two_gauss_fit(f, mu_0, delta_mu, beta2, k, sigma, off):
    return gauss_fit(f, mu_0, sigma, off, k * np.exp(-beta2)) + gauss_fit(
        f, mu_0 - delta_mu, sigma, off, k * beta2 * np.exp(-beta2)
    )

def three_gauss_fit(f, mu0, sigma0, A0, mu1, sigma1, A1, mu2, sigma2, A2, offset):
    return (gauss_simple_fit(f, mu0, sigma0, A0) + gauss_simple_fit(f, mu1, sigma1, A1) + gauss_simple_fit(f, mu2, sigma2, A2) + offset)

def drag_fit(x, a, off, freq, beta, gate_reps=5):

    return off + a * np.cos(2 * np.pi * freq * gate_reps * (x - beta))


def ape_phase_fit(x, a, off, phi, dtheta, gate_reps=5):

    return off + a * np.cos(x - phi + gate_reps * dtheta * 2)


def gate_rotation_fit(n_reps, a, off, d_theta, theta_target=np.pi / 2):

    return off + a * np.cos(n_reps * (theta_target + d_theta))


def exponential_decay(x: np.ndarray, a: float, time_constant: float) -> np.ndarray:
    """
    Calculates the exponential decay of a signal over a given range.

    The formula used is: a * exp(-x / time_constant), where 'a' is the initial amplitude,
    'x' is the time or position array, and 'time_constant' is the decay constant of the exponential
    function.

    Parameters:
    x (np.ndarray): An array of time or position values where the exponential decay is calculated.
    a (float): The initial amplitude of the exponential decay.
    time_constant (float): The time constant of the exponential decay, which determines the rate of decay.

    Returns:
    np.ndarray: An array of the same shape as 'x', containing the calculated exponential decay values at each point.

    Note:
    The time constant should be a positive number for decay.
    """
    return a * np.exp(-x / time_constant)


def gaussian_no_offset(x: np.ndarray, x0: float, a: float, gamma: float) -> np.ndarray:
    """
    Computes the values of a Gaussian function.

    This function represents a Gaussian distribution centered at 'x0' with amplitude 'a' and
    a standard deviation of 'gamma'. It calculates the Gaussian value for each point in 'x',
    without including any offset (baseline) in the Gaussian equation.

    Parameters:
    x (np.ndarray): A NumPy array of points at which the Gaussian function is to be evaluated.
    x0 (float): The position of the center of the Gaussian peak.
    a (float): The amplitude of the Gaussian peak.
    gamma (float): The standard deviation of the Gaussian peak, related to its width.

    Returns:
    np.ndarray: An array of the same shape as 'x', containing the Gaussian function values at the given points.

    Note:
    - This function assumes that 'gamma' is positive. A gamma value of zero or negative will
    result in division by zero or an invalid Gaussian function.

    - Offset value is not included in this function.
    """
    return a * np.exp(-((x - x0) ** 2) / (2 * gamma**2))


def get_fit_errors(popt: np.ndarray, pcov: np.ndarray, scaling: str = "percent") -> np.ndarray:
    """
    Calculates the errors of the fit parameters with different scaling options.

    This function computes the errors of the fit parameters from the covariance matrix.
    It offers different scaling options for these errors: as a percentage, as standard
    deviations, or as a fraction of the parameter values.

    Parameters:
    - popt (array-like): Optimal values for the fit parameters obtained from a fitting function.
    - pcov (2D array-like): Covariance matrix of the fit parameters, obtained from a fitting function.
    - scaling (str, optional): The scaling mode for the errors. Options are 'percent' for percentage errors,
                             'standard_deviation' for standard deviation errors, and any other string for fractional errors.
                             Defaults to 'percent' to maintain backward compatibility.

    Returns:
    - numpy.ndarray: An array of the calculated errors for each fit parameter according to the chosen scaling.

    Raises:
    - NotImplementedError: If an invalid scaling option is provided.

    Example:
    >>> popt = [2.0, 3.0]
    >>> pcov = np.array([[0.04, 0], [0, 0.09]])
    >>> errors = get_fit_errors(popt, pcov, scaling="percent")
    >>> print(errors)
    [10.0, 10.0]  # Indicates 10% error on each parameter

    Notes:
    The function assumes that the covariance matrix is correctly estimated and that the fit parameters are non-zero.
    A division by zero will occur if any of the parameters in 'popt' are zero while using 'percent' or default scaling.
    """

    valid_scalings = ["percent", "standard_deviation", "fractional"]
    if scaling not in valid_scalings:
        raise NotImplementedError(
            "scaling key invalid. Valid choices are {}. Using default {:s}".format(valid_scalings, scaling)
        )

    sigma = np.sqrt(np.diag(pcov))
    if scaling == "percent":
        return 100 * np.abs(sigma / popt)
    elif scaling == "standard_deviation":
        return sigma
    elif scaling == "fractional":
        return np.abs(sigma / popt)


def get_square_error(y_data, y_fit):
    return np.sum(np.abs(y_data - y_fit) ** 2)
