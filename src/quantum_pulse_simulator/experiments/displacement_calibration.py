import math
import cmath as cm
from dataclasses import dataclass, field
from typing import List, Literal, Callable, Optional
from itertools import groupby
from operator import itemgetter

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize, integrate, fftpack
from scipy.special import erf
from qutip import wigner, destroy, basis

from quantum_pulse_simulator.simulator import QutipPulseSimulator
from quantum_pulse_simulator.pulse import PulseSequence
from quantum_pulse_simulator.devices import QuantumSystem
from quantum_pulse_simulator.experiments.utils.basic_utils import gaussian, gaussian_periodic_2pi

from scipy.optimize import least_squares

import json
from datetime import datetime

def funcD(x, n, scale, Ascale):
    return scale * np.exp(-((Ascale * x) ** 2)) * (Ascale * x) ** (2 * n) / (math.factorial(n))


@dataclass
class DisplacementCalibrationSettings:
    pulse_length: float = 10e-9
    eta: float = 500e6  # 500 MHz linear driving strength
    amp_scale_range: List[float] = field(default_factory=lambda: [0.0, 1.0])
    amp_scale_num_points: int = 15

def disp_calib_fitting(xdata, ydata):
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
        yplot[:, i] = funcD(xdata, i, *fit.x,)

    plt.plot(xdata, ydata)
    plt.gca().set_prop_cycle(None)
    plt.plot(xdata, yplot, lw=4, alpha=0.5)
    plt.xlabel("Voltage [a.u]")
    plt.ylabel("Population")
    plt.title(title_extension + f"Displacement calibration scale: {fit.x[1]:.2f} ") 

def run_displacement_calibration_for_amp_scale(SYSTEM:QuantumSystem , 
                                    settings:DisplacementCalibrationSettings,
                                    amp_scale:float ):
    
    ps = PulseSequence(systems=[SYSTEM])
    ps.add_drive(
        duration = settings.PULSE_LENGTH,
        strength = amp_scale * settings.eta,
        system = SYSTEM,
        shape = ps.flattop_gaussian_shape(settings.PULSE_LENGTH, 2e-9, fall=True),
        phase = 0.0,
        name = "Drive",
        order = 1
    )

    SYSTEMS = [SYSTEM]
    PULSECHAINS = [ps]
    sim = QutipPulseSimulator(
        systems=SYSTEMS,
        pulse_chains=PULSECHAINS,
    )
    # Don't save files for every single point in the sweep to avoid clutter/speed up
    batch_files = sim.simulate(batch_size=1000, save_prefix=f"Displacement_{amp_scale:.4f}", store_batch_file=False)
    
    # Get final state
    final_state = sim.result.states[-1]
    
    # Calculate populations
    # Trace out if there were other systems (here only one)
    # But sim.result.states[-1] is already the full state
    # We want population of Fock |0> and |1>
    
    P0 = np.abs(final_state.overlap(SYSTEM.state))**2 # ground state is initial state usually
    # Better: explicit projection
    rho = final_state # It's a ket if using sesolve
    
    # Careful: simulator returns Qobj states.
    # Check if sesolve or mesolve. ats_device has no c_ops so likely sesolve -> kets
    
    # Project onto |0>
    basis_0 = SYSTEM.state # This is initialized as basis(N, 0) in QuantumSystem
    # Project onto |1>
    import qutip as qt
    basis_1 = qt.basis(SYSTEM.num_fock, 1)
    
    pop0 = qt.expect(qt.ket2dm(basis_0), rho)
    pop1 = qt.expect(qt.ket2dm(basis_1), rho)
    
    return pop0, pop1


def run_displacement_calibration(SYSTEM:QuantumSystem , 
                                    settings:DisplacementCalibrationSettings,
                                    data_save_path:str,
                                    calibration_database_path:str,
                                    show_plot:bool = True):

    amp_scales = np.linspace(settings.amp_scale_range[0], settings.amp_scale_range[1], settings.amp_scale_num_points)
    pop0_list = []
    pop1_list = []
    for scale in amp_scales:
        p0, p1 = run_displacement_calibration_for_amp_scale(SYSTEM, settings, scale)
        pop0_list.append(p0)
        pop1_list.append(p1)
    
    xdata = amp_scales
    ydata = np.array([pop0_list, pop1_list])

    # Save data
    data = {
        "Amp_scale": xdata.tolist(),
        "Populations": ydata.tolist(),
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_path = data_save_path + f"/{timestamp}_displacement_calibration.json"
    with open(save_path, "w") as f:
        json.dump(data, f)
    
    print("Fitting displacement calibration...")
    fit = disp_calib_fitting(xdata, ydata)
    print("Fitted displacement calibration scale:", fit.x[1])
    
    disp_calib_plot(xdata, ydata, fit)
    Ascale = fit.x[1]
    amp_scale_for_alpha_1 = 1.0 / Ascale
    
    print(f"\n calibration result:")
    print(f"  Fitted Ascale: {Ascale}")
    print(f"  Amp scale for alpha=1: {amp_scale_for_alpha_1}")
    
    plt.xlabel("Amplitude scale")
    plt.ylabel("Population")
    plt.axvline(amp_scale_for_alpha_1, color='r', linestyle='--', label=f'alpha=1 @ scale={amp_scale_for_alpha_1:.4f}')
    plt.legend()
    plt.title(f"Displacement calibration scale: {fit.x[1]:.2f} with eta={settings.eta/1e6:.3f} MHz")
    # Save plot
    plt.savefig(data_save_path + f"/{timestamp}_displacement_calibration.png")

    if show_plot:
        plt.show()

    try:
        with open(calibration_database_path, "r") as f:
            calibration_file = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        calibration_file = {}

    calibration_file.setdefault("amp_scale_for_alpha_1", amp_scale_for_alpha_1)

    with open(calibration_database_path, "w") as f:
        json.dump(calibration_file, f, indent=4)

    return amp_scale_for_alpha_1
