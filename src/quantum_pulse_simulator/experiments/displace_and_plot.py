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
from quantum_pulse_simulator.experiments.utils.fit_helper import fit_coherent_from_rho
from scipy.optimize import least_squares

import json
from datetime import datetime


@dataclass
class DisplaceAndFitCoherentSettings:
    pulse_length: float = 10e-9
    eta: float = 500e6  # 500 MHz linear driving strength
    amp_scale_for_alpha_1: float = 1.0
    alpha: float = 0.5
    waiting_time: float = 1e-6
    xlim: tuple = (-5, 5)


def run_displace_and_fit_coherent(SYSTEM:QuantumSystem , 
                                    settings:DisplaceAndFitCoherentSettings,
                                    data_save_path:str,
                                    calibration_database_path:str,
                                    show_plot:bool = True):
    dt = datetime.now().strftime('%Y%m%d_%H%M%S')
    ps = PulseSequence(systems=[SYSTEM])
    ps.add_drive(
        duration = settings.pulse_length,
        strength = settings.eta*settings.amp_scale_for_alpha_1*settings.alpha,
        system = SYSTEM,
        shape = ps.flattop_gaussian_shape(settings.pulse_length, 0e-9, fall=True),
        phase = 0.0,
        name = "Drive",
        order = 1
    )

    ps.add_waiting(duration=settings.waiting_time)

    SYSTEMS = [SYSTEM]
    PULSECHAINS = [ps]
    sim = QutipPulseSimulator(
        systems=SYSTEMS,
        pulse_chains=PULSECHAINS,
    )
    
    sim.simulate()
    rho = sim.result.states[-1]
    result = fit_coherent_from_rho(rho)
    alpha = result['alpha']

    xvec = np.linspace(settings.xlim[0], settings.xlim[1], 200)
    w_mat = wigner(rho, xvec, xvec)

    plt.figure()
    plt.imshow(w_mat, extent=[*settings.xlim, *settings.xlim], origin='lower', cmap='RdBu_r')
    plt.colorbar()
    # The arrow was longer because scale_units and angles weren't set to 'xy'.
    # This ensures the arrow length matches the data coordinates.
    plt.quiver(0, 0, alpha.real, alpha.imag, angles='xy', scale_units='xy', scale=1, color='red', width=0.01)
    plt.xlabel("Re(α)")
    plt.ylabel("Im(α)")
    plt.title(f"Wigner Function (α={alpha:.2f})")
    plt.savefig(f"{data_save_path}/wigner_{dt}.png")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
        
    return result
