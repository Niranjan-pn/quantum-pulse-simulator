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
from qutip import wigner, destroy, basis,ket2dm

from quantum_pulse_simulator.simulator import QutipPulseSimulator
from quantum_pulse_simulator.pulse import PulseSequence
from quantum_pulse_simulator.devices import QuantumSystem
from quantum_pulse_simulator.experiments.utils.basic_utils import gaussian, gaussian_periodic_2pi
from quantum_pulse_simulator.experiments.utils.fit_helper import fit_coherent_from_rho
from quantum_pulse_simulator.experiments.utils.calibration_utils import fit_trisqueeze,fit_wigner_trisqueezed_state
from scipy.optimize import least_squares

import json
from datetime import datetime


@dataclass
class TrisqueezeandFitSettings:
    pulse_length: float = 60e-9
    g3AC: float = 500e6  # 500 MHz linear driving strength
    eta: float = 1
    waiting_time: float = 1e-6
    xlim: tuple = (-5, 5)

def run_trisqueeze_and_fit(SYSTEM: QuantumSystem, 
                           settings: TrisqueezeandFitSettings, 
                           data_save_path: str, 
                           show_plot: bool = True):
    
    dt = datetime.now().strftime('%Y%m%d_%H%M%S')
    ps = PulseSequence(systems=[SYSTEM])
    ps.add_drive(
        duration = settings.pulse_length,
        strength = settings.g3AC*settings.eta,
        system = SYSTEM,
        shape = ps.flattop_gaussian_shape(settings.pulse_length, 0e-9, fall=True),
        phase = 0.0,
        name = "Drive",
        order = 3
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
    if rho.isket:
            rho_dm =ket2dm(rho)
    else:
        rho_dm = rho

    photon_number_dist = []
    for i in range(SYSTEM.num_fock):
        fock_n = basis(SYSTEM.num_fock, i)
        expected_photon_number =  np.abs(rho.overlap(fock_n))**2 
        photon_number_dist.append(expected_photon_number)

    xvec = np.linspace(settings.xlim[0],settings.xlim[1],100)
    yvec = xvec

    W = wigner(rho_dm, xvec, yvec)
    t_param, fid = fit_trisqueeze(W, xvec, yvec, SYSTEM.num_fock)
    db, angle = fit_wigner_trisqueezed_state(
        wigner_data=W, 
        alphax=xvec, 
        alphay=yvec, 
        cavity_dim=SYSTEM.num_fock, 
        plot=False
    )

    print(f"  Fidelity: {fid:.4f}")
    print(f"  Tri-squeezing: {db:.4f} dB")
    print(f"  Trisqeezing angle : {angle:.4f} rad")

    fig,ax = plt.subplots(1,2,figsize=(10,5))
    im = ax[0].imshow(W, extent=[*settings.xlim, *settings.xlim], origin='lower', cmap='RdBu_r')
    fig.colorbar(im, ax=ax[0])
    ax[0].set_xlabel("Re(α)")
    ax[0].set_ylabel("Im(α)")
    ax[0].set_title(f"Wigner Function")
    
    ax[1].bar(range(SYSTEM.num_fock),photon_number_dist)
    ax[1].set_xlabel("Fock")
    ax[1].set_ylabel("Population")
    ax[1].set_title(f"Photon Number Distribution")
    fig.suptitle(f"Trisqueeze t={(settings.pulse_length + settings.waiting_time) * 1e9:.2f} ns, $\eta$={settings.eta:.2f}, $\\tau$={db:.2f} dB, fid={fid:.4f}")
    
    fig.tight_layout()
    plt.savefig(f"{data_save_path}/{SYSTEM.name}_wigner_{dt}.png")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

    return rho,photon_number_dist,fid,db,angle