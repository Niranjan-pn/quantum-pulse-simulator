import math
import cmath as cm
from dataclasses import dataclass, field
from typing import List, Literal, Callable, Optional
from itertools import groupby
from operator import itemgetter
import qutip
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

from quantum_pulse_simulator.experiments.utils.fit_helper import fit_peak, get_peak_direction
@dataclass
class KerrDriftAngleSettings:
    DISPLACEMENT_PULSE_LENGTH: float = 10e-9
    DISPLACEMENT_ETA: float = 500e6 
    alpha: float = 1.0
    wait_time: float = 100e-9,
    amp_scale_for_alpha_1: float = 1.0,
    probe_angles_range: tuple = (0, np.pi),
    probe_angles_num: int = 15,
    
def run_kerr_drift_angle(   SYSTEM:QuantumSystem , 
                            settings:KerrDriftAngleSettings,
                            data_save_path:str,
                            calibration_database_path:str,
                            show_plot:bool = True,
                            save_plot:bool = True,
                            ):
    
    probe_angles = np.linspace(settings.probe_angles_range[0], settings.probe_angles_range[1], settings.probe_angles_num)
    
    pop0_list = []
    
    for probe_angle in probe_angles:
        # 1. Define the pulse sequence
        ps = PulseSequence(systems=[SYSTEM])
        
        # Drive 1: Prepare |alpha>
        ps.add_drive(
        duration=settings.DISPLACEMENT_PULSE_LENGTH,
        strength=settings.DISPLACEMENT_ETA * settings.alpha * settings.amp_scale_for_alpha_1,
        system=SYSTEM,
        shape=ps.flattop_gaussian_shape(settings.DISPLACEMENT_PULSE_LENGTH, 2e-9, fall=True),
        phase=0.0,
        name="Drive_in",
        order=1
        )
    
        # Wait time
        ps.add_waiting(settings.wait_time, SYSTEM)
    
        # # Drive 2: Prepare |-alpha>
        # ps.add_drive(
        # duration=settings.DISPLACEMENT_PULSE_LENGTH,
        # strength=settings.DISPLACEMENT_ETA * settings.alpha,
        # system=SYSTEM,
        # shape=ps.flattop_gaussian_shape(settings.DISPLACEMENT_PULSE_LENGTH, 2e-9, fall=True),
        # phase=probe_angle,  # Phase shift by pi for |-alpha>
        # name="Drive_out",
        # order=2
        # )
        
        # 2. Setup Simulator
        SYSTEMS = [SYSTEM]
        PULSECHAINS = [ps]
        sim = QutipPulseSimulator(
            systems=SYSTEMS,
            pulse_chains=PULSECHAINS,
        )

        sim.simulate(batch_size=1000, save_prefix=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_drift_angle", store_batch_file=False)
        
        final_state = sim.result.states[-1]
        vacuum = qutip.basis(SYSTEM.num_fock, 0)
        expectation = qutip.expect(vacuum.proj(), final_state)
        pop0_list.append(expectation)
    

    result = fit_peak(
                probe_angles,
                pop0_list,
                peak_found_threshold=4,
                peak_func=gaussian_periodic_2pi,
                peak_direction='up',
            )
    # Extract fit results
    peak_position = -result["popt"][0]  # minus sign because we are displacing back
    peak_width = result["popt"][2]  # Sigma of Gaussian

    print("===========================")
    print("Alpha Target:", settings.alpha)
    print("Peak Position:", peak_position)
    print("Peak Width:", peak_width)
    print("===========================")
    
    data = {
        "probe_angles": probe_angles.tolist(),
        "pop0_list": pop0_list,
        "peak_position": peak_position,
        "peak_width": peak_width,
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"{data_save_path}/kerr_drift_angle_{timestamp}.json"
    with open(save_path, "w") as f:
        json.dump(data, f)
    
    if save_plot:
        plt.figure()
        plt.plot(probe_angles/np.pi, pop0_list, 'o')
        plt.plot(probe_angles/np.pi, gaussian_periodic_2pi(probe_angles, *result["popt"]))
        plt.xlabel("Probe Angle (pi)")
        plt.ylabel("Vacuum Population")
        plt.title(f"Vacuum Population vs Probe Angle (Alpha={settings.alpha:.2f})")
        plt.savefig(f"{data_save_path}/kerr_drift_angle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    
    if show_plot:
        plt.show()
    
    return peak_position, peak_width, probe_angles, pop0_list