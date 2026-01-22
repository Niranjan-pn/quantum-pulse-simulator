import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../ATS')))

from quantum_pulse_simulator.simulator import QutipPulseSimulator
from quantum_pulse_simulator.pulse import PulseSequence
from quantum_pulse_simulator.devices import QuantumSystem
import qutip
from snail_device import eta, DISPLACEMNT_PULSE_LENGTH,amp_1_alpha_working_spot,OMEGA,dc
import pprint as pp
import numpy as np
import matplotlib.pyplot as plt
from calibration_utils import disp_calib, disp_calib_plot, fit_peak
from basic_utils import gaussian_periodic_2pi
import json
from scipy.optimize import curve_fit

#========================
# Simulation setup
#========================
NUM_FOCK = 30
OSC_FREQ = OMEGA 
probe_angles = np.linspace(0, np.pi, 15)
alpha_target = 1
def Run_Kerr_angle(alpha_target, ax=None):
    Vacuum_population = []
    for angle in probe_angles:
        SNAIL_osc = QuantumSystem(
        num_fock=NUM_FOCK,
        omega=OSC_FREQ,
        name="SNAIL Oscillator @ KFP" 
        )

        SNAIL_osc.add_static_nonlinearities(
            strengths=dc
        )

        ps = PulseSequence(systems=[SNAIL_osc])

        ps.add_drive(
            duration = DISPLACEMNT_PULSE_LENGTH,
            strength = amp_1_alpha_working_spot * eta*alpha_target,
            system = SNAIL_osc,
            shape = ps.flattop_gaussian_shape(DISPLACEMNT_PULSE_LENGTH, 2e-9, fall=True),
            phase = 0.0,
            name = "Drive",
            order = 1
        )

        ps.add_waiting(duration = 100e-9)

        ps.add_drive(
            duration = DISPLACEMNT_PULSE_LENGTH,
            strength = amp_1_alpha_working_spot * eta*alpha_target,
            system = SNAIL_osc,
            shape = ps.flattop_gaussian_shape(DISPLACEMNT_PULSE_LENGTH, 2e-9, fall=True),
            phase = angle,
            name = "Drive",
            order = 1
        )

        SYSTEMS = [SNAIL_osc]
        PULSECHAINS = [ps]
        sim = QutipPulseSimulator(
            systems=SYSTEMS,
            pulse_chains=PULSECHAINS,
        )
        
        # Run simulation for this amplitude
        sim.simulate(batch_size=1000, save_prefix=f"SNAIL_kerr_out_in_out", store_batch_file=False)
        
        final_state = sim.result.states[-1]
        vacuum = qutip.basis(NUM_FOCK, 0)
        expectation = qutip.expect(vacuum.proj(), final_state)
        Vacuum_population.append(expectation)

    direction = "auto"
    result = fit_peak(
                probe_angles,
                Vacuum_population,
                peak_found_threshold=4,
                peak_func=gaussian_periodic_2pi,
                peak_direction=direction,
            )
    # Extract fit results
    peak_position = -result["popt"][0]  # minus sign because we are displacing back
    peak_width = result["popt"][2]  # Sigma of Gaussian

    print("===========================")
    print("Alpha Target:", alpha_target)
    print("Peak Position:", peak_position)
    print("Peak Width:", peak_width)
    print("===========================")



    if ax is not None:
        ax.plot(probe_angles/np.pi, Vacuum_population)
        ax.set_xlabel("Probe Angle (pi)")
        ax.set_ylabel("Vacuum Population")
        ax.set_title(f"Alpha={alpha_target:.2f}")
    else:
        plt.plot(probe_angles/np.pi, Vacuum_population)
        plt.xlabel("Probe Angle (pi)")
        plt.ylabel("Vacuum Population")
        plt.title(f"Vacuum Population vs Probe Angle (Alpha={alpha_target:.2f})")
        plt.show()

    return peak_position, peak_width, Vacuum_population


if __name__ == "__main__":
    ALPHA = np.linspace(0,2,10)
    drift_angles = []
    peak_positions = []
    peak_widths = []
    fig, axes = plt.subplots(2, 5, figsize=(15, 10))
    axes = axes.flatten()
    
    def phase_kerr_kerr_prime(x: np.ndarray, offset: float, kerr: float, kerr_p: float) -> np.ndarray:
        """Calculate phase with both Kerr and Kerr prime terms."""
        return -offset - kerr * x**2 - kerr_p * x**4

    def phase_kerr(x: np.ndarray, offset: float, kerr: float) -> np.ndarray:
        """Calculate phase with only Kerr term."""
        return -offset - kerr * x**2    
    
    all_vacuum_populations = []
    for i, alpha_target in enumerate(ALPHA):
        peak_position, peak_width, vac_pop = Run_Kerr_angle(alpha_target, ax=axes[i])
        drift_angles.append(peak_position)
        peak_positions.append(peak_position)
        peak_widths.append(peak_width)
        all_vacuum_populations.append(vac_pop)
    
    drift_angles = np.unwrap(drift_angles)
    popt_kerr_kerr_prime, pcov = curve_fit(
            phase_kerr_kerr_prime, np.array(ALPHA), drift_angles / (2 * np.pi)
        )

    # Fit without Kerr prime term for comparison
    popt_kerr_without_kerr_prime, pcov_kerr = curve_fit(
        phase_kerr, np.array(ALPHA), drift_angles / (2 * np.pi)
    )

    plt.plot(ALPHA, drift_angles, '-o')
    plt.xlabel("Alpha Target")
    plt.ylabel("Drift Angle")
    plt.title(f"SNAIL Kerr - Drift Angle vs Alpha Target (Kerr: {popt_kerr_kerr_prime[1]:.2f}, Kerr Prime: {popt_kerr_kerr_prime[2]:.2f})")
    
    # Store results in a JSON file
    results = {}
    for i, alpha in enumerate(ALPHA):
        results[str(alpha)] = {
            "probe_angles": probe_angles.tolist(),
            "vacuum_population": all_vacuum_populations[i],
            "drift_angle": drift_angles[i]
        }
    
    filename = "snail_kerr_out_in_out_results.json"

    with open(filename, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {filename}")

    plt.show()
