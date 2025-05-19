"""
Example: Kerr Oscillator Squeezing with Two-Photon Drive

This script demonstrates squeezing dynamics in a Kerr oscillator under a two-photon drive,
showing Wigner function evolution using QuTiP simulations.

Key Components:
1. Creates Kerr oscillator with specified nonlinearity
2. Applies shaped two-photon drive pulse
3. Simulates time evolution
4. Animates Wigner function dynamics
"""

import numpy as np
import qutip as qt
from quantum_pulse_simulator.simulator import QutipPulseSimulator

from quantum_pulse_simulator.pulse import PulseSequence
from quantum_pulse_simulator.devices import QuantumSystem
# =========================
# System Parameters
# =========================
KERR_COEFF = 6.7e6 * 2 * np.pi  # Kerr nonlinearity (rad/s)
OSC_FREQ = 6e9 * 2 * np.pi  # Oscillator frequency (rad/s)
NUM_FOCK = 50  # Hilbert space dimension
ALPHA_TARGET = 2.6  # Target coherent state amplitude

# =========================
# Pulse Parameters 
# =========================
PULSE_DURATION = 1000e-9  # Total pulse duration (s)
RISE_TIME = 320e-9  # Pulse rise/fall time (s)

def main():
    """Main simulation sequence for Kerr oscillator squeezing."""
    
    # Calculate derived parameters
    e2_drive = ALPHA_TARGET**2 * KERR_COEFF  # Two-photon drive strength
    
    # Initialize Kerr oscillator
    kerr_osc = QuantumSystem(
        num_fock=NUM_FOCK,
        omega=OSC_FREQ,
        name="Kerr Oscillator"
    )

    kerr_osc.add_kerr_oscillator(KERR_COEFF)

    # Pulse sequence setup
    ps = PulseSequence(systems=[kerr_osc])
    ps.add_two_photon_drive(
        duration=PULSE_DURATION,
        strength=e2_drive,
        system=kerr_osc,
        shape=ps.flattop_gaussian_shape(PULSE_DURATION, RISE_TIME),
        phase=0.0,
        name="TwoPhotonDrive"
    )


    SYSTEMS = [kerr_osc]
    PULSECHAINS = [ps]
    # Run simulation
    sim = QutipPulseSimulator(
        systems=SYSTEMS,
        pulse_chains=PULSECHAINS,
    )
    sim.simulate()

    # Visualize results
    sim.animate_wigner(
        systems=SYSTEMS,
        number_of_frames=200,
        fps=25,
    )

if __name__ == "__main__":
    main()
