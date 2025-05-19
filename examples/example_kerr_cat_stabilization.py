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

from quantum_pulse_simulator.pulse import flattop_gaussian_shape, TwoPhotonDrivePulse, PulseChain
from quantum_pulse_simulator.devices import KerrOscillator

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
    kerr_osc = KerrOscillator(
        num_fock=NUM_FOCK,
        omega=OSC_FREQ,
        Kerr=KERR_COEFF,
        state=qt.basis(NUM_FOCK, 1)
    )

    # Create pulse envelope
    pulse_shape = flattop_gaussian_shape(PULSE_DURATION, RISE_TIME)

    # Configure two-photon drive pulse
    squeezing_pulse = TwoPhotonDrivePulse(
        duration=PULSE_DURATION,
        strength=e2_drive,
        sys=kerr_osc,
        shape=pulse_shape,
        phase=0.0
    )

    squeezing_pulse2 = TwoPhotonDrivePulse(
        duration=PULSE_DURATION,
        strength=e2_drive,
        theta=np.pi,
        sys=kerr_osc,
        shape=pulse_shape,
        phase=np.pi
    )


    # Build pulse sequence
    pulse_chain = PulseChain(system=kerr_osc)
    pulse_chain.add_pulse(squeezing_pulse)
    pulse_chain.add_empty_pulse(duration=100e-9)  # Add waiting period

    pulse_chain2 = PulseChain(system=kerr_osc)
    pulse_chain2.add_pulse(squeezing_pulse2)
    pulse_chain2.add_empty_pulse(duration=100e-9)  # Add waiting period

    SYSTEMS = [kerr_osc]
    PULSECHAINS = [pulse_chain,pulse_chain2]
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
