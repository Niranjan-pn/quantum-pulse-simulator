"""
Author: Niranjan
Date: 11-5-2025

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

from quantum_pulse_simulator.pulse import flattop_gaussian_shape, TwoPhotonDrivePulse, PulseChain,BeamSplitterPulse, WaitingPulse,ControlledCouplingPulse
from quantum_pulse_simulator.devices import KerrOscillator, HarmonicOscillator

# =========================
# System Parameters
# =========================
KERR_COEFF = 6.7e6 * 2 * np.pi  # Kerr nonlinearity (rad/s)
KERR_OSC_FREQ = 6e9 * 2 * np.pi  # Oscillator frequency (rad/s)
KERR_OSC_FREQ2 = 5e9 * 2 * np.pi 
OSC_FREQ = 5e9 * 2 * np.pi  # Oscillator frequency (rad/s)
NUM_FOCK = 10  # Hilbert space dimension
ALPHA_TARGET = 2.6  # Target coherent state amplitude

# =========================
# Pulse Parameters 
# =========================
PULSE_DURATION = 500e-9 #1500E-9  # Total pulse duration (s)
RISE_TIME = 360e-9  # Pulse rise/fall time (s)
PULSE_DURATION_BS = (PULSE_DURATION-2*RISE_TIME)  # Total pulse duration (s)
RISE_TIME_BS = 120e-9  # Pulse rise/fall time (s)

def main():
    """stabilising 4 component cat state"""

    e2_drive = ALPHA_TARGET**2 * KERR_COEFF  # Two-photon drive strength
    
    # Initialize Kerr oscillator 1
    kerr_osc_1 = KerrOscillator(
        num_fock=NUM_FOCK,
        omega=KERR_OSC_FREQ,
        Kerr=KERR_COEFF,
        state=qt.basis(NUM_FOCK, 0),
        
    )

    # Initialize Kerr oscillator 2
    kerr_osc_2 = KerrOscillator(
        num_fock=NUM_FOCK,
        omega=KERR_OSC_FREQ2,
        Kerr=KERR_COEFF,
        state=qt.basis(NUM_FOCK, 0),
    )

    # Initialize Harmonic oscillator
    ho = HarmonicOscillator(
        num_fock=NUM_FOCK,
        omega=OSC_FREQ,
        state=qt.basis(NUM_FOCK, 0),
    )

    SYSTEMS = [kerr_osc_1, ho, kerr_osc_2]
    # Create pulse envelope
    pulse_shape = flattop_gaussian_shape(PULSE_DURATION, RISE_TIME,fall=False)
    pulse_shape_bs = flattop_gaussian_shape(PULSE_DURATION-RISE_TIME, RISE_TIME_BS, fall=False)

    bs1 = BeamSplitterPulse(
        duration = PULSE_DURATION, 
        strength=KERR_COEFF/5, 
        all_systems=SYSTEMS,
        interacting_systems=[kerr_osc_1, kerr_osc_2],
        tone=None, 
        shape=pulse_shape_bs, 
        phase=0.0,
        name="bs1",
    )

    waiting_pulse = WaitingPulse(
        duration=RISE_TIME,
        all_systems=SYSTEMS,
        name ="waiting_pulse",
    )

    squeezing_pulse_1 = TwoPhotonDrivePulse(
        duration=PULSE_DURATION,
        strength=e2_drive,
        sys=kerr_osc_1,
        shape=pulse_shape,
        phase=0.0
    )

    squeezing_pulse_2 = TwoPhotonDrivePulse(
        duration=PULSE_DURATION,
        strength=e2_drive,
        sys=kerr_osc_2,
        shape=pulse_shape,
        phase=np.pi
    )

    

    pulse_chain_1 = PulseChain(system=kerr_osc_1)
    pulse_chain_1.add_pulse(squeezing_pulse_1)

    pulse_chain_2 = PulseChain(system=kerr_osc_2)
    pulse_chain_2.add_pulse(squeezing_pulse_2)

    pulse_chain_bs1 = PulseChain(system=[kerr_osc_1,  kerr_osc_2])
    pulse_chain_bs1.add_pulse(waiting_pulse)
    pulse_chain_bs1.add_pulse(bs1)
    # pulse_chain_bs1.add_pulse(waiting_pulse)

    PULSE_CHAINS = [pulse_chain_1, 
                    pulse_chain_2,
                    pulse_chain_bs1,
                    ]
      # Add waiting period
    
    sim = QutipPulseSimulator(
        systems=SYSTEMS,
        pulse_chains=PULSE_CHAINS,
    )

    # Add beam splitter pulse to the simulation
    

    # Run simulation
    sim.simulate()

    # Visualize results
    sim.animate_wigner(
        systems=SYSTEMS,
        number_of_frames=200,
        fps=25,
        # save_path="4_component_cat.gif",
    )

if __name__ == "__main__":
    main()
