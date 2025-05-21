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

from quantum_pulse_simulator.pulse import PulseSequence
from quantum_pulse_simulator.devices import QuantumSystem
qt.CoreOptions.default_dtype = "jax"
# =========================
# System Parameters
# =========================
KERR_COEFF = 6.7e6 * 2 * np.pi  # Kerr nonlinearity (rad/s)
KERR_OSC_FREQ = 6e9 * 2 * np.pi  # Oscillator frequency (rad/s)
OSC_FREQ = 5e9 * 2 * np.pi  # Oscillator frequency (rad/s)
NUM_FOCK = 10  # Hilbert space dimension
ALPHA_TARGET = 2.6  # Target coherent state amplitude

# =========================
# Pulse Parameters 
# =========================
PULSE_DURATION = 2000e-9 #1500E-9  # Total pulse duration (s)
RISE_TIME = 320e-9  # Pulse rise/fall time (s)
PULSE_DURATION_BS = PULSE_DURATION-RISE_TIME  # Total pulse duration (s)
RISE_TIME_BS = 320e-9  # Pulse rise/fall time (s)

def main():
    """stabilising 4 component cat state"""

    e2_drive = ALPHA_TARGET**2 * KERR_COEFF  # Two-photon drive strength
    
    #initialize Kerr oscillator 1
    Kerr_osc_1 = QuantumSystem(
        num_fock=NUM_FOCK,
        omega=KERR_OSC_FREQ,
        name="Kerr Oscillator 1",

    )

    Kerr_osc_1.add_kerr_oscillator(KERR_COEFF)

    # Initialize Kerr oscillator 2
    Kerr_osc_2 = QuantumSystem(
        num_fock=NUM_FOCK,
        omega=KERR_OSC_FREQ,
        name="Kerr Oscillator 2",
    )
    Kerr_osc_2.add_kerr_oscillator(KERR_COEFF)

    # Initialize Harmonic oscillator
    ho = QuantumSystem(
        num_fock=NUM_FOCK,
        omega=OSC_FREQ,
        name="Harmonic Oscillator"
    )

    SYSTEMS = [Kerr_osc_1, Kerr_osc_2, ho]

    # ===================Pulse sequence setup=================

    # Kerr oscillator 1 pulse sequence
    ps_1 = PulseSequence(systems=[Kerr_osc_1])
    ps_1.add_two_photon_drive(
        duration=PULSE_DURATION,
        strength=e2_drive,
        system=Kerr_osc_1,
        shape=ps_1.flattop_gaussian_shape(PULSE_DURATION, RISE_TIME,fall=False),
        phase=0.0,
        name="TwoPhotonDrive"
    )


    # Kerr oscillator 2 pulse sequence
    ps_2 = PulseSequence(systems=[Kerr_osc_2])
    ps_2.add_two_photon_drive(
        duration=PULSE_DURATION,
        strength=e2_drive,
        system=Kerr_osc_2,
        shape=ps_2.flattop_gaussian_shape(PULSE_DURATION, RISE_TIME,fall=False),
        phase=np.pi,
        name="TwoPhotonDrive"
    )

    # # Beam splitter pulse sequence
    # ps_bs_1 = PulseSequence(systems=SYSTEMS)
    # ps_bs_1.add_waiting(
    #     duration=RISE_TIME,
    #     name="Waiting",
    # )
    # ps_bs_1.add_second_order_beamsplitter(
    #     duration=PULSE_DURATION_BS,
    #     strength=KERR_COEFF/10,
    #     system1=Kerr_osc_1,
    #     system2=Kerr_osc_2,
    #     system3=ho,
    #     shape=ps_bs_1.flattop_gaussian_shape(PULSE_DURATION_BS, RISE_TIME_BS,fall=False),
    #     phase=0.0,
    #     name="K1_K2_HO_BS",
    # )

    ps_bs_1 = PulseSequence(systems=SYSTEMS)
    ps_bs_1.add_waiting(
        duration=RISE_TIME,
        name="Waiting",
    )
    #add first beam splitter

    ps_bs_1.add_beamsplitter(
        duration=PULSE_DURATION_BS,
        strength=KERR_COEFF/10,
        system1=Kerr_osc_1,
        system2=ho,
        shape=ps_bs_1.flattop_gaussian_shape(PULSE_DURATION_BS, RISE_TIME_BS,fall=False),
        phase=0.0,
        name="K1_HO_BS",
    )

    #add second beam splitter
    ps_bs_2 = PulseSequence(systems=SYSTEMS)
    ps_bs_2.add_waiting(
        duration=RISE_TIME,
        name="Waiting",
    )
    ps_bs_2.add_beamsplitter(
        duration=PULSE_DURATION_BS,
        strength=KERR_COEFF/10,
        system1=Kerr_osc_2,
        system2=ho,
        shape=ps_bs_2.flattop_gaussian_shape(PULSE_DURATION_BS, RISE_TIME_BS,fall=False),
        phase=0.0,
        name="K2_HO_BS",
    )

    # Combine all systems into a list

    PULSE_CHAINS = [ps_1,ps_2,ps_bs_1,ps_bs_2]

    sim = QutipPulseSimulator(
        systems=SYSTEMS,
        pulse_chains=PULSE_CHAINS,
    )

    # Add beam splitter pulse to the simulation
    
    # sim.plot_pulse_sequence()
    # Run simulation
    sim.simulate(save_dir=r"D:\PhD_All_Code\quantum-pulse-simulator\examples\results\four_component_bs_cat",
                 batch_size=1000,
                 save_prefix="4_component_cat_bs",)


    # Visualize results
    sim.animate_wigner(
        systems=SYSTEMS,
        number_of_frames=200,
        fps=25,
        save_path=r"D:\PhD_All_Code\quantum-pulse-simulator\examples\results\four_component_bs_cat\four_component_cat_linear_bs.gif",
    )

if __name__ == "__main__":
    main()
