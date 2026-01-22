from quantum_pulse_simulator.simulator import QutipPulseSimulator
from quantum_pulse_simulator.pulse import PulseSequence
from quantum_pulse_simulator.devices import QuantumSystem
from ats_device import get_ats_parameter, sweet_spot, working_spot, amp_1_alpha_working_spot
import pprint as pp
import numpy as np
import matplotlib.pyplot as plt
from calibration_utils import fit_trisqueeze, fit_wigner_trisqueezed_state, calculate_phase_space_metrics
import qutip as qt

dc, ac_plus, ac_minus, omega = get_ats_parameter(*working_spot)
pp.pprint(ac_plus)
pp.pprint(ac_minus)
NUM_FOCK = 100
OSC_FREQ = omega 
TRI_SQUEEZING_TIME = 60e-9*2
SQUEEZING_TIME = 100e-9

def main():
    ats_osc = QuantumSystem(
        num_fock=NUM_FOCK,
        omega=OSC_FREQ,
        name="ATS Oscillator @ KFP" 
    )
    
    ats_osc.add_static_nonlinearities(
        strengths=dc
    )
    
    ps = PulseSequence(systems=[ats_osc])

    ps.add_drive(
        duration = SQUEEZING_TIME,
        strength = ac_minus[2]*0.06,
        system = ats_osc,
        shape = ps.flattop_gaussian_shape(SQUEEZING_TIME, 2e-9, fall=True), 
        phase = 0,
        name = "Drive",
        order = 2
    )

    ps.add_drive(
        duration = TRI_SQUEEZING_TIME,
        strength = ac_plus[3],
        system = ats_osc,
        shape = ps.flattop_gaussian_shape(TRI_SQUEEZING_TIME, 2e-9, fall=True), 
        phase = 0,
        name = "Drive",
        order = 3
    )

    SYSTEMS = [ats_osc]
    PULSECHAINS = [ps]
    sim = QutipPulseSimulator(
        systems=SYSTEMS,
        pulse_chains=PULSECHAINS,
    )
    
    # Run simulation for this amplitude
    sim.simulate(batch_size=1000, save_prefix=f"ATS_cubic_phase", store_batch_file=False)
    
    sim.plot_wigner()


if __name__ == "__main__":
    main()