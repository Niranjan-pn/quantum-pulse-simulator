
from quantum_pulse_simulator.simulator import QutipPulseSimulator
from quantum_pulse_simulator.pulse import PulseSequence
from quantum_pulse_simulator.devices import QuantumSystem
from ats_device import get_ats_parameter, sweet_spot, working_spot
import pprint as pp
import numpy as np
import matplotlib.pyplot as plt
from calibration_utils import disp_calib, disp_calib_plot

dc, ac_plus, ac_minus, omega = get_ats_parameter(*working_spot)
# pp.pprint(dc)
# pp.pprint(ac_plus)
# pp.pprint(ac_minus)

#========================
# Simulation setup
#========================
NUM_FOCK = 30
OSC_FREQ = omega 
DISPLACEMNT_PULSE_LENGTH = 10e-9 # 10ns

def run_simulation(amp_scale):
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
        duration = DISPLACEMNT_PULSE_LENGTH,
        strength = ac_plus[1] * amp_scale,
        system = ats_osc,
        shape = ps.flattop_gaussian_shape(DISPLACEMNT_PULSE_LENGTH, 2e-9, fall=True),
        phase = 0.0,
        name = "Drive",
        order = 1
    )

    SYSTEMS = [ats_osc]
    PULSECHAINS = [ps]
    sim = QutipPulseSimulator(
        systems=SYSTEMS,
        pulse_chains=PULSECHAINS,
    )
    # Don't save files for every single point in the sweep to avoid clutter/speed up
    batch_files = sim.simulate(batch_size=1000, save_prefix=f"ATS_Displacement_{amp_scale:.4f}", store_batch_file=False)
    
    # Get final state
    final_state = sim.result.states[-1]
    
    # Calculate populations
    # Trace out if there were other systems (here only one)
    # But sim.result.states[-1] is already the full state
    # We want population of Fock |0> and |1>
    
    P0 = np.abs(final_state.overlap(ats_osc.state))**2 # ground state is initial state usually
    # Better: explicit projection
    rho = final_state # It's a ket if using sesolve
    
    # Careful: simulator returns Qobj states.
    # Check if sesolve or mesolve. ats_device has no c_ops so likely sesolve -> kets
    
    # Project onto |0>
    basis_0 = ats_osc.state # This is initialized as basis(N, 0) in QuantumSystem
    # Project onto |1>
    import qutip as qt
    basis_1 = qt.basis(NUM_FOCK, 1)
    
    pop0 = qt.expect(qt.ket2dm(basis_0), rho)
    pop1 = qt.expect(qt.ket2dm(basis_1), rho)
    
    return pop0, pop1

def main(): 
    print("Starting displacement calibration sweep...")
    
    # Sweep parameters
    amp_scales = np.linspace(0.0, 0.3, 15) 
    pop0_list = []
    pop1_list = []

    for scale in amp_scales:
        print(f"Simulating amp_scale = {scale:.4f}")
        p0, p1 = run_simulation(scale)
        pop0_list.append(p0)
        pop1_list.append(p1)

    xdata = amp_scales
    ydata = np.array([pop0_list, pop1_list])

    # Fit
    print("Fitting data...")
    fit_result = disp_calib(xdata, ydata)
    
    # Plot
    plt.figure(figsize=(10, 6))
    disp_calib_plot(xdata, ydata, fit_result, title_extension="ATS Displacement Calibration\n")
    
    # Calculate amplitude for alpha=1
    # The fit return Ascale (fit.x[1]). 
    # Argument to funcD is (Ascale * x).
    # For alpha=1, we want effective displacement to be 1.
    # In funcD: exp(-((Ascale * x) ** 2)) corresponds to exp(-|alpha|^2)
    # So alpha = Ascale * x
    # To get alpha=1, we need x = 1 / Ascale
    
    Ascale = fit_result.x[1]
    amp_scale_for_alpha_1 = 1.0 / Ascale
    
    print(f"\n calibration result:")
    print(f"  Fitted Ascale: {Ascale}")
    print(f"  Amp scale for alpha=1: {amp_scale_for_alpha_1}")
    print(f"  Actual Drive Strength needed: {ac_plus[1] * amp_scale_for_alpha_1}")
    
    plt.axvline(amp_scale_for_alpha_1, color='r', linestyle='--', label=f'alpha=1 @ scale={amp_scale_for_alpha_1:.4f}')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()