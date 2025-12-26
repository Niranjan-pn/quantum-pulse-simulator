from ATS_Hamiltonion import ATS, get_Ec_from_C, newton_minimize
import numpy as np
import pprint as pp

from quantum_pulse_simulator.simulator import QutipPulseSimulator
from quantum_pulse_simulator.pulse import PulseSequence
from quantum_pulse_simulator.devices import QuantumSystem
def get_ats_parameter(phi_l,phi_r,max_order=6,AC_max_order=3):
        Ej = 409.6e9 * 2.0 * np.pi
        El = 8.344e9 * 2.0 * np.pi
        Er = 8.344e9 * 2.0 * np.pi
        Ec = get_Ec_from_C(C=1750e-15)
        N = 3
        ats_pure = ATS(
            Ej=Ej, El=El, Er=Er, Ec=Ec, N=N,
            T1=200e-6
        )
        phi_plus, phi_minus = ats_pure.get_phi_plus_minus(phi_l, phi_r)
        phi_min = newton_minimize(ats_pure.U_plus_minus_basis, phi_plus, phi_minus)
        parameters_dc = {}
        for i in range(3,max_order+1):
            parameters_dc[i] = ats_pure.gi(order = i, 
            phi_min = phi_min, 
            phi_plus = phi_plus, 
            phi_minus = phi_minus).item()
        parameters_ac_plus = {}
        for i in range(1, AC_max_order+1):
            parameters_ac_plus[i] = ats_pure.gi_ac_plus(i = i, 
            phi_min = phi_min, 
            phi_plus = phi_plus, 
            phi_minus = phi_minus).item()
        parameters_ac_minus = {}
        for i in range(1, AC_max_order+1):
            parameters_ac_minus[i] = ats_pure.gi_ac_minus(i = i, 
            phi_min = phi_min, 
            phi_plus = phi_plus, 
            phi_minus = phi_minus).item()

        omega = ats_pure.omega_renormalised(phi_min, phi_plus, phi_minus)
        return parameters_dc, parameters_ac_plus, parameters_ac_minus, omega

phi_l = -1.14*np.pi
phi_r = 1.14*np.pi

dc, ac_plus, ac_minus, omega = get_ats_parameter(phi_l,phi_r)
pp.pprint(dc)
pp.pprint(ac_plus)
pp.pprint(ac_minus)
phi_l = 0.4*np.pi
phi_r = 0
dc2,ac_plus2,ac_minus2,omega2 = get_ats_parameter(phi_l,phi_r,AC_max_order=6)
pp.pprint(dc2)
pp.pprint(ac_plus2)
pp.pprint(ac_minus2)
#========================
# Simulation
#========================
NUM_FOCK = 10
OSC_FREQ = omega 
OSC_FREQ2 = omega2

def main():
    ats_osc = QuantumSystem(
        num_fock=NUM_FOCK,
        omega=OSC_FREQ,
        name="ATS Oscillator @ KFP" 
    )

    ats_osc.add_static_nonlinearities(
        strengths=dc
    )

    ats_osc2 = QuantumSystem(
        num_fock=NUM_FOCK,
        omega=OSC_FREQ2,
        name="ATS Oscillator @ away KFP"
    )
    ats_osc2.add_static_nonlinearities(
        strengths=dc2
    )
    ps = PulseSequence(systems=[ats_osc])
    ps.add_drive(
        duration = 100e-9,
        strength = ac_plus[3]*2,
        system = ats_osc,
        shape = ps.flattop_gaussian_shape(50e-9,10e-9, fall=True),
        phase = 0.0,
        name = "Drive",
        order = 3
    )

    ps2 = PulseSequence(systems=[ats_osc2])
    ps2.add_drive(
        duration = 100e-9,
        strength = ac_plus[3]*2,
        system = ats_osc2,
        shape = ps.flattop_gaussian_shape(50e-9,10e-9, fall=True),
        phase = 0.0,
        name = "Drive",
        order = 3
    )

    SYSTEMS = [ats_osc,ats_osc2]
    PULSECHAINS = [ps,ps2]
    sim = QutipPulseSimulator(
        systems=SYSTEMS,
        pulse_chains=PULSECHAINS,
    )
    sim.simulate(save_dir=r"E:\codes\quantum-pulse-simulator\quantum-pulse-simulator\examples\ATS",
                 batch_size=1000,
                 save_prefix="ATS_Trisqeezing",)

    sim.animate_wigner(
        systems=SYSTEMS,
        number_of_frames=200,
        fps=25,
        save=False
        
    )
    sim.plot_wigner(system_index=[0,1])

if __name__ == "__main__":
    main()
