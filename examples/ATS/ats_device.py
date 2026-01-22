from ATS_Hamiltonion import ATS, get_Ec_from_C, newton_minimize
import numpy as np
import pprint as pp

working_spot =  [ -1.14*np.pi, 1.14*np.pi]
sweet_spot = [0,0]
amp_1_alpha_working_spot = 0.4915507857089623
eta = 500e6 # Linear displacement driving strength assuming to be 500MHZ
DISPLACEMNT_PULSE_LENGTH = 10e-9 # 10ns

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