from ATS_Hamiltonion import ATS, get_Ec_from_C, newton_minimize
import numpy as np
import pprint as pp    
from tabulate import tabulate

working_spot_3_junctions =  [ -1.1369*np.pi,  3.1369*np.pi]
working_spot_4_junctions = [1.3747*np.pi,0.6253*np.pi]
working_spot_5_junctions = [0.5629*np.pi, 1.4371*np.pi]
sweet_spot = [0,0]
amp_1_alpha_working_spot = 0.4915507857089623
eta = 500e6 # Linear displacement driving strength assuming to be 500MHZ
DISPLACEMNT_PULSE_LENGTH = 10e-9 # 10ns
TRISQUEEZE_PULSE_LENGTH = 60e-9
WAITING_TIME_FOR_KERR_DRIFT = 10e-9
def get_ats_parameter_3_junctions(phi_l,phi_r,max_order=6,AC_max_order=3,N=3,C=1750e-15):
        Ej = 392.54e9 * 2.0 * np.pi
        El = 8.344e9 * 2.0 * np.pi * (392.54/409.6)
        Er = 8.344e9 * 2.0 * np.pi * (392.54/409.6)
        Ec = get_Ec_from_C(C=C)
        N = N
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

def get_ats_parameter_5_junctions(phi_l,phi_r,max_order=6,AC_max_order=3,C = 1750e-15):
        Ej = 607.5e9 * 2.0 * np.pi
        El = 8.344e9 * 2.0 * np.pi * (607.5/409.6)
        Er = 8.344e9 * 2.0 * np.pi * (607.5/409.6)
        Ec = get_Ec_from_C(C=C)
        N = 5
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

def get_ats_parameter_4_junctions(phi_l,phi_r,max_order=6,AC_max_order=3,C = 1750e-15):
        Ej =496.7e9 * 2.0 * np.pi
        El = 8.344e9 * 2.0 * np.pi * (496.7/409.6)
        Er = 8.344e9 * 2.0 * np.pi * (496.7/409.6)
        Ec = get_Ec_from_C(C=C)
        N = 4
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

def show_parameters():


    
    results = []
    
    # N = 5
    dc, acp, acm, omega = get_ats_parameter_5_junctions(*working_spot_5_junctions)
    results.append(["N=5", np.round(omega/(2*np.pi*1e9), 3)] + [np.round(dc[i]/2/np.pi, 3) for i in range(3, 7)] + [np.round(acp[i]/1e6/2/np.pi, 3) for i in range(1, 4)] + [np.round(acm[i]/1e6/2/np.pi, 3) for i in range(1, 4)])
    
    # N = 4
    dc4, acp4, acm4, omega4 = get_ats_parameter_4_junctions(*working_spot_4_junctions)
    results.append(["N=4", np.round(omega4/(2*np.pi*1e9), 3)] + [np.round(dc4[i]/2/np.pi, 3) for i in range(3, 7)] + [np.round(acp4[i]/1e6/2/np.pi, 3) for i in range(1, 4)] + [np.round(acm4[i]/1e6/2/np.pi, 3) for i in range(1, 4)])

    # N = 3
    dc3, acp3, acm3, omega3 = get_ats_parameter_3_junctions(*working_spot_3_junctions)
    results.append(["N=3", np.round(omega3/(2*np.pi*1e9), 3)] + [np.round(dc3[i]/2/np.pi, 3) for i in range(3, 7)] + [np.round(acp3[i]/1e6/2/np.pi, 3) for i in range(1, 4)] + [np.round(acm3[i]/1e6/2/np.pi, 3) for i in range(1, 4)])
    print('================================ATS Parameters================================')
    print(tabulate(results, headers=["N", "Omega/(2pi) (GHz)"] + ["g" + str(i) +"DC/(2pi)" for i in range(3, 7)] + ["g" + str(i) +"AC+/(2pi) (MHz)" for i in range(1, 4)] + ["g" + str(i) +"AC-/(2pi) (MHz)" for i in range(1, 4)]))


show_parameters()