from quantum_pulse_simulator.simulator import QutipPulseSimulator
from quantum_pulse_simulator.pulse import PulseSequence
from quantum_pulse_simulator.devices import QuantumSystem
from ats_device import get_ats_parameter_5_junctions, working_spot_5_junctions, eta, DISPLACEMNT_PULSE_LENGTH, TRISQUEEZE_PULSE_LENGTH, WAITING_TIME_FOR_KERR_DRIFT
import pprint as pp
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
dc, ac_plus, ac_minus, omega = get_ats_parameter_5_junctions(*working_spot_5_junctions)
import os
#========================
# Experiment imports
#========================
from quantum_pulse_simulator.experiments.displacement_calibration import run_displacement_calibration, DisplacementCalibrationSettings
from quantum_pulse_simulator.experiments.kerr_drift_angle import run_kerr_drift_angle, KerrDriftAngleSettings
from quantum_pulse_simulator.experiments.displace_and_plot import run_displace_and_fit_coherent, DisplaceAndFitCoherentSettings
from quantum_pulse_simulator.experiments.trisqueeze_and_plot import run_trisqueeze_and_fit, TrisqueezeandFitSettings
#========================
# Device setup
#========================
NUM_FOCK = 100
OSC_FREQ = omega
ats_osc = QuantumSystem(
        num_fock=NUM_FOCK,
        omega=OSC_FREQ,
        name="ATS Oscillator 5J " 
    )

ats_osc.add_static_nonlinearities(
    strengths=dc
)


year = datetime.now().year
month = datetime.now().month
day = datetime.now().day


data_save_path = f"examples/ATS/sim_results/Data/{year}/{month}/{day}"
calibration_database_path = "examples/ATS/calibration_database_5j.json"

if not os.path.exists(data_save_path):
    os.makedirs(data_save_path)

if not os.path.exists(calibration_database_path):
    raise FileNotFoundError("Calibration database not found")
#========================
# Experiment flags
#========================
DISPLACEMENT_CALIBRATION = False    
KERR_DRIFT_ANGLE = False    
DISPLACE_AND_PLOT = True   
TRISQUEEZE_AND_PLOT = False
TRISQUEEZE_AND_PLOT_ETA_VS_PULSE_LENGTH = False
#========================
# Experiment functions
#========================
def load_calibration_database(calibration_database_path):
    with open(calibration_database_path, 'r') as f:
        calibration_database = json.load(f)
    return calibration_database

if DISPLACEMENT_CALIBRATION:
    settings = DisplacementCalibrationSettings()
    settings.PULSE_LENGTH = DISPLACEMNT_PULSE_LENGTH
    settings.eta = eta
    amp_scale_for_alpha_1 = run_displacement_calibration(ats_osc, settings, data_save_path, calibration_database_path, show_plot=False)

if KERR_DRIFT_ANGLE:
    data = {
        
    }
    probe_alpha = np.linspace(0.2, 2, 10)
    drift_angle_list = []
    fig, ax = plt.subplots(2,5,figsize=(10,10))
    ax = ax.flatten()
    calibration_database = load_calibration_database(calibration_database_path)
    settings = KerrDriftAngleSettings()
    settings.DISPLACEMENT_PULSE_LENGTH = DISPLACEMNT_PULSE_LENGTH
    settings.DISPLACEMENT_ETA = eta
    settings.wait_time = 10e-9
    settings.amp_scale_for_alpha_1 = calibration_database["amp_scale_for_alpha_1"]
    settings.probe_angles_range = (0,4*np.pi)
    settings.probe_angles_num = 15
    date_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    for i, alpha in enumerate(probe_alpha):
        settings.alpha = alpha
        peak_position, peak_width, probe_angles, pop0_list = run_kerr_drift_angle(ats_osc, settings, data_save_path, calibration_database_path, show_plot=False)
        drift_angle_list.append(peak_position)

        ax[i].plot(probe_angles/np.pi, pop0_list, 'o')
        ax[i].set_xlabel("Probe Angle (pi)")
        ax[i].set_ylabel("Vacuum Population")
        ax[i].set_title(f"Vacuum Population vs Probe Angle (Alpha={alpha:.2f})")
    
        data[alpha] = {
            "probe_angles": probe_angles.tolist(),
            "pop0_list": pop0_list,
            "peak_position": peak_position,
            "peak_width": peak_width,
        }
        with open(f"{data_save_path}/drift_angles_vs_alpha_{date_time}.json", "w") as f:
            json.dump(data, f)
    plt.tight_layout()
    plt.savefig(f"{data_save_path}/vacuum_population_vs_probe_angle_{date_time}.png")
    plt.show()

    

    plt.plot(probe_alpha, drift_angle_list, 'o')
    plt.xlabel("Alpha")
    plt.ylabel("Drift Angle (rad)")
    plt.title("Drift Angle vs Alpha")
    plt.savefig(f"{data_save_path}/drift_angle_vs_alpha_{date_time}.png")
    plt.show()
    
if DISPLACE_AND_PLOT:
    calibration_database = load_calibration_database(calibration_database_path)
    settings = DisplaceAndFitCoherentSettings()
    settings.pulse_length = DISPLACEMNT_PULSE_LENGTH
    settings.eta = eta
    
    settings.waiting_time = WAITING_TIME_FOR_KERR_DRIFT
    settings.xlim = (-10, 10)
    settings.amp_scale_for_alpha_1 = calibration_database["amp_scale_for_alpha_1"]

    alpha_list = np.linspace(0.2, 4, 20)
    drift_angle_list = []
    date_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    data = {}
    for alpha in alpha_list:
        settings.alpha = alpha
        result = run_displace_and_fit_coherent(ats_osc, settings, data_save_path, calibration_database_path, show_plot=False)
        drift_angle_list.append(result['phase'])
        data[alpha] = {
            "phase": result['phase'],
        }
    
    with open(f"{data_save_path}/drift_angles_vs_alpha_N5_{date_time}.json", "w") as f:
        json.dump(data, f)

    delta_drift_angle = [i - drift_angle_list[0] for i in drift_angle_list]
    plt.plot(alpha_list, delta_drift_angle, '-o')
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\Delta \theta$ (rad)")
    plt.title(r"$\Delta \theta$ vs $\alpha$")
    plt.savefig(f"{data_save_path}/delta_drift_angle_vs_alpha_N5_{date_time}.png")
    plt.show()

if TRISQUEEZE_AND_PLOT_ETA_VS_PULSE_LENGTH:
    eta_list = np.linspace(0.1, 0.9, 9)
    pulse_length_list = np.linspace(50e-9, 400e-9, 9)
    data = {}
    date_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    fidility_array = np.zeros((len(eta_list), len(pulse_length_list)))
    db_array = np.zeros((len(eta_list), len(pulse_length_list)))
    angle_array = np.zeros((len(eta_list), len(pulse_length_list)))
    for i,eta in enumerate(eta_list):
        data[eta] = {}
        for j,pulse_length in enumerate(pulse_length_list):
            settings = TrisqueezeandFitSettings()
            settings.pulse_length = pulse_length
            settings.g3AC = ac_plus[3]
            settings.eta = eta
            settings.waiting_time = 0e-9
            settings.xlim = (-10, 10)
            
            rho, photon_number_dist, fid, db, angle = run_trisqueeze_and_fit(
                ats_osc, settings, data_save_path, show_plot=False
            )
            fidility_array[i,j] = fid
            db_array[i,j] = db
            angle_array[i,j] = angle
            data={

                "eta": eta_list,
                "pulse_length": pulse_length_list,
                "fid": fidility_array,
                "db": db_array,
                "angle": angle_array
            }
            
            with open(f"{data_save_path}/{ats_osc.name}_trisqueeze_data_{date_time}.json", "w") as f:
                json.dump(data, f)

            fig,ax=plt.subplots(1,3,figsize=(15,5))
            extent = [pulse_length_list[0], pulse_length_list[-1], eta_list[0], eta_list[-1]]
            
            im0 = ax[0].imshow(fidility_array, extent=extent, origin='lower', aspect='auto')
            ax[0].set_xlabel("Pulse Length (s)")
            ax[0].set_ylabel("eta")
            ax[0].set_title("Fidelity")
            fig.colorbar(im0, ax=ax[0])
            
            im1 = ax[1].imshow(db_array, extent=extent, origin='lower', aspect='auto')
            ax[1].set_xlabel("Pulse Length (s)")
            ax[1].set_ylabel("eta")
            ax[1].set_title("Tri-squeezing")
            fig.colorbar(im1, ax=ax[1])
            
            im2 = ax[2].imshow(angle_array, extent=extent, origin='lower', aspect='auto')
            ax[2].set_xlabel("Pulse Length (s)")
            ax[2].set_ylabel("eta")
            ax[2].set_title("Angle")
            fig.colorbar(im2, ax=ax[2])
            
            fig.tight_layout()
            plt.savefig(f"{data_save_path}/{ats_osc.name}_trisqueeze_data_{date_time}.png")
    plt.show()

if TRISQUEEZE_AND_PLOT:
    settings = TrisqueezeandFitSettings()
    settings.pulse_length = 270e-9
    settings.g3AC = ac_plus[3]
    settings.eta = 0.6
    settings.waiting_time = 0e-9
    settings.xlim = (-10, 10)
    
    rho, photon_number_dist, fid, db, angle = run_trisqueeze_and_fit(
        ats_osc, settings, data_save_path, show_plot=False
    )
    
    print(f"Fidelity: {fid:.4f}")
    print(f"Tri-squeezing: {db:.4f} dB")
    print(f"Trisqeezing angle : {angle:.4f} rad")

    