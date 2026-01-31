from quantum_pulse_simulator.simulator import QutipPulseSimulator
from quantum_pulse_simulator.pulse import PulseSequence
from quantum_pulse_simulator.devices import QuantumSystem
from snail_device import dc,OMEGA,g3AC,eta,DISPLACEMNT_PULSE_LENGTH
import pprint as pp
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime


#========================
# Experiment imports
#========================
from quantum_pulse_simulator.experiments.displacement_calibration import run_displacement_calibration, DisplacementCalibrationSettings
from quantum_pulse_simulator.experiments.kerr_drift_angle import run_kerr_drift_angle, KerrDriftAngleSettings
from quantum_pulse_simulator.experiments.displace_and_plot import run_displace_and_plot, DisplaceAndPlotSettings
#========================
# Device setup
#========================
NUM_FOCK = 30
OSC_FREQ = OMEGA 
snail_osc = QuantumSystem(
        num_fock=NUM_FOCK,
        omega=OSC_FREQ,
        name="SNAIL Oscillator @ KFP" 
    )

snail_osc.add_static_nonlinearities(dc)
year = datetime.now().year
month = datetime.now().month
day = datetime.now().day

data_save_path = f"examples/SNAIL/sim_results/Data/{year}/{month}/{day}"
calibration_database_path = "examples/SNAIL/calibration_database.json"

if not os.path.exists(data_save_path):
    os.makedirs(data_save_path)

if not os.path.exists(calibration_database_path):
    raise FileNotFoundError("Calibration database not found")
#========================
# Experiment flags
#========================
DISPLACEMENT_CALIBRATION = False    
KERR_DRIFT_ANGLE = True   
SINGLE_DRIFT_ANGLE = False 
DISPLACE_AND_PLOT = False
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
    amp_scale_for_alpha_1 = run_displacement_calibration(snail_osc, settings, data_save_path, calibration_database_path, show_plot=False)

if KERR_DRIFT_ANGLE:
    data = {
        
    }
    probe_alpha = np.linspace(0.1, 2, 20)
    drift_angle_list = []
    fig, ax = plt.subplots(4,5,figsize=(10,10))
    ax = ax.flatten()
    calibration_database = load_calibration_database(calibration_database_path)
    settings = KerrDriftAngleSettings()
    settings.DISPLACEMENT_PULSE_LENGTH = DISPLACEMNT_PULSE_LENGTH
    settings.DISPLACEMENT_ETA = eta
    settings.wait_time = 10e-9
    settings.amp_scale_for_alpha_1 = calibration_database["amp_scale_for_alpha_1"]
    settings.probe_angles_range = (0, 4*np.pi)
    settings.probe_angles_num = 30
    dt =datetime.now().strftime('%Y%m%d_%H%M%S')
    for i, alpha in enumerate(probe_alpha):
        settings.alpha = alpha
        peak_position, peak_width, probe_angles, pop0_list = run_kerr_drift_angle(snail_osc, settings, data_save_path, calibration_database_path, show_plot=False)
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
        with open(f"{data_save_path}/drift_angles_vs_alpha_{dt}.json", "w") as f:
            json.dump(data, f)
    plt.tight_layout()
    plt.savefig(f"{data_save_path}/vacuum_population_vs_probe_angle_{dt}.png")
    plt.show()

    

    plt.plot(probe_alpha, drift_angle_list, 'o')
    plt.xlabel("Alpha")
    plt.ylabel("Drift Angle (rad)")
    plt.title("Drift Angle vs Alpha")
    plt.savefig(f"{data_save_path}/drift_angle_vs_alpha_{dt}.png")
    plt.show()

if SINGLE_DRIFT_ANGLE:
    calibration_database = load_calibration_database(calibration_database_path)
    settings = KerrDriftAngleSettings()
    settings.DISPLACEMENT_PULSE_LENGTH = DISPLACEMNT_PULSE_LENGTH
    settings.DISPLACEMENT_ETA = eta
    settings.wait_time = 10e-9
    settings.amp_scale_for_alpha_1 = calibration_database["amp_scale_for_alpha_1"]
    settings.probe_angles_range = (0, 4*np.pi)
    settings.probe_angles_num = 30
    settings.alpha = 0.5
    peak_position, peak_width, probe_angles, pop0_list = run_kerr_drift_angle(snail_osc, settings, data_save_path, calibration_database_path, show_plot=False)
    
if DISPLACE_AND_PLOT:
    calibration_database = load_calibration_database(calibration_database_path)
    settings = DisplaceAndPlotSettings()
    settings.pulse_length = DISPLACEMNT_PULSE_LENGTH
    settings.eta = eta
    settings.alpha = 2
    settings.waiting_time = 20e-9
    settings.xlim = (-10, 10)
    settings.amp_scale_for_alpha_1 = calibration_database["amp_scale_for_alpha_1"]
    run_displace_and_plot(snail_osc, settings, data_save_path, calibration_database_path, show_plot=False)
