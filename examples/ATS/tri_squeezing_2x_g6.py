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
pp.pprint(dc)
pp.pprint(ac_plus)
pp.pprint(ac_minus)
NUM_FOCK = 100
OSC_FREQ = omega 

def main():
    
    # Sweep parameters
    flux_amplitudes = np.linspace(0, 1, 10)
    fixed_duration = 60e-9
    
    fidelities = []
    tri_squeezing = []
    angles = []
    r99_list = []
    v_neg_list = []
    wigners = [] 
    
    print(f"Starting Tri-Squeezing Amplitude Sweep (Duration: {fixed_duration*1e9:.1f} ns)...")
    
    for flux_amp in flux_amplitudes:
        print(f"Simulating Flux Amplitude = {flux_amp:.4f}")
        
        ats_osc = QuantumSystem(
            num_fock=NUM_FOCK,
            omega=OSC_FREQ,
            name="ATS Oscillator @ KFP (2x g6)" 
        )

        ats_osc.add_static_nonlinearities(
            strengths=dc
        )

        ps = PulseSequence(systems=[ats_osc])

        ps.add_drive(
            duration = fixed_duration,
            strength = ac_plus[3] * flux_amp,
            system = ats_osc,
            shape = ps.flattop_gaussian_shape(fixed_duration, 2e-9, fall=True), 
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
        sim.simulate(batch_size=1000, save_prefix=f"ATS_Trisq_Amp_{flux_amp:.2f}", store_batch_file=False)
        
        # Analyze final state
        rho = sim.result.states[-1]
        
        if rho.isket:
            rho_dm = qt.ket2dm(rho)
        else:
            rho_dm = rho

        # Wigner range might need adjustment if displacement is large
        xvec = np.linspace(-7, 7, 80)
        yvec = np.linspace(-7, 7, 80)
        
        W = qt.wigner(rho_dm, xvec, yvec)
        wigners.append(W)
        
        # Fidelity calculation
        t_param, fid = fit_trisqueeze(W, xvec, yvec, NUM_FOCK)
        db, angle = fit_wigner_trisqueezed_state(
            wigner_data=W, 
            alphax=xvec, 
            alphay=yvec, 
            cavity_dim=NUM_FOCK, 
            plot=False
        )
        
        # Phase Space Support Metrics
        r99, v_neg,total_trace = calculate_phase_space_metrics(W, xvec, yvec)
        
        print(f"  Fidelity: {fid:.4f}")
        print(f"  Tri-squeezing: {db:.4f} dB")
        print(f"  R99: {r99:.4f}")
        print(f"  V_neg: {v_neg:.4f}")
        print(f"  Trace: {total_trace:.4f}")
        
        fidelities.append(fid)
        tri_squeezing.append(db)
        angles.append(angle)
        r99_list.append(r99)
        v_neg_list.append(v_neg)

    product_list = [r99_list[i] * fidelities[i] for i in range(len(r99_list))]

    # Plot Wigners in 4x3 grid (first 12 points)
    print("Plotting Wigner grid (first 12 points)...")
    fig_wig, axes = plt.subplots(4, 3, figsize=(15, 20))
    axes = axes.flatten()
    
    for i, W_data in enumerate(wigners):
        if i >= len(axes): break
        ax = axes[i]
        im = ax.imshow(W_data, origin='lower', extent=[xvec[0], xvec[-1], yvec[0], yvec[-1]], cmap='RdBu', vmin=-1, vmax=1)
        ax.set_title(f"Flux: {flux_amplitudes[i]:.2f}\nFid: {fidelities[i]:.3f}, SQ: {tri_squeezing[i]:.1f}dB\nR99: {r99_list[i]:.2f}")
        ax.set_xlabel('Re(alpha)')
        ax.set_ylabel('Im(alpha)')
        
    plt.tight_layout()
    plt.show()

    # Plot Metrics vs Flux Amplitude
    fig, ax = plt.subplots(2, 2, figsize=(16, 12))
    
    # Fidelity
    ax[0, 0].plot(flux_amplitudes, fidelities, 'o-', linewidth=2)
    ax[0, 0].set_xlabel('Flux Amplitude', fontsize=14)
    ax[0, 0].set_ylabel('Fidelity', fontsize=14)
    ax[0, 0].set_title('Fidelity vs. Flux Amplitude', fontsize=16)
    ax[0, 0].grid(True)
    
    # Tri-Squeezing
    ax[0, 1].plot(flux_amplitudes, tri_squeezing, 'o-', linewidth=2, color='orange')
    ax[0, 1].set_xlabel('Flux Amplitude', fontsize=14)
    ax[0, 1].set_ylabel('Tri-Squeezing (dB)', fontsize=14)
    ax[0, 1].set_title('Tri-Squeezing vs. Flux Amplitude', fontsize=16)
    ax[0, 1].grid(True)
    
    # R99 Support
    ax[1, 0].plot(flux_amplitudes, r99_list, 'o-', linewidth=2, color='green')
    ax[1, 0].set_xlabel('Flux Amplitude', fontsize=14)
    ax[1, 0].set_ylabel('R99 Support', fontsize=14)
    ax[1, 0].set_title('Phase Space Support (R99) vs. Flux Amplitude', fontsize=16)
    ax[1, 0].grid(True)
    
    # Negativity Volume
    ax[1, 1].plot(flux_amplitudes, product_list, 'o-', linewidth=2, color='red')
    ax[1, 1].set_xlabel('Flux Amplitude', fontsize=14)
    ax[1, 1].set_ylabel('R99 * Fidelity', fontsize=14)
    ax[1, 1].set_title('R99 * Fidelity vs. Flux Amplitude', fontsize=16)
    ax[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

    # Find max fidelity
    max_fid = max(fidelities)
    best_amp = flux_amplitudes[np.argmax(fidelities)]
    print(f"\nBest Fidelity: {max_fid:.4f} at Flux Amplitude {best_amp:.4f}")

if __name__ == "__main__":
    main()
