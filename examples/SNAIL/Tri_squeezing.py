from quantum_pulse_simulator.simulator import QutipPulseSimulator
from quantum_pulse_simulator.pulse import PulseSequence
from quantum_pulse_simulator.devices import QuantumSystem
from snail_device import dc, OMEGA, g3AC
import pprint as pp
import numpy as np
import matplotlib.pyplot as plt
from calibration_utils import fit_trisqueeze, fit_wigner_trisqueezed_state, calculate_phase_space_metrics
import qutip as qt

NUM_FOCK = 100
OSC_FREQ = OMEGA 
ALPHA_TARGET = 22

def main():
    
    # Pulse length parameters
    max_duration = 100e-9
    num_points = 20
    # We want to analyze states at these specific times
    target_times = np.linspace(0, max_duration, num_points)
    # Remove t=0 if it causes issues, or handle it (state is vacuum)
    if target_times[0] == 0:
        target_times[0] = 1e-12 
        
    fidelities = []
    tri_squeezing = []
    angles = []
    r99_list = []
    v_neg_list = []
    wigners = [] 
    
    print(f"Starting Tri-Squeezing Single Simulation (Max duration: {max_duration*1e9:.1f} ns)...")
    
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
        duration = max_duration,
        strength = g3AC * ALPHA_TARGET,
        system = ats_osc,
        shape = ps.flattop_gaussian_shape(max_duration, 2e-9, fall=True), 
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
    
    # Run ONE simulation
    sim.simulate(batch_size=2000, save_prefix="ATS_Trisq", store_batch_file=False)
    
    # Analyze states at target times
    for t_target in target_times:
        # Find closest index
        idx = np.argmin(np.abs(sim.tlist - t_target))
        actual_time = sim.tlist[idx]
        print(f"Analyzing t = {actual_time*1e9:.2f} ns (Target: {t_target*1e9:.2f} ns)")
        
        rho = sim.result.states[idx]
        
        if rho.isket:
            rho_dm = qt.ket2dm(rho)
        else:
            rho_dm = rho

        xvec = np.linspace(-15, 15, 200)
        yvec = np.linspace(-15, 15, 200)
        
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
        r99, v_neg, trace = calculate_phase_space_metrics(W, xvec, yvec)
        
        print(f"  Fidelity: {fid:.4f}")
        print(f"  Tri-squeezing: {db:.4f} dB")
        print(f"  R99: {r99:.4f}")
        print(f"  V_neg: {v_neg:.4f}")
        print(f"  Trace: {trace:.4f}")
        
        fidelities.append(fid)
        tri_squeezing.append(db)
        angles.append(angle)
        r99_list.append(r99)
        v_neg_list.append(v_neg)

    product_list = [r99_list[i] * fidelities[i] for i in range(len(r99_list))]
    # Plot Wigners in 4x3 grid
    print("Plotting Wigner grid...")
    fig_wig, axes = plt.subplots(4, 3, figsize=(15, 20))
    axes = axes.flatten()
    
    for i, W_data in enumerate(wigners):
        if i >= len(axes): break
        ax = axes[i]
        im = ax.imshow(W_data, origin='lower', extent=[xvec[0], xvec[-1], yvec[0], yvec[-1]], cmap='RdBu', vmin=-1, vmax=1)
        ax.set_title(f"{target_times[i]*1e9:.1f} ns\nFid: {fidelities[i]:.3f}, SQ: {tri_squeezing[i]:.1f}dB\nR99: {r99_list[i]:.2f}, V-: {v_neg_list[i]:.3f}")
        ax.set_xlabel('Re(alpha)')
        ax.set_ylabel('Im(alpha)')
        
    plt.tight_layout()
    plt.show()

    # Plot Metrics vs Pulse Length
    pulse_lengths_ns = target_times * 1e9

    fig, ax = plt.subplots(2, 2, figsize=(16, 12))
    
    # Fidelity
    ax[0, 0].plot(pulse_lengths_ns, fidelities, 'o-', linewidth=2)
    ax[0, 0].set_xlabel('Time (ns)', fontsize=14)
    ax[0, 0].set_ylabel('Fidelity', fontsize=14)
    ax[0, 0].set_title('Fidelity vs. Time', fontsize=16)
    ax[0, 0].grid(True)
    
    # Tri-Squeezing
    ax[0, 1].plot(pulse_lengths_ns, tri_squeezing, 'o-', linewidth=2, color='orange')
    ax[0, 1].set_xlabel('Time (ns)', fontsize=14)
    ax[0, 1].set_ylabel('Tri-Squeezing (dB)', fontsize=14)
    ax[0, 1].set_title('Tri-Squeezing vs. Time', fontsize=16)
    ax[0, 1].grid(True)
    
    # R99 Support
    ax[1, 0].plot(pulse_lengths_ns, r99_list, 'o-', linewidth=2, color='green')
    ax[1, 0].set_xlabel('Time (ns)', fontsize=14)
    ax[1, 0].set_ylabel('R99 Support', fontsize=14)
    ax[1, 0].set_title('Phase Space Support (R99) vs. Time', fontsize=16)
    ax[1, 0].grid(True)
    
    # Negativity Volume
    ax[1, 1].plot(pulse_lengths_ns, product_list, 'o-', linewidth=2, color='red')
    ax[1, 1].set_xlabel('Time (ns)', fontsize=14)
    ax[1, 1].set_ylabel('R99 * Fidelity', fontsize=14)
    ax[1, 1].set_title('R99 * Fidelity vs. Time', fontsize=16)
    ax[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

    # Find max fidelity
    max_fid = max(fidelities)
    best_time = pulse_lengths_ns[np.argmax(fidelities)]
    print(f"\nBest Fidelity: {max_fid:.4f} at {best_time:.1f} ns")

if __name__ == "__main__":
    main()
