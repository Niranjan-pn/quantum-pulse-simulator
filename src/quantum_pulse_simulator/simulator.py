import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import qutip as qt
from scipy.constants import hbar
from tqdm import tqdm
from matplotlib.cm import get_cmap
class QutipPulseSimulator:
    """
    Simulator for quantum systems using QuTiP, supporting pulse-driven dynamics,
    time evolution, and visualization of quantum states via Wigner functions and Fock state populations.
    """

    def __init__(self, systems, pulse_chains=None, num_time_points=100):
        """
        Initialize the simulator.

        Args:
            systems (list): List of quantum system objects, each with attributes such as `omega`, `num_fock`, `state`, and `H0`.
            pulse_chains (list, optional): List of pulse chain objects for each system. If None, initializes with no pulses for each system.
            num_time_points (int, optional): Number of time points for simulation (default: 100).
        """
        self.systems = systems
        self.pulse_chains = pulse_chains if pulse_chains is not None else [None] * len(systems)
        self.tlist = self._generate_tlist()
        self.initial_state = self._build_initial_state()
        self.result = None

    def _generate_tlist(self):
        """
        Generate the time list for simulation based on the highest tone frequency in pulses or system frequency.

        Returns:
            np.ndarray: Array of time points for the simulation.
        """
        max_tone = 0
        for pc in self.pulse_chains:
            if pc is not None:
                for pulse in pc.pulses:
                    if hasattr(pulse, "tone"):
                        max_tone = max(max_tone, abs(pulse.tone))
        if max_tone == 0:
            max_tone = max([s.omega for s in self.systems])
        min_period = 1.0 / max_tone
        dt = min_period / 20  # 20 points per period

        max_duration = 0
        for pc in self.pulse_chains:
            if pc is not None:
                max_duration = max(max_duration, pc.total_duration())
        if max_duration == 0:
            max_duration = 10e-9  # 10 ns default

        N = int(np.ceil(max_duration / dt)) + 1
        return np.linspace(0, max_duration, N)

    def _build_initial_state(self):
        """
        Build the initial state as a tensor product of subsystem states.

        Returns:
            Qobj: The initial state as a QuTiP Qobj.
        """
        return qt.tensor([s.state for s in self.systems])

    def _build_total_hamiltonian(self):
        """
        Construct the total Hamiltonian, including static and time-dependent parts.
        Treats single and multi-system pulses in a unified way.
        """
        # Static Hamiltonian part
        H0 = 0
        for i, sys in enumerate(self.systems):
            ops = [qt.identity(s.num_fock) for s in self.systems]
            ops[i] = sys.H0
            H0 += qt.tensor(ops)
        
        # Time-dependent Hamiltonian parts
        H_td_list = []
        
        # Process all pulse chains sequentially
        for pc in self.pulse_chains:
            if pc is None:
                continue
            
            elapsed = 0
            for pulse in pc.pulses:
                # Determine the Hamiltonian based on whether it's a multi-system or single-system pulse
                if not hasattr(pulse, "multi_system") or not pulse.multi_system:
                    # Single-system pulse
                    sys_idx = self.systems.index(pc.system)
                    ops = [qt.identity(s.num_fock) for s in self.systems]
                    ops[sys_idx] = pulse.hamiltonian
                    H_full = qt.tensor(ops)
                else:
                    # Multi-system pulse - already has tensor product Hamiltonian
                    H_full = pulse.hamiltonian
                
                # Create time-dependent coefficient function for this pulse
                def coeff_factory(pulse, t_start):
                    return lambda t, args: pulse.time_dep_coeff(t, t_start)

                
                coeff = coeff_factory(pulse, elapsed)
                H_td_list.append([H_full, coeff])
                elapsed += pulse.duration
        
        return [H0] + H_td_list




    def simulate(self):
        """
        Run the time evolution of the system using QuTiP's mesolve and store the result.

        Returns:
            Result: QuTiP Result object containing the time-evolved states.
        """
        H = self._build_total_hamiltonian()
        self.result = qt.mesolve(
            H,
            self.initial_state,
            self.tlist,
            c_ops=[],
            args=None,
            options={"progress_bar": "tqdm",
                     }
        )
        return self.result

    def plot_wigner(self, time=-1, system_index=[0]):
        """
        Plot the Wigner function for the specified systems at a given simulation time.

        Args:
            time (float, optional): Time at which to plot the Wigner function (default: last time point).
            system_index (list of int, optional): Indices of subsystems to plot (default: [0]).
        """
        idx = np.argmin(np.abs(self.tlist - time))
        rho = self.result.states[idx]
        xvec = np.linspace(-5, 5, 200)
        fig, ax = plt.subplots(1, len(system_index), figsize=(4 * len(system_index), 8), sharex=True, sharey=True)
        if len(system_index) == 1:
            ax = [ax]
        for i, system in enumerate(system_index):
            rho_reduced = qt.ptrace(rho, system)
            W = qt.wigner(rho_reduced, xvec, xvec)
            ax[i].imshow(W, extent=[-5, 5, -5, 5], aspect='auto', origin='lower')
            ax[i].set_title(f'{self.systems[system].name} Wigner at t={self.tlist[idx] * 1e9:.2f} ns')
            ax[i].set_xlabel('x')
            ax[i].set_ylabel('p')
            ax[i].set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.show()

    def animate_wigner(self, systems=None, number_of_frames=20, fps=100, save_path=None, writer='ffmpeg'):


        if systems is None:
            raise ValueError("No systems provided for animation. Please specify a list of system objects.")

        n_systems = len(systems)
        xvec = np.linspace(-5, 5, 100)
        total_frames = len(self.tlist)
        frame_step = max(1, total_frames // number_of_frames)
        selected_frames = np.arange(0, total_frames, frame_step)
        tlist_ns = self.tlist * 1e9  # ns for display

        # Create a color map for pulse chains
        cmap = get_cmap('tab10')
        pulse_chain_colors = {pc_idx: cmap(pc_idx % 10) for pc_idx, pc in enumerate(self.pulse_chains) if pc is not None}

        # Precompute Wigner functions and global clim
        wigner_list = []
        global_clim = 0
        for frame in tqdm(selected_frames, desc="Computing Wigner frames"):
            rho = self.result.states[frame]
            wigner_frames = []
            for sys in systems:
                sys_idx = self.systems.index(sys)
                rho_reduced = qt.ptrace(rho, sys_idx)
                W = qt.wigner(rho_reduced, xvec, xvec)
                wigner_frames.append(W)
                global_clim = max(global_clim, np.max(np.abs(W)))
            wigner_list.append(wigner_frames)

        # Prepare the figure
        n_rows = 2
        has_multi = any(
            pc is not None and isinstance(pc.system, (list, tuple)) and len(pc.system) > 1
            for pc in self.pulse_chains
        )
        if has_multi:
            n_rows = 3

        fig, axes = plt.subplots(n_rows, n_systems, figsize=(5 * n_systems, 4 * n_rows), squeeze=False)

        # ----------- SHARE X AND Y ONLY FOR ROW 2 AND ROW 3 -----------
        # (axes[1, :] and axes[2, :] if n_rows == 3)
        if n_rows == 3:
            # Use axes[1,0] as the reference for row 2, axes[2,0] for row 3
            # Then join all axes in row 1 and 2 to the same group
            for i in range(n_systems):
                if i == 0:
                    continue
                # Share axes[1, i] and axes[2, i] with axes[1, 0] and axes[2, 0]
                axes[1, i].sharex(axes[1, 0])
                axes[1, i].sharey(axes[1, 0])
                axes[2, i].sharex(axes[2, 0])
                axes[2, i].sharey(axes[2, 0])
            # Now join row 2 and row 3 groups together for both x and y
            axes[2, 0].sharex(axes[1, 0])
            axes[2, 0].sharey(axes[1, 0])

        # --------------------------------------------------------------

        ims = []
        vlines = [None for _ in range(n_systems)]
        multi_vlines = {}

        # --- Initialize Wigner plots ---
        for i in range(n_systems):
            im = axes[0, i].imshow(wigner_list[0][i], extent=[-5, 5, -5, 5],
                                aspect='auto', origin='lower', cmap='RdBu',
                                vmin=-global_clim, vmax=global_clim)
            axes[0, i].set_title(f"{getattr(systems[i], 'name', i)}")
            axes[0, i].set_xlabel('x')
            axes[0, i].set_ylabel('p')
            ims.append(im)

        # --- Plot single-system pulses (including waiting pulses) ---
        for i, sys in enumerate(systems):
            sys_idx = self.systems.index(sys)
            chain_handles = []
            chain_labels = []
            for pc_idx, pc in enumerate(self.pulse_chains):
                if pc is not None and not isinstance(pc.system, (list, tuple)) and pc.system == sys:
                    if hasattr(pc, 'pulses') and pc.pulses:
                        t_start = 0.0
                        chain_color = pulse_chain_colors[pc_idx]
                        chain_handles.append(plt.Line2D([0], [0], color=chain_color))
                        chain_labels.append(f"Chain {pc_idx}")
                        for pulse_idx, pulse in enumerate(pc.pulses):
                            t_end = t_start + pulse.duration
                            idxs = np.where((tlist_ns >= t_start * 1e9) & (tlist_ns < t_end * 1e9))[0]
                            if len(idxs) == 0:
                                t_start += pulse.duration
                                continue
                            pulse_start_ns = t_start * 1e9
                            axes[1, i].axvline(pulse_start_ns, color='black', linestyle='--', alpha=0.7)
                            pulse_name = getattr(pulse, 'name', f'pulse_{pulse_idx}')
                            y_pos = axes[1, i].get_ylim()[1] * 0.9
                            axes[1, i].text(pulse_start_ns, y_pos, pulse_name, 
                                        rotation=90, verticalalignment='top', 
                                        color='black', fontsize=8)
                            t_pulse = tlist_ns[idxs]
                            rel_times = [(t - t_start * 1e9) / 1e9 for t in t_pulse]
                            envelope = np.array([pulse.shape(t) if pulse.shape else 1.0 for t in rel_times])
                            if hasattr(pulse, "tone") and pulse.tone is not None:
                                tone = np.cos(pulse.tone * np.array(rel_times) + getattr(pulse, "phase", 0))
                                pulse_wave = pulse.strength * envelope * tone
                            else:
                                pulse_wave = pulse.strength * envelope
                            axes[1, i].plot(t_pulse, pulse_wave, color=chain_color, linestyle='-')
                            t_start += pulse.duration
                        vlines[i] = axes[1, i].axvline(tlist_ns[0], color='r', linestyle='-')
                    axes[1, i].set_xlabel('Time (ns)')
                    axes[1, i].set_ylabel('Drive')
            if chain_handles:
                axes[1, i].legend(handles=chain_handles, labels=chain_labels, loc='upper right', fontsize='small')

        # --- Plot multi-system pulses (only in relevant subplots) ---
        if has_multi:
            global_time_map = {}
            for pc_idx, pc in enumerate(self.pulse_chains):
                if pc is not None and isinstance(pc.system, (list, tuple)):
                    affected_systems = pc.system if isinstance(pc.system, (list, tuple)) else [pc.system]
                    for sys in affected_systems:
                        if sys not in global_time_map:
                            global_time_map[sys] = 0.0
            for i, sys in enumerate(systems):
                chain_handles = []
                chain_labels = []
                for pc_idx, pc in enumerate(self.pulse_chains):
                    if pc is not None and isinstance(pc.system, (list, tuple)) and sys in pc.system:
                        chain_color = pulse_chain_colors[pc_idx]
                        chain_handles.append(plt.Line2D([0], [0], color=chain_color, linestyle='--'))
                        chain_labels.append(f"Chain {pc_idx} (multi)")
                        if hasattr(pc, 'pulses') and pc.pulses:
                            t_start = 0
                            for pulse_idx, pulse in enumerate(pc.pulses):
                                if not hasattr(pulse, "multi_system") or not pulse.multi_system:
                                    t_start += pulse.duration
                                    continue
                                t_end = t_start + pulse.duration
                                idxs = np.where((tlist_ns >= t_start * 1e9) & 
                                            (tlist_ns < t_end * 1e9))[0]
                                if len(idxs) == 0:
                                    t_start += pulse.duration
                                    continue
                                pulse_start_ns = t_start * 1e9
                                axes[2, i].axvline(pulse_start_ns, color="black", linestyle='--', alpha=0.7)
                                pulse_name = getattr(pulse, 'name', f'multi-pulse_{pulse_idx}')
                                y_pos = axes[2, i].get_ylim()[1] * 0.9
                                axes[2, i].text(pulse_start_ns, y_pos, pulse_name, 
                                            rotation=90, verticalalignment='top', 
                                            color="black", fontsize=8)
                                t_pulse = tlist_ns[idxs]
                                rel_times = [(t - t_start * 1e9) / 1e9 for t in t_pulse]
                                envelope = np.array([pulse.shape(t) if pulse.shape else 1.0 for t in rel_times])
                                if hasattr(pulse, "tone") and pulse.tone is not None:
                                    tone = np.cos(pulse.tone * np.array(rel_times) + getattr(pulse, "phase", 0))
                                    pulse_wave = pulse.strength * envelope * tone
                                else:
                                    pulse_wave = pulse.strength * envelope
                                axes[2, i].plot(t_pulse, pulse_wave, color=chain_color, linestyle='--', alpha=0.7)
                                t_start += pulse.duration
                            global_time_map[sys] = max(global_time_map[sys], t_start)
                        multi_vlines[sys] = axes[2, i].axvline(tlist_ns[0], color='r', linestyle='--')
                    axes[2, i].set_xlabel('Time (ns)')
                    axes[2, i].set_ylabel('Drive')
                if chain_handles:
                    axes[2, i].legend(handles=chain_handles, labels=chain_labels, loc='upper right', fontsize='small')

        fig.tight_layout()

        def update(frame_idx):
            t_ns = tlist_ns[selected_frames[frame_idx]]
            for i, sys in enumerate(systems):
                ims[i].set_data(wigner_list[frame_idx][i])
                axes[0, i].set_title(f"{getattr(sys, 'name', f'System {i}')}")
                if vlines[i] is not None:
                    vlines[i].set_xdata([t_ns, t_ns])
                if sys in multi_vlines:
                    multi_vlines[sys].set_xdata([t_ns, t_ns])
            return ims + [v for v in vlines if v is not None] + list(multi_vlines.values())

        ani = FuncAnimation(
            fig,
            update,
            frames=len(selected_frames),
            interval=1000 / fps,
            blit=True,
            cache_frame_data=False
        )

        if save_path is not None:
            ani.save(save_path, writer=writer)
            print(f"Animation saved to {save_path}")

        plt.show()
        return ani


    def plot_fock_expectations(self, system_indices, fock_states):
        """
        Plot expectation values of multiple Fock states for multiple systems.

        Args:
            system_indices (list of int): Indices of systems to plot.
            fock_states (list of int): Fock state numbers to plot for each system.
        """
        tlist = self.tlist * 1e9  # Convert to ns for plotting
        num_systems = len(system_indices)
        plt.figure(figsize=(7, 3 * num_systems))

        for i, sys_idx in enumerate(system_indices):
            plt.subplot(num_systems, 1, i + 1)
            for fock_n in fock_states:
                N = self.systems[sys_idx].num_fock
                proj_n = qt.fock_dm(N, fock_n)
                probs = []
                for state in self.result.states:
                    reduced = qt.ptrace(state, sys_idx)
                    probs.append((proj_n * reduced).tr().real)
                plt.plot(tlist, probs, label=f'Fock |{fock_n}⟩')
            plt.xlabel('Time (ns)')
            plt.ylabel('Probability')
            plt.title(f'System {sys_idx} Fock state populations')
            plt.legend()
            plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_pulse_sequence(self, systems=None):
        """
        Plot the pulse sequence as in animate_wigner, but static (no animation, no Wigner).
        - Row 2: single-system pulses
        - Row 3: multi-system pulses
        """
        if systems is None:
            systems = self.systems
        n_systems = len(systems)
        tlist_ns = self.tlist * 1e9  # ns for display

        # Prepare color map for pulse chains
        cmap = get_cmap('tab10')
        pulse_chain_colors = {pc_idx: cmap(pc_idx % 10) for pc_idx, pc in enumerate(self.pulse_chains) if pc is not None}

        # Check if there are any multi-system pulses
        has_multi = any(
            pc is not None and isinstance(pc.system, (list, tuple)) and len(pc.system) > 1
            for pc in self.pulse_chains
        )
        n_rows = 3 if has_multi else 2

        fig, axes = plt.subplots(n_rows, n_systems, figsize=(5 * n_systems, 3 * n_rows), squeeze=False)

        # --- Plot single-system pulses ---
        for i, sys in enumerate(systems):
            sys_idx = self.systems.index(sys)
            chain_handles = []
            chain_labels = []
            for pc_idx, pc in enumerate(self.pulse_chains):
                if pc is not None and not isinstance(pc.system, (list, tuple)) and pc.system == sys:
                    if hasattr(pc, 'pulses') and pc.pulses:
                        t_start = 0.0
                        chain_color = pulse_chain_colors[pc_idx]
                        chain_handles.append(plt.Line2D([0], [0], color=chain_color))
                        chain_labels.append(f"Chain {pc_idx}")
                        for pulse_idx, pulse in enumerate(pc.pulses):
                            t_end = t_start + pulse.duration
                            idxs = np.where((tlist_ns >= t_start * 1e9) & (tlist_ns < t_end * 1e9))[0]
                            if len(idxs) == 0:
                                t_start += pulse.duration
                                continue
                            pulse_start_ns = t_start * 1e9
                            axes[1, i].axvline(pulse_start_ns, color='black', linestyle='--', alpha=0.7)
                            pulse_name = getattr(pulse, 'name', f'pulse_{pulse_idx}')
                            y_pos = 0.9  # relative position; adjust as needed
                            axes[1, i].text(pulse_start_ns, y_pos, pulse_name, 
                                        rotation=90, verticalalignment='top', 
                                        color='black', fontsize=8, transform=axes[1, i].get_xaxis_transform())
                            t_pulse = tlist_ns[idxs]
                            rel_times = [(t - t_start * 1e9) / 1e9 for t in t_pulse]
                            envelope = np.array([pulse.shape(t) if pulse.shape else 1.0 for t in rel_times])
                            if hasattr(pulse, "tone") and pulse.tone is not None:
                                tone = np.cos(pulse.tone * np.array(rel_times) + getattr(pulse, "phase", 0))
                                pulse_wave = pulse.strength * envelope * tone
                            else:
                                pulse_wave = pulse.strength * envelope
                            axes[1, i].plot(t_pulse, pulse_wave, color=chain_color, linestyle='-')
                            t_start += pulse.duration
            axes[1, i].set_xlabel('Time (ns)')
            axes[1, i].set_ylabel('Drive')
            axes[1, i].set_title(f"{getattr(sys, 'name', f'System {i}')}: Single-system pulses")
            if chain_handles:
                axes[1, i].legend(handles=chain_handles, labels=chain_labels, loc='upper right', fontsize='small')
            axes[1, i].grid(True)

        # --- Plot multi-system pulses (if any) ---
        if has_multi:
            for i, sys in enumerate(systems):
                chain_handles = []
                chain_labels = []
                for pc_idx, pc in enumerate(self.pulse_chains):
                    if pc is not None and isinstance(pc.system, (list, tuple)) and sys in pc.system:
                        chain_color = pulse_chain_colors[pc_idx]
                        chain_handles.append(plt.Line2D([0], [0], color=chain_color, linestyle='--'))
                        chain_labels.append(f"Chain {pc_idx} (multi)")
                        if hasattr(pc, 'pulses') and pc.pulses:
                            t_start = 0
                            for pulse_idx, pulse in enumerate(pc.pulses):
                                if not hasattr(pulse, "multi_system") or not pulse.multi_system:
                                    t_start += pulse.duration
                                    continue
                                t_end = t_start + pulse.duration
                                idxs = np.where((tlist_ns >= t_start * 1e9) & (tlist_ns < t_end * 1e9))[0]
                                if len(idxs) == 0:
                                    t_start += pulse.duration
                                    continue
                                pulse_start_ns = t_start * 1e9
                                axes[2, i].axvline(pulse_start_ns, color="black", linestyle='--', alpha=0.7)
                                pulse_name = getattr(pulse, 'name', f'multi-pulse_{pulse_idx}')
                                y_pos = 0.9
                                axes[2, i].text(pulse_start_ns, y_pos, pulse_name, 
                                            rotation=90, verticalalignment='top', 
                                            color="black", fontsize=8, transform=axes[2, i].get_xaxis_transform())
                                t_pulse = tlist_ns[idxs]
                                rel_times = [(t - t_start * 1e9) / 1e9 for t in t_pulse]
                                envelope = np.array([pulse.shape(t) if pulse.shape else 1.0 for t in rel_times])
                                if hasattr(pulse, "tone") and pulse.tone is not None:
                                    tone = np.cos(pulse.tone * np.array(rel_times) + getattr(pulse, "phase", 0))
                                    pulse_wave = pulse.strength * envelope * tone
                                else:
                                    pulse_wave = pulse.strength * envelope
                                axes[2, i].plot(t_pulse, pulse_wave, color=chain_color, linestyle='--', alpha=0.7)
                                t_start += pulse.duration
                axes[2, i].set_xlabel('Time (ns)')
                axes[2, i].set_ylabel('Drive')
                axes[2, i].set_title(f"{getattr(sys, 'name', f'System {i}')}: Multi-system pulses")
                if chain_handles:
                    axes[2, i].legend(handles=chain_handles, labels=chain_labels, loc='upper right', fontsize='small')
                axes[2, i].grid(True)

        plt.tight_layout()
        plt.show()
