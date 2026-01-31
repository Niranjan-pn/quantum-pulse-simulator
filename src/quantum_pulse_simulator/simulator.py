import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import qutip as qt
from scipy.constants import hbar
from tqdm import tqdm
from matplotlib.cm import get_cmap
qt.CoreOptions.default_dtype = "jax"
import os
class QutipPulseSimulator:
    """
    Simulator for quantum systems using QuTiP, supporting pulse-driven dynamics,
    time evolution, and visualization of quantum states via Wigner functions and Fock state populations.
    """

    def __init__(self, systems, pulse_chains=None, num_time_points=100):
        """
        Args:
            systems (list): List of quantum system objects, each with attributes such as `omega`, `num_fock`, `state`, and `H0`.
            pulse_chains (list, optional): List of PulseSequence objects for each system or for multi-system pulses.
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
        """
        max_tone = 0
        for pc in self.pulse_chains:
            if pc is not None:
                for pulse in pc.pulses:
                    if "tone" in pulse and pulse["tone"] is not None:
                        max_tone = max(max_tone, abs(pulse["tone"]))
        if max_tone == 0:
            max_tone = max([s.omega for s in self.systems])
        min_period = 1.0 / max_tone
        dt = min_period / 1  # 3 points per period

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
                if not pulse.get("multi_system", False):
                    # Single-system pulse
                    # Find the index of the system this pulse acts on
                    # If pc.systems has only one system, use it
                    if len(pc.systems) == 1:
                        sys_idx = self.systems.index(pc.systems[0])
                    else:
                        # Try to infer from the operator structure (not recommended)
                        raise ValueError("Ambiguous system index for single-system pulse.")
                    ops = [qt.identity(s.num_fock) for s in self.systems]
                    ops[sys_idx] = pulse["hamiltonian"]
                    H_full = qt.tensor(ops)
                else:
                    # Multi-system pulse - already has tensor product Hamiltonian
                    H_full = pulse["hamiltonian"]

                # Create time-dependent coefficient function for this pulse
                duration = pulse["duration"]
                strength = pulse["strength"]
                tone = pulse["tone"]
                shape = pulse["shape"]
                phase = pulse.get("phase", 0.0)
                t_start = elapsed

                def coeff_factory(duration, strength, tone, shape, phase, t_start):
                    # Returns a function of (t, args)
                    def coeff(t, args):
                        rel_t = t - t_start
                        if rel_t < 0 or rel_t > duration:
                            return 0.0
                        envelope = shape(rel_t) if shape is not None else 1.0
                        if tone is not None and tone != 0:
                            return strength * envelope * np.cos(tone * rel_t + phase)
                        else:
                            return strength * envelope
                    return coeff

                coeff = coeff_factory(duration, strength, tone, shape, phase, t_start)
                H_td_list.append([H_full, coeff])
                elapsed += duration

        return [H0] + H_td_list

    def _build_c_ops(self):
        """
        Construct the total collapse operators list by tensor product of system-specific c_ops
        with identity operators for other systems.
        """
        total_c_ops = []
        for i, sys in enumerate(self.systems):
            for op in sys.c_ops:
                ops = [qt.identity(s.num_fock) for s in self.systems]
                ops[i] = op
                total_c_ops.append(qt.tensor(ops))
        return total_c_ops

    def simulate(self, batch_size=1000, save_dir=".", save_prefix="batch_result", simulation_name="sim", store_batch_file=False):
        """
        Run time evolution in batches to avoid memory errors.
        Each batch result is saved to disk in the specified directory with the given prefix.
        Args:
            batch_size (int): Number of time points per batch.
            save_dir (str, optional): Base path for results. Defaults to current directory ("."). 
                                      But if simulation_name is provided, it will create a folder inside lib/.
            save_prefix (str): Prefix for saved batch files.
            simulation_name (str): Name of the simulation, used for folder naming.
            store_batch_file (bool): If True, keep batch files. If False, delete them after loading.
        """
        from datetime import datetime
        import shutil

        # Construct result directory: lib/{simulation_name}_{date}_{time}
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # Determine the directory of the running script to place results nearby
        try:
            import sys
            script_path = os.path.abspath(sys.argv[0])
            # Check if it looks like a python script file
            if os.path.isfile(script_path):
                script_dir = os.path.dirname(script_path)
            else:
                script_dir = os.getcwd()
        except:
            script_dir = os.getcwd()

        # If the user passed default save_dir=".", we override it to use the new structure logic
        # which places sim_results relative to the script
        if save_dir == ".":
            result_dir = os.path.join(script_dir, "sim_results", f"{simulation_name}_{timestamp}")
        else:
            result_dir = os.path.join(save_dir, f"{simulation_name}_{timestamp}")

        os.makedirs(result_dir, exist_ok=True)
        self.save_dir = result_dir # Store for plotting functions

        H = self._build_total_hamiltonian()
        c_ops = self._build_c_ops()
        n_times = len(self.tlist)
        batch_starts = list(range(0, n_times-1, batch_size))
        initial_state = self.initial_state
        batch_files = []
        self.batch_size = int(batch_size)  # set in __init__ or simulate()

        print(f"Simulating in {len(batch_starts)} batches of up to {batch_size} time points each...")
        print(f"Results will be stored in: {result_dir}")
        if c_ops:
            print(f"Collapse operators detected ({len(c_ops)}). Using mesolve.")
        else:
            print("No collapse operators. Using sesolve.")

        for i, start in enumerate(tqdm(batch_starts, desc="Batches")):
            end = min(start + batch_size, n_times-1)
            t_batch = self.tlist[start:end+1]  # include endpoint
            
            common_args = {
                "H": H,
                "rho0": initial_state,
                "tlist": t_batch,
                "args": None,
                "options": {"progress_bar": None, "atol": 1e-6, "rtol": 1e-6},
            }

            if c_ops is None or len(c_ops) == 0:
                # Use Schrodinger equation solver for unitary dynamics (faster)
                # Note: parameter name for initial state in sesolve is 'psi0', but positionally it's 2nd.
                # using positional args for H, state, tlist to match both signatures roughly, 
                # but explicit keywords are safer if they differ.
                # sesolve signature: sesolve(H, psi0, tlist, e_ops=[], args={}, options=None, ...)
                result = qt.sesolve(
                    H,
                    initial_state,
                    t_batch,
                    e_ops=[],
                    args=None,
                    options={"progress_bar": None, "atol": 1e-6, "rtol": 1e-6},
                )
            else:
                # Use Master equation solver (Lindblad dynamics)
                result = qt.mesolve(
                    H,
                    initial_state,
                    t_batch,
                    c_ops=c_ops,
                    args=None,
                    options={"progress_bar": None, "atol": 1e-6, "rtol": 1e-6},
                )
            filename = f"{save_prefix}_{i}"
            full_path = os.path.join(result_dir, filename)
            qt.qsave(result, full_path)
            batch_files.append(full_path)
            initial_state = result.states[-1]  # carry last state to next batch

        print(f"All {len(batch_files)} batches completed and saved to disk in '{result_dir}'.")
        self.result_files = batch_files  # Store for later loading

        # Load full result immediately so we can clean up if needed
        # (The user didn't explicitly ask to load it here, but if we delete files, we MUST load it first)
        # However, looking at original code, load_full_result was separate. 
        # But if store_batch_file is False, we generally want the result in memory but no files.
        # Let's perform cleanup logic *after* the user would usually load. 
        # Actually to be safe and "atomic", we should probably load it here if we are going to delete.
        # But let's stick to the request: "If store_batch_file = False, then it should only ssave the plots there."
        # This implies we might need the result object available.
        
        # Let's automatically load the result so we have it in memory before deleting files
        if not store_batch_file:
            print("store_batch_file=False: Loading full result into memory explicitly to allow file cleanup...")
            self.load_full_result()
            print(f"Cleaning up result directory: {result_dir}...")
            if os.path.exists(result_dir):
                shutil.rmtree(result_dir)
            print("Batch files and directory deleted.")

        return batch_files

    def load_full_result(self):
        """
        Load and concatenate all batch results into a single result object.
        """
        all_states = []
        all_times = []
        for filename in self.result_files:
            result = qt.qload(filename)
            # Avoid duplicate time points at batch boundaries
            if all_times and result.times[0] == all_times[-1]:
                all_states.extend(result.states[1:])
                all_times.extend(result.times[1:])
            else:
                all_states.extend(result.states)
                all_times.extend(result.times)
        # Create a dummy result object for compatibility
        class DummyResult:
            pass
        full_result = DummyResult()
        full_result.states = all_states
        full_result.times = np.array(all_times)
        self.result = full_result
        print("Full result loaded and concatenated from disk.")
        return full_result

    def get_state_at_frame(self, frame):
        # Prefer loading from memory if full result is loaded (e.g. if batch files were deleted)
        if self.result is not None:
             return self.result.states[frame]

        # Fallback to disk loading
        batch_idx = frame // self.batch_size
        state_idx = frame % self.batch_size
        result = qt.qload(self.result_files[batch_idx])
        return result.states[state_idx]

    
    def plot_wigner(self, time=-1, system_index=[0],save_path=None, show_plot=True, xlim=None):
        """
        Plot the Wigner function for the specified systems at a given simulation time.
        """
        idx = np.argmin(np.abs(self.tlist - time)) if time >= 0 else -1
        rho = self.result.states[idx]
        if xlim is None:
            xvec = np.linspace(-5, 5, 200)
        else:
            xvec = np.linspace(xlim[0], xlim[1], 200)
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
        if save_path:
            plt.savefig(save_path)
        if show_plot:
            plt.show()

    def animate_wigner(self, systems=None, number_of_frames=20, fps=100, save=True, writer='Pillow'):
        """
        Animate Wigner functions.
        Args:
            save (bool or str): If True, saves to 'wigner_animation.gif' in the result directory. 
                                If str, saves to that filename. 
                                If False, does not save.
        """
        save_path = None
        if save:
            filename = save if isinstance(save, str) else "wigner_animation.gif"
            if hasattr(self, 'save_dir'):
                save_path = os.path.join(self.save_dir, filename)
            else:
                save_path = filename # Fallback if save_dir not set

        
        if systems is None:
            raise ValueError("No systems provided for animation. Please specify a list of system objects.")
        n_systems = len(systems)
        xvec = np.linspace(-5, 5, 100)
        total_frames = len(self.tlist)
        frame_step = max(1, total_frames // number_of_frames)
        selected_frames = np.arange(0, total_frames, frame_step)
        tlist_ns = self.tlist * 1e9  # ns for display

        # Prepare color map for pulse chains
        cmap = get_cmap('tab10')
        pulse_chain_colors = {pc_idx: cmap(pc_idx % 10) for pc_idx, pc in enumerate(self.pulse_chains) if pc is not None}

        # --- Progress bar for global_clim computation ---
        global_clim = 0
        sample_idxs = selected_frames[::max(1, len(selected_frames)//5)]
        for frame in tqdm(sample_idxs, desc="Computing Wigner (global_clim)"):
            rho =  self.get_state_at_frame(frame)
            for sys in systems:
                sys_idx = self.systems.index(sys)
                rho_reduced = qt.ptrace(rho, sys_idx)
                W = qt.wigner(rho_reduced, xvec, xvec)
                global_clim = max(global_clim, np.max(np.abs(W)))
        if global_clim == 0:
            global_clim = 0.2  # fallback

        # Prepare the figure
        n_rows = 2
        has_multi = any(
            pc is not None and len(pc.systems) > 1
            for pc in self.pulse_chains
        )
        if has_multi:
            n_rows = 3

        fig, axes = plt.subplots(n_rows, n_systems, figsize=(5 * n_systems, 4 * n_rows), squeeze=False)

        ims = []
        vlines = [None for _ in range(n_systems)]
        multi_vlines = {}

        # --- Initialize Wigner plots ---
        for i in range(n_systems):
            # Dummy initial data
            im = axes[0, i].imshow(np.zeros((len(xvec), len(xvec))), extent=[-5, 5, -5, 5],
                                aspect='auto', origin='lower', cmap='RdBu',
                                vmin=-global_clim, vmax=global_clim)
            axes[0, i].set_title(f"{getattr(systems[i], 'name', i)}")
            axes[0, i].set_xlabel('x')
            axes[0, i].set_ylabel('p')
            axes[0, i].set_aspect('equal', adjustable='box')
            ims.append(im)

        # --- Plot single-system pulses (including waiting pulses) ---
        for i, sys in enumerate(systems):
            sys_idx = self.systems.index(sys)
            chain_handles = []
            chain_labels = []
            for pc_idx, pc in enumerate(self.pulse_chains):
                if pc is not None and len(pc.systems) == 1 and pc.systems[0] == sys:
                    if hasattr(pc, 'pulses') and pc.pulses:
                        t_start = 0.0
                        chain_color = pulse_chain_colors[pc_idx]
                        chain_handles.append(plt.Line2D([0], [0], color=chain_color))
                        chain_labels.append(f"Chain {pc_idx}")
                        for pulse_idx, pulse in enumerate(pc.pulses):
                            t_end = t_start + pulse["duration"]
                            idxs = np.where((tlist_ns >= t_start * 1e9) & (tlist_ns < t_end * 1e9))[0]
                            if len(idxs) == 0:
                                t_start += pulse["duration"]
                                continue
                            pulse_start_ns = t_start * 1e9
                            axes[1, i].axvline(pulse_start_ns, color='black', linestyle='--', alpha=0.7)
                            pulse_name = pulse.get('name', f'pulse_{pulse_idx}')
                            y_pos = axes[1, i].get_ylim()[1] * 0.9 if axes[1, i].get_ylim()[1] > 0 else 0.9
                            axes[1, i].text(pulse_start_ns, y_pos, pulse_name,
                                            rotation=90, verticalalignment='top',
                                            color='black', fontsize=8)
                            t_pulse = tlist_ns[idxs]
                            rel_times = [(t - t_start * 1e9) / 1e9 for t in t_pulse]
                            envelope = np.array([pulse["shape"](t) if pulse["shape"] else 1.0 for t in rel_times])
                            if "tone" in pulse and pulse["tone"] is not None:
                                tone = np.cos(pulse["tone"] * np.array(rel_times) + pulse.get("phase", 0))
                                pulse_wave = pulse["strength"] * envelope * tone
                            else:
                                pulse_wave = pulse["strength"] * envelope
                            axes[1, i].plot(t_pulse, pulse_wave, color=chain_color, linestyle='-')
                            t_start += pulse["duration"]
                        vlines[i] = axes[1, i].axvline(tlist_ns[0], color='r', linestyle='-')
                    axes[1, i].set_xlabel('Time (ns)')
                    axes[1, i].set_ylabel('Drive')
            if chain_handles:
                axes[1, i].legend(handles=chain_handles, labels=chain_labels, loc='upper right', fontsize='small')

        # --- Plot multi-system pulses (only in relevant subplots) ---
        if has_multi:
            global_time_map = {}
            for pc_idx, pc in enumerate(self.pulse_chains):
                if pc is not None and len(pc.systems) > 1:
                    for sys in pc.systems:
                        if sys not in global_time_map:
                            global_time_map[sys] = 0.0
            for i, sys in enumerate(systems):
                chain_handles = []
                chain_labels = []
                for pc_idx, pc in enumerate(self.pulse_chains):
                    if pc is not None and len(pc.systems) > 1 and sys in pc.systems:
                        chain_color = pulse_chain_colors[pc_idx]
                        chain_handles.append(plt.Line2D([0], [0], color=chain_color, linestyle='--'))
                        chain_labels.append(f"Chain {pc_idx} (multi)")
                        if hasattr(pc, 'pulses') and pc.pulses:
                            t_start = 0
                            for pulse_idx, pulse in enumerate(pc.pulses):
                                if not pulse.get("multi_system", False):
                                    t_start += pulse["duration"]
                                    continue
                                t_end = t_start + pulse["duration"]
                                idxs = np.where((tlist_ns >= t_start * 1e9) & (tlist_ns < t_end * 1e9))[0]
                                if len(idxs) == 0:
                                    t_start += pulse["duration"]
                                    continue
                                pulse_start_ns = t_start * 1e9
                                axes[2, i].axvline(pulse_start_ns, color="black", linestyle='--', alpha=0.7)
                                pulse_name = pulse.get('name', f'multi-pulse_{pulse_idx}')
                                y_pos = axes[2, i].get_ylim()[1] * 0.9 if axes[2, i].get_ylim()[1] > 0 else 0.9
                                axes[2, i].text(pulse_start_ns, y_pos, pulse_name,
                                                rotation=90, verticalalignment='top',
                                                color="black", fontsize=8)
                                t_pulse = tlist_ns[idxs]
                                rel_times = [(t - t_start * 1e9) / 1e9 for t in t_pulse]
                                envelope = np.array([pulse["shape"](t) if pulse["shape"] else 1.0 for t in rel_times])
                                if "tone" in pulse and pulse["tone"] is not None:
                                    tone = np.cos(pulse["tone"] * np.array(rel_times) + pulse.get("phase", 0))
                                    pulse_wave = pulse["strength"] * envelope * tone
                                else:
                                    pulse_wave = pulse["strength"] * envelope
                                axes[2, i].plot(t_pulse, pulse_wave, color=chain_color, linestyle='--', alpha=0.7)
                                t_start += pulse["duration"]
                            global_time_map[sys] = max(global_time_map[sys], t_start)
                        multi_vlines[sys] = axes[2, i].axvline(tlist_ns[0], color='r', linestyle='--')
                    axes[2, i].set_xlabel('Time (ns)')
                    axes[2, i].set_ylabel('Drive')
                if chain_handles:
                    axes[2, i].legend(handles=chain_handles, labels=chain_labels, loc='upper right', fontsize='small')

        fig.tight_layout()

        def update(frame_idx):
            frame = selected_frames[frame_idx]
            t_ns = tlist_ns[frame]
            rho = self.get_state_at_frame(frame)
            for i, sys in enumerate(systems):
                sys_idx = self.systems.index(sys)
                rho_reduced = qt.ptrace(rho, sys_idx)
                W = qt.wigner(rho_reduced, xvec, xvec)
                ims[i].set_data(W)
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

        # --- Progress bar for saving animation ---
        if save_path is not None:
            with tqdm(total=len(selected_frames), desc="Saving Wigner Animation") as pbar:
                def progress_callback(frame, total_frames):
                    pbar.n = frame + 1
                    pbar.refresh()
                ani.save(save_path, writer=writer, progress_callback=progress_callback)
            print(f"Animation saved to {save_path}")

        plt.show()
        return ani


    def plot_fock_expectations(self, system_indices, fock_states):
        """
        Plot expectation values of multiple Fock states for multiple systems.
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
            pc is not None and len(pc.systems) > 1
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
                if pc is not None and len(pc.systems) == 1 and pc.systems[0] == sys:
                    if hasattr(pc, 'pulses') and pc.pulses:
                        t_start = 0.0
                        chain_color = pulse_chain_colors[pc_idx]
                        chain_handles.append(plt.Line2D([0], [0], color=chain_color))
                        chain_labels.append(f"Chain {pc_idx}")
                        for pulse_idx, pulse in enumerate(pc.pulses):
                            t_end = t_start + pulse["duration"]
                            idxs = np.where((tlist_ns >= t_start * 1e9) & (tlist_ns < t_end * 1e9))[0]
                            if len(idxs) == 0:
                                t_start += pulse["duration"]
                                continue
                            pulse_start_ns = t_start * 1e9
                            axes[1, i].axvline(pulse_start_ns, color='black', linestyle='--', alpha=0.7)
                            pulse_name = pulse.get('name', f'pulse_{pulse_idx}')
                            y_pos = 0.9
                            axes[1, i].text(pulse_start_ns, y_pos, pulse_name,
                                            rotation=90, verticalalignment='top',
                                            color='black', fontsize=8, transform=axes[1, i].get_xaxis_transform())
                            t_pulse = tlist_ns[idxs]
                            rel_times = [(t - t_start * 1e9) / 1e9 for t in t_pulse]
                            envelope = np.array([pulse["shape"](t) if pulse["shape"] else 1.0 for t in rel_times])
                            if "tone" in pulse and pulse["tone"] is not None:
                                tone = np.cos(pulse["tone"] * np.array(rel_times) + pulse.get("phase", 0))
                                pulse_wave = pulse["strength"] * envelope * tone
                            else:
                                pulse_wave = pulse["strength"] * envelope
                            axes[1, i].plot(t_pulse, pulse_wave, color=chain_color, linestyle='-')
                            t_start += pulse["duration"]
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
                    if pc is not None and len(pc.systems) > 1 and sys in pc.systems:
                        chain_color = pulse_chain_colors[pc_idx]
                        chain_handles.append(plt.Line2D([0], [0], color=chain_color, linestyle='--'))
                        chain_labels.append(f"Chain {pc_idx} (multi)")
                        if hasattr(pc, 'pulses') and pc.pulses:
                            t_start = 0
                            for pulse_idx, pulse in enumerate(pc.pulses):
                                if not pulse.get("multi_system", False):
                                    t_start += pulse["duration"]
                                    continue
                                t_end = t_start + pulse["duration"]
                                idxs = np.where((tlist_ns >= t_start * 1e9) & (tlist_ns < t_end * 1e9))[0]
                                if len(idxs) == 0:
                                    t_start += pulse["duration"]
                                    continue
                                pulse_start_ns = t_start * 1e9
                                axes[2, i].axvline(pulse_start_ns, color="black", linestyle='--', alpha=0.7)
                                pulse_name = pulse.get('name', f'multi-pulse_{pulse_idx}')
                                y_pos = 0.9
                                axes[2, i].text(pulse_start_ns, y_pos, pulse_name,
                                            rotation=90, verticalalignment='top',
                                            color="black", fontsize=8, transform=axes[2, i].get_xaxis_transform())
                                t_pulse = tlist_ns[idxs]
                                rel_times = [(t - t_start * 1e9) / 1e9 for t in t_pulse]
                                envelope = np.array([pulse["shape"](t) if pulse["shape"] else 1.0 for t in rel_times])
                                if "tone" in pulse and pulse["tone"] is not None:
                                    tone = np.cos(pulse["tone"] * np.array(rel_times) + pulse.get("phase", 0))
                                    pulse_wave = pulse["strength"] * envelope * tone
                                else:
                                    pulse_wave = pulse["strength"] * envelope
                                axes[2, i].plot(t_pulse, pulse_wave, color=chain_color, linestyle='--', alpha=0.7)
                                t_start += pulse["duration"]
                axes[2, i].set_xlabel('Time (ns)')
                axes[2, i].set_ylabel('Drive')
                axes[2, i].set_title(f"{getattr(sys, 'name', f'System {i}')}: Multi-system pulses")
                if chain_handles:
                    axes[2, i].legend(handles=chain_handles, labels=chain_labels, loc='upper right', fontsize='small')
                axes[2, i].grid(True)

        plt.tight_layout()
        plt.show()
