import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import qutip as qt
from scipy.constants import hbar
from tqdm import tqdm

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

        Returns:
            list: Hamiltonian for mesolve (static + time-dependent terms).
        """
        H0 = 0
        for i, sys in enumerate(self.systems):
            ops = [qt.identity(s.num_fock) for s in self.systems]
            ops[i] = sys.H0
            H0 += qt.tensor(ops)

        H_td_list = []
        for i, pc in enumerate(self.pulse_chains):
            if pc is None:
                continue
            elapsed = 0
            for pulse in pc.pulses:
                if not pulse.multi_system:
                    ops = [qt.identity(s.num_fock) for s in self.systems]
                    ops[i] = pulse.hamiltonian
                    H_full = qt.tensor(ops)
                else:
                    H_full = pulse.hamiltonian

                def coeff_factory(pulse, t_start):
                    return lambda t, args: pulse.time_dep_coeff(t - t_start, t_start)
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
            options={"progress_bar": "tqdm"}
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
            ax[i].set_title(f'System {system} Wigner Function')
            ax[i].set_xlabel('x')
            ax[i].set_ylabel('p')
            ax[i].set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.show()

    def animate_wigner(self, system_index=[0], number_of_frames=20, speed=100, save_path=None, writer='ffmpeg'):
        """
        Animate Wigner function evolution for specified systems, with optional saving to file.

        Args:
            system_index (list of int, optional): Indices of subsystems to animate (default: [0, 1]).
            number_of_frames (int, optional): Number of frames in the animation (default: 20).
            speed (int, optional): Animation speed in frames per second (default: 100).
            save_path (str, optional): If provided, path to save the animation (e.g., 'wigner.mp4' or 'wigner.gif').
            writer (str, optional): Animation writer backend ('ffmpeg', 'imagemagick', etc.).
        
        Returns:
            FuncAnimation: The animation object.
        """
        xvec = np.linspace(-5, 5, 100)  # Reduced resolution for faster computation
        total_frames = len(self.tlist)
        frame_step = max(1, total_frames // number_of_frames)  # Calculate frame_step based on desired number of frames
        selected_frames = np.arange(0, total_frames, frame_step)
        tlist_ns = self.tlist * 1e9  # ns for display

        # Precompute Wigner functions and global clim
        wigner_list = []
        global_clim = 0
        for frame in tqdm(selected_frames, desc="Computing Wigner frames"):
            rho = self.result.states[frame]
            wigner_frames = []
            for sys_idx in system_index:
                rho_reduced = qt.ptrace(rho, sys_idx)
                W = qt.wigner(rho_reduced, xvec, xvec)
                wigner_frames.append(W)
                global_clim = max(global_clim, np.max(np.abs(W)))
            wigner_list.append(wigner_frames)

        # Precompute multi-system pulses
        multi_pulse_waves = []
        multi_pulse_labels = []
        for pc in self.pulse_chains:
            if pc is not None and getattr(pc, "system", None) is None:
                for pulse in pc.pulses:
                    if not getattr(pulse, "multi_system", False):
                        continue
                    envelope = np.array([pulse.shape(t) if pulse.shape else 1.0 for t in self.tlist])
                    tone = np.cos(pulse.tone * self.tlist + getattr(pulse, "phase", 0))
                    pulse_wave = pulse.strength * envelope * tone
                    multi_pulse_waves.append(pulse_wave)
                    multi_pulse_labels.append(type(pulse).__name__)

        n_systems = len(system_index)
        n_multi = len(multi_pulse_waves)
        n_cols = n_systems
        n_rows = 2 if n_multi == 0 else 3

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
        plt.tight_layout()

        ims = []
        vlines = []

        # Initialize Wigner plots with global clim
        for i in range(n_systems):
            im = axes[0, i].imshow(wigner_list[0][i], extent=[-5, 5, -5, 5],
                                   aspect='auto', origin='lower', cmap='RdBu',
                                   vmin=-global_clim, vmax=global_clim)
            axes[0, i].set_title(f"System {system_index[i]} Wigner")
            axes[0, i].set_xlabel('x')
            axes[0, i].set_ylabel('p')
            ims.append(im)

        # Initialize pulse plots
        for i, sys_idx in enumerate(system_index):
            pc = self.pulse_chains[sys_idx] if sys_idx < len(self.pulse_chains) else None
            vline = None
            if pc is not None and getattr(pc, "system", None) is not None and pc.pulses:
                pulse = pc.pulses[0]
                envelope = np.array([pulse.shape(t) if pulse.shape else 1.0 for t in self.tlist])
                tone = np.cos(pulse.tone * self.tlist + getattr(pulse, "phase", 0))
                pulse_wave = pulse.strength * envelope * tone
                axes[1, i].plot(tlist_ns, pulse_wave)
                vline = axes[1, i].axvline(tlist_ns[0], color='r', linestyle='--')
            vlines.append(vline)

        # Multi-system pulses
        if n_multi > 0:
            for i in range(n_cols):
                for wave in multi_pulse_waves:
                    axes[2, i].plot(tlist_ns, wave)
                vline = axes[2, i].axvline(tlist_ns[0], color='r', linestyle='--')
                vlines.append(vline)

        def update(frame_idx):
            """
            Update function for animation.

            Args:
                frame_idx (int): Index of the current animation frame.

            Returns:
                list: List of updated artists.
            """
            t_ns = tlist_ns[selected_frames[frame_idx]]
            for i in range(n_systems):
                ims[i].set_data(wigner_list[frame_idx][i])
                axes[0, i].set_title(f"System {system_index[i]}")
                if vlines[i] is not None:
                    vlines[i].set_xdata([t_ns, t_ns])
                if n_multi > 0 and vlines[n_systems + i] is not None:
                    vlines[n_systems + i].set_xdata([t_ns, t_ns])
            return ims + [v for v in vlines if v is not None]

        ani = FuncAnimation(
            fig,
            update,
            frames=len(selected_frames),
            interval=1000 / speed,
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
