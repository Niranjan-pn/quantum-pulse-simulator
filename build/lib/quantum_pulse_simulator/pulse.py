import numpy as np
import qutip as qt

class PulseSequence:
    """Unified interface for creating and managing quantum pulse sequences."""

    def __init__(self, systems):
        """
        Args:
            systems (list): List of quantum system objects (with .a, .omega, .num_fock, etc.)
        """
        self.systems = systems
        self.pulses = []

    # --- Envelope Shapes ---
    @staticmethod
    def sin2_shape(duration):
        return lambda t: np.sin(np.pi * t / duration) ** 2 if 0 <= t <= duration else 0.0

    @staticmethod
    def gaussian_shape(duration, sigma=None):
        if sigma is None:
            sigma = duration / 5
        center = duration / 2
        return lambda t: np.exp(-0.5 * ((t - center) / sigma) ** 2) if 0 <= t <= duration else 0.0

    @staticmethod
    def square_shape(duration):
        return lambda t: 1.0 if 0 <= t <= duration else 0.0

    @staticmethod
    def flattop_gaussian_shape(duration, rise_time, fall=True):
        def shape(t):
            if t < 0 or t > duration:
                return 0.0
            if rise_time <= 0:
                return 1.0
            edge_scale = 1 / (1 - np.exp(-2))
            if t < rise_time:
                x = (t - rise_time) / (rise_time / 2)
                return (np.exp(-0.5 * x**2) - np.exp(-2)) * edge_scale
            elif t > duration - rise_time:
                if not fall:
                    return 1.0
                x = (t - (duration - rise_time)) / (rise_time / 2)
                return (np.exp(-0.5 * x**2) - np.exp(-2)) * edge_scale
            else:
                return 1.0
        return shape

    # --- Add Standard Pulses ---

    def add_drive(self, duration, strength, system, order=1, tone=None, shape=None, phase=0.0, name="Drive"):
        """Generalized (a + a†)^n drive."""
        a = system.a
        op = (a + a.dag()) ** order
        if tone is None:
            tone = order * system.omega
        self._add_pulse(duration, strength, op, tone, shape, False, phase, name)

    def add_two_photon_drive(self, duration, strength, system, tone=None, shape=None, phase=0.0, name="TwoPhotonDrive"):
        a = system.a
        op = a**2 + a.dag()**2
        if tone is None:
            tone = 2 * system.omega
        self._add_pulse(duration, strength, op, tone, shape, False, phase, name)

    def add_trisqz(self, duration, strength, system, g3ac=None, tone=None, shape=None, phase=0.0, name="Trisqz"):
        a = system.a
        op = (a**3 + a.dag()**3) * (g3ac if g3ac is not None else 1) / np.sqrt(3)
        if tone is None:
            tone = 3 * system.omega
        self._add_pulse(duration, strength, op, tone, shape, False, phase, name)

    def add_beamsplitter(self, duration, strength, system1, system2, tone=None, shape=None, phase=0.0, name="BeamSplitter"):
        all_systems = self.systems
        a1 = system1.a
        a2 = system2.a
        # a1 a2_dag
        op_list1 = [qt.identity(s.num_fock) for s in all_systems]
        op_list1[all_systems.index(system1)] = a1
        op_list1[all_systems.index(system2)] = a2.dag()

        # a1_dag a2
        op_list2 = [qt.identity(s.num_fock) for s in all_systems]
        op_list2[all_systems.index(system1)] = a1.dag()
        op_list2[all_systems.index(system2)] = a2
        hamiltonian = qt.tensor(*op_list1) - qt.tensor(*op_list2)
    
        if tone is None:
            tone = abs(system1.omega - system2.omega)
        self._add_pulse(duration, strength, hamiltonian, tone, shape, True, phase, name)

    def add_second_order_beamsplitter(self, duration, strength, system1, system2, system3, tone=None, shape=None, phase=0.0, name="SecondOrderBeamSplitter"):
        """ Intraction : a1_dag a2_dag a3**2  + a1 a2 a3_dag**2 """
        all_systems = self.systems
        a1 = system1.a
        a2 = system2.a
        a3 = system3.a

        # a1_dag a2_dag a3**2
        op_list1 = [qt.identity(s.num_fock) for s in all_systems]
        op_list1[all_systems.index(system1)] = a1.dag()
        op_list1[all_systems.index(system2)] = a2.dag()
        op_list1[all_systems.index(system3)] = a3**2

        # a1 a2 a3_dag**2
        op_list2 = [qt.identity(s.num_fock) for s in all_systems]
        op_list2[all_systems.index(system1)] = a1
        op_list2[all_systems.index(system2)] = a2
        op_list2[all_systems.index(system3)] = a3.dag()**2

        hamiltonian = qt.tensor(*op_list1) + qt.tensor(*op_list2)

        if tone is None:
            tone = abs(system1.omega + system2.omega - 2 * system3.omega)
        self._add_pulse(duration, strength, hamiltonian, tone, shape, True, phase, name)

    def add_waiting(self, duration, name="Waiting"):
        hamiltonian = qt.tensor(*[qt.identity(s.num_fock) for s in self.systems])
        self._add_pulse(duration, 0, hamiltonian, 0, None, True, 0, name)

    # --- Custom Pulse ---
    def add_custom_drive(self, duration, strength, hamiltonian, tone=0.0, shape=None, phase=0.0, multi_system=False, name="CustomDrive"):
        """Add a user-defined pulse with arbitrary Hamiltonian."""
        self._add_pulse(duration, strength, hamiltonian, tone, shape, multi_system, phase, name)

    # --- Internal Pulse Storage ---
    def _add_pulse(self, duration, strength, hamiltonian, tone, shape, multi_system, phase, name):
        if shape is None:
            shape = lambda t: 1.0
        pulse = {
            "duration": duration,
            "strength": strength,
            "hamiltonian": hamiltonian,
            "tone": tone,
            "shape": shape,
            "multi_system": multi_system,
            "phase": phase,
            "name": name,
        }
        self.pulses.append(pulse)

    def total_duration(self):
        return sum(p["duration"] for p in self.pulses)

    def get_active_pulse(self, t):
        elapsed = 0
        for pulse in self.pulses:
            if elapsed <= t < elapsed + pulse["duration"]:
                return pulse, t - elapsed
            elapsed += pulse["duration"]
        return None, None

    def __repr__(self):
        return f"PulseSequence({len(self.pulses)} pulses, total duration={self.total_duration()}s)"
