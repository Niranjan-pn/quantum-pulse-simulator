import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import qutip as qt
from scipy.constants import hbar

# =========================
# Pulse Envelope Shapes
# =========================

def sin2_shape(duration):
    """Create a sin² envelope shape function for a pulse.
    
    Args:
        duration (float): Total duration of the pulse in seconds
        
    Returns:
        function: Time-dependent envelope function that returns:
            sin²(πt/duration) for 0 ≤ t ≤ duration, 0 otherwise
    """
    return lambda t: np.sin(np.pi * t / duration)**2 if 0 <= t <= duration else 0.0

def gaussian_shape(duration, sigma=None):
    """Create a Gaussian envelope shape function for a pulse.
    
    Args:
        duration (float): Total pulse duration in seconds
        sigma (float, optional): Gaussian standard deviation. Defaults to duration/5
        
    Returns:
        function: Gaussian envelope function centered at duration/2 with:
            exp(-½((t-center)/σ)²) for 0 ≤ t ≤ duration, 0 otherwise
    """
    if sigma is None:
        sigma = duration / 5
    center = duration / 2
    return lambda t: np.exp(-0.5 * ((t - center) / sigma)**2 if 0 <= t <= duration else 0.0)

def square_shape(duration):
    """Create a square (constant) envelope shape function.
    
    Args:
        duration (float): Pulse duration in seconds
        
    Returns:
        function: Constant 1.0 envelope for 0 ≤ t ≤ duration, 0 otherwise
    """
    return lambda t: 1.0 if 0 <= t <= duration else 0.0

def flattop_gaussian_shape(duration, rise_time):
    """Amplitude
        ^
    1.0 |       __________
        |      /          \
        |  ___/            \___
    0.0 |________________________
        0  ↑           ↑     duration
        rise_time   duration-rise_time

    Create a flattop Gaussian envelope with smooth 0→1 rise and 1→0 fall.
    
    Args:
        duration (float): Total pulse duration in seconds
        rise_time (float): Duration of rising/falling edges in seconds
        
    Returns:
        function: Envelope function with:
            - Smooth Gaussian rise from 0 to 1 during first rise_time
            - Constant 1.0 during flat top
            - Smooth Gaussian fall from 1 to 0 during last rise_time
    """
    def shape(t):
        if t < 0 or t > duration:
            return 0.0
        
        # Handle edge case with zero rise time
        if rise_time <= 0:
            return 1.0 if 0 <= t <= duration else 0.0
        
        # Scaling factor to normalize Gaussian edges (0 to 1)
        edge_scale = 1 / (1 - np.exp(-2))
        
        if t < rise_time:
            # Rising edge: Gaussian scaled to start at 0, reach 1 at rise_time
            x = (t - rise_time) / (rise_time/2)
            return (np.exp(-0.5 * x**2) - np.exp(-2)) * edge_scale
        elif t > duration - rise_time:
            # Falling edge: Gaussian scaled to start at 1, reach 0 at duration
            x = (t - (duration - rise_time)) / (rise_time/2)
            return (np.exp(-0.5 * x**2) - np.exp(-2)) * edge_scale
        else:
            # Flat top
            return 1.0
    return shape

# =========================
# Pulse Classes
# =========================

class Pulse:
    """General time-dependent Hamiltonian pulse for quantum systems.
    
    Attributes:
        duration (float): Pulse duration in seconds
        strength (float): Pulse amplitude scaling factor
        hamiltonian (qutip.Qobj): System Hamiltonian term
        tone (float): Frequency tone for time modulation (rad/s)
        shape (function): Envelope shape function
        multi_system (bool): True if pulse acts on multiple subsystems
        phase (float): Phase offset for time modulation (radians)
    """

    def __init__(self, duration, strength, hamiltonian, tone=0.0, shape=None, multi_system=False, phase=0.0):
        """Initialize a time-dependent Hamiltonian pulse.
        
        Args:
            duration (float): Pulse duration in seconds
            strength (float): Scaling factor for Hamiltonian term
            hamiltonian (qutip.Qobj): Quantum operator for the pulse
            tone (float, optional): Modulation frequency in rad/s. Default 0
            shape (function, optional): Envelope shape function. Default constant 1.0
            multi_system (bool, optional): True for multi-system pulses. Default False
            phase (float, optional): Phase offset in radians. Default 0
        """
        self.duration = duration
        self.strength = strength
        self.hamiltonian = hamiltonian
        self.tone = tone
        self.shape = shape if shape is not None else (lambda t: 1.0)
        self.multi_system = multi_system
        self.phase = phase

    def time_dep_coeff(self, t, t0):
        """Calculate time-dependent coefficient for Hamiltonian term.
        
        Args:
            t (float): Current simulation time
            t0 (float): Pulse start time
            
        Returns:
            float: Time-dependent coefficient value at time t
        """
        if t0 <= t < (t0 + self.duration):
            envelope = self.shape(t - t0)
            return self.strength * envelope * np.cos(self.tone * t + self.phase)
        return 0.0

# =========================
# Pulse Chain
# =========================

class PulseChain:
    """Container for sequencing pulses on a quantum system.
    
    Attributes:
        system (object): Target quantum system
        pulses (list): Sequence of Pulse objects
    """

    def __init__(self, system):
        """Initialize empty pulse chain for a quantum system.
        
        Args:
            system (object): Target quantum system with num_fock attribute
        """
        self.system = system
        self.pulses = []

    def add_pulse(self, pulse):
        """Add a pulse to the chain.
        
        Args:
            pulse (Pulse): Pulse object to append to sequence
        """
        self.pulses.append(pulse)

    def add_empty_pulse(self, duration):
        """Add zero-strength pulse of specified duration.
        
        Args:
            duration (float): Duration of empty pulse in seconds
        """
        zero_ham = qt.Qobj(np.zeros((self.system.num_fock, self.system.num_fock)))
        self.add_pulse(Pulse(duration, 0, zero_ham))

    def total_duration(self):
        """Calculate total duration of all pulses in chain.
        
        Returns:
            float: Sum of all pulse durations in seconds
        """
        return sum(p.duration for p in self.pulses)

    def get_active_pulse(self, t):
        """Find active pulse at time t and its relative time.
        
        Args:
            t (float): Current time in pulse chain
            
        Returns:
            tuple: (Pulse object, relative time) if found, (None, None) otherwise
        """
        elapsed = 0
        for pulse in self.pulses:
            if elapsed <= t < elapsed + pulse.duration:
                return pulse, t - elapsed
            elapsed += pulse.duration
        return None, None


class GeneralizedDrivePulse(Pulse):
    """Drive pulse for (a + a†)^n operator with frequency tone nω.
    
    Attributes:
        order (int): Expansion order of (a + a†) operator
    """

    def __init__(self, duration, strength, sys, order=1, tone=None, shape=None, phase=0.0):
        """Initialize generalized drive pulse.
        
        Args:
            duration (float): Pulse duration in seconds
            strength (float): Drive strength
            sys (object): Quantum system with a (annihilation) and omega (frequency)
            order (int, optional): Expansion order of (a + a†). Default 1
            tone (float, optional): Drive frequency. Defaults to order * sys.omega
            shape (function, optional): Envelope shape. Default constant 1.0
            phase (float, optional): Phase offset. Default 0
        """
        a = sys.a
        op = (a + a.dag()) ** order
        if tone is None:
            tone = order * sys.omega
        super().__init__(duration, strength, op, tone, shape, multi_system=False, phase=phase)


class TrisqzPulse(Pulse):
    """Third-order nonlinear pulse with (a³ + a†³) operator at 3ω."""

    def __init__(self, duration, strength, sys, tone=None, shape=None, phase=0.0):
        """Initialize trisqueezing pulse.
        
        Args:
            duration (float): Pulse duration in seconds
            strength (float): Pulse strength
            sys (object): Quantum system with a and g3ac (third-order nonlinearity)
            tone (float, optional): Pulse frequency. Defaults to 3 * sys.omega
            shape (function, optional): Envelope shape. Default constant 1.0
            phase (float, optional): Phase offset. Default 0
        """
        a = sys.a
        if sys.g3ac is not None:
            op = (a**3 + a.dag()**3) * sys.g3ac / np.sqrt(3)
        else:
            op = (a**3 + a.dag()**3) / np.sqrt(3)
        if tone is None:
            tone = 3 * sys.omega
        super().__init__(duration, strength, op, tone, shape, multi_system=False, phase=phase)


class BeamSplitterPulse(Pulse):
    """Beamsplitter interaction between two modes with tone |ω1 - ω2|."""

    def __init__(self, duration, strength, all_systems, sys1_index, sys2_index, tone=None, shape=None, phase=0.0):
        """Initialize beamsplitter pulse.
        
        Args:
            duration (float): Pulse duration in seconds
            strength (float): Interaction strength
            all_systems (list): List of all quantum systems
            sys1_index (int): Index of first system in all_systems
            sys2_index (int): Index of second system in all_systems
            tone (float, optional): Frequency tone. Defaults to |ω1 - ω2|
            shape (function, optional): Envelope shape. Default constant 1.0
            phase (float, optional): Phase offset. Default 0
        """
        H_full = [qt.identity(_.num_fock) for _ in all_systems]
        sys1 = all_systems[sys1_index]
        sys2 = all_systems[sys2_index]
        
        a1 = sys1.a
        a2 = sys2.a
        H_full[sys1_index] = a1 + a1.dag()
        H_full[sys2_index] = a2 + a2.dag()
        
        hamiltonian = qt.tensor(*H_full)
        if tone is None:
            tone = abs(sys1.omega - sys2.omega)
        super().__init__(duration, strength, hamiltonian, tone, shape, multi_system=True, phase=phase)


class ThreemodeBeamSplitterPulse(Pulse):
    """Three-mode nonlinear interaction pulse."""

    def __init__(self, duration, strength, all_systems, sys1_index, sys2_index, sys3_index, tone=None, shape=None, phase=0.0):
        """Initialize three-mode interaction pulse.
        
        Args:
            duration (float): Pulse duration in seconds
            strength (float): Interaction strength
            all_systems (list): List of all quantum systems
            sys1_index (int): Index of first system
            sys2_index (int): Index of second system 
            sys3_index (int): Index of third system
            tone (float, optional): Frequency tone. Defaults to |ω1 + ω3 - 2ω2|
            shape (function, optional): Envelope shape. Default constant 1.0
            phase (float, optional): Phase offset. Default 0
        """
        H_full = [qt.identity(_.num_fock) for _ in all_systems]
        sys1 = all_systems[sys1_index]
        sys2 = all_systems[sys2_index]
        sys3 = all_systems[sys3_index]
        
        a1 = sys1.a
        a2 = sys2.a
        a3 = sys3.a
        H_full[sys1_index] = a1 + a1.dag()
        H_full[sys2_index] = (a2 + a2.dag())**2
        H_full[sys3_index] = a3 + a3.dag()
        
        hamiltonian = qt.tensor(*H_full)
        if tone is None:
            tone = abs(sys1.omega + sys3.omega - 2*sys2.omega)
        super().__init__(duration, strength, hamiltonian, tone, shape, multi_system=True, phase=phase)


class TwoPhotonDrivePulse(Pulse):
    """Two-photon drive pulse with (a² + a†²) operator at 2ω."""

    def __init__(self, duration, strength, sys, tone=None, shape=None, phase=0.0):
        """Initialize two-photon drive pulse.
        
        Args:
            duration (float): Pulse duration in seconds
            strength (float): Drive strength
            sys (object): Quantum system with a and omega
            tone (float, optional): Drive frequency. Defaults to 2 * sys.omega
            shape (function, optional): Envelope shape. Default constant 1.0
            phase (float, optional): Phase offset. Default 0
        """
        a = sys.a
        op = a**2 + a.dag()**2
        if tone is None:
            tone = 2 * sys.omega
        super().__init__(duration, strength, op, tone, shape, multi_system=False, phase=phase)
