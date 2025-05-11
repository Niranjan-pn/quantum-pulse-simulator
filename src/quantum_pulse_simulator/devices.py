import qutip as qt

class QuantumSystem:
    """Base class for quantum systems with Fock space representation.
    
    Attributes:
        num_fock (int): Dimension of Fock space
        omega (float): Oscillator frequency in radians/second
        a (Qobj): Annihilation operator for the system
        H0 (Qobj): Base Hamiltonian of the system
        state (Qobj): Current quantum state (ket or density matrix)
    """

    def __init__(self, num_fock, omega, state=None):
        """
        Args:
            num_fock (int): Number of Fock states in Hilbert space
            omega (float): Oscillator frequency in radians/second
            state (Qobj, optional): Initial quantum state. Defaults to ground state.
        """
        self.num_fock = num_fock
        self.omega = omega
        self.a = qt.destroy(num_fock)
        self.H0 = omega * (self.a.dag() * self.a + 0.5)
        self.state = qt.basis(num_fock, 0) if state is None else state


class HarmonicOscillator(QuantumSystem):
    """Standard harmonic oscillator system.
    
    Inherits all attributes and methods from QuantumSystem with H0 = ħω(a†a + ½).
    """


class KerrOscillator(QuantumSystem):
    """Nonlinear oscillator with Kerr (fourth-order) nonlinearity.
    
    Attributes:
        Kerr (float): Kerr nonlinearity strength in radians/second
    """

    def __init__(self, num_fock, omega, Kerr, state=None):
        """
        Args:
            Kerr (float): Kerr nonlinearity strength in radians/second
        """
        super().__init__(num_fock, omega, state=state)
        a = qt.destroy(num_fock)
        self.H0 = omega * (a.dag() * a) 
        self.H0 -= Kerr * (a.dag() * a.dag() * a * a)


