import qutip as qt
import numpy as np
class QuantumSystem:
    """Unified quantum system class with flexible Hamiltonian composition.
    
    Attributes:
        num_fock (int): Dimension of Fock space
        omega (float): Base oscillator frequency (radians/second)
        a (Qobj): Annihilation operator
        H0 (Qobj): Total Hamiltonian
        state (Qobj): Current quantum state
        name (str): System identifier
    """

    def __init__(self, num_fock, omega=0, state=None, name=None):
        """
        Args:
            num_fock (int): Fock space dimension
            omega (float): Base frequency (default: 0)
            state (Qobj): Initial state (default: ground state)
            name (str): System name
        """
        self.num_fock = num_fock
        self.omega = omega
        self.a = qt.destroy(num_fock)
        self.state = qt.basis(num_fock, 0) if state is None else state
        self.name = name
        self.c_ops = []  # List of collapse operators
        self.add_harmonic_oscillator()  # Initialize with harmonic oscillator term

    def add_single_photon_loss(self, Kappa):
        """
        Add single photon loss collapse operator to the system.
        
        Args:
            Kappa (float): Loss rate.
        """
        self.c_ops.append(np.sqrt(Kappa) * self.a)
    
    def add_multi_photon_loss(self, Kappa, n_photons):
        """
        Add multi photon loss collapse operator to the system.
        
        Args:
            Kappa (float): Loss rate.
            n_photons (int): Number of photons to be lost.
        """
        self.c_ops.append(np.sqrt(Kappa) * self.a** n_photons)

    def add_harmonic_oscillator(self,):
        """Adds harmonic oscillator term: ħω(a†a + ½)"""
        self.H0 = self.omega * (self.a.dag() * self.a + 0.5)

    def add_kerr_oscillator(self, Kerr):
        """Adds Kerr nonlinearity: -Kerr(a†a†aa)"""
        self.H0 -= Kerr * (self.a.dag()**2 * self.a**2)


    def add_four_wave_mixer(self, g4):
        """Adds four-wave mixing term: g4(a + a†)^4"""
        self.H0 -= g4 * (self.a.dag()**4 * self.a**4)

    def add_static_nonlinearities(self, strengths):
        """
        Adds polynomial terms: Σ g_i(a + a†)^i
        
        Args:
            strengths (dict): {power: strength} pairs
        """
        for power, strength in strengths.items():
            self.H0 += strength * (self.a.dag() + self.a)**power

    def add_custom_hamiltonian(self, custom_H):
        """Adds user-provided Hamiltonian term (Qobj)"""
        self.H0 += custom_H
