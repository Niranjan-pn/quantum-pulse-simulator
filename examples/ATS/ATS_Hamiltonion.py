"""
This script is a copy of script available in qt-codebase. Always manually copy the script from qt-codebase to this location until this simualtor is merged with qt-codebase.
"""

import jax
from jax import grad, jit
from jax import config as jax_config; jax_config.update("jax_enable_x64", True)

import jax.numpy as np
from jax.scipy.special import factorial
import matplotlib.pyplot as plt
from scipy import constants as ct
from scipy.optimize import curve_fit, minimize
from matplotlib.lines import Line2D
from typing import Callable, Tuple, Union, Optional, List
from pathlib import Path
import json
import pandas as pd
import numpy as onp
import time
from scipy.constants import h, e

flux_q = h/(2*e)

def get_jj_inductance_from_energy(energy_rad_s: float):
    """
    Calculate the Josephson junction inductance from the Josephson energy.

    Args:
        energy_rad_s (float): Josephson energy in angular-frequency units (E/Ä§), rad/s.

    Returns:
        float: Josephson junction inductance in Henries.
    """
    return (flux_q/(2*np.pi))**2/np.abs(h*energy_rad_s/2/np.pi)

def get_Ec_from_C(C: float) -> float:
    """Calculate the charging energy Ec from capacitance C.
    Definition from https://arxiv.org/pdf/1904.06560 : Check equation 13
    Args:
        C (float): Capacitance in Farads.

    Returns:
        float: Charging energy Ec in Joules.
    """
    return ct.e**2 / 2 / C / ct.hbar


def newton_minimize(U_func: Callable[[float, float, float], float], phi_plus: float, phi_minus: float, init: float = 0.0, max_iter: int = 50) -> float:
    """Find the minimum of a function using Newton's method.

    Args:
        U_func (callable): Function to minimize. Should take (phi, phi_plus, phi_minus).
        phi_plus (float): Value of phi_plus.
        phi_minus (float): Value of phi_minus.
        init (float, optional): Initial guess for phi. Defaults to 0.0.
        max_iter (int, optional): Maximum number of iterations. Defaults to 50.

    Returns:
        float: Value of phi at the minimum.
    """
    phi = init
    for _ in range(max_iter):
        grad1 = jax.grad(U_func, argnums=0)(phi, phi_plus, phi_minus)
        grad2 = jax.grad(jax.grad(U_func, argnums=0), argnums=0)(phi, phi_plus, phi_minus)
        phi = phi - grad1 / grad2
    return phi


class ATS:
    """Ansymmetrically threaded squids (ATS) model for circuit quantum electrodynamics."""

    def __init__(self, Ej: float, El: float, Er: float, Ec: float, N: int,
                 A_L: float = 3.63e-5*1e-9, B_L: float = 5.01e-5,
                 A_R: float = 3.63e-5*1e-9, B_R: float = 5.01e-5,
                 A_plus: float = 3.63e-5*1e-9, B_plus: float = 5.01e-5,
                 A_minus: float = 3.63e-5*1e-9, B_minus: float = 5.01e-5,
                 A_cross_lr: float = 3.63e-5*1e-9,
                 A_cross_pm: float = 3.63e-5*1e-9,
                 B_cross_lr: float = 5.01e-5,
                 B_cross_pm: float = 5.01e-5,
                 T1: float = 50e-6) -> None:
        """Initialize the ATS model.

        Args:
            Ej (float): Josephson energy.
            El (float): Left inductive energy.
            Er (float): Right inductive energy.
            Ec (float): Charging energy.
            N (int): Number of junctions.
            A_L (float, optional): 1/f noise prefactor for left. Defaults to 3.63e-5*1e-9.
            B_L (float, optional): White noise prefactor for left. Defaults to 5.01e-5.
            A_R (float, optional): 1/f noise prefactor for right. Defaults to 3.63e-5*1e-9.
            B_R (float, optional): White noise prefactor for right. Defaults to 5.01e-5.
            A_plus (float, optional): 1/f noise prefactor for plus. Defaults to 3.63e-5*1e-9.
            B_plus (float, optional): White noise prefactor for plus. Defaults to 5.01e-5.
            A_minus (float, optional): 1/f noise prefactor for minus. Defaults to 3.63e-5*1e-9.
            B_minus (float, optional): White noise prefactor for minus. Defaults to 5.01e-5.
            A_cross_lr (float, optional): Cross noise prefactor left-right. Defaults to 3.63e-5*1e-9.
            A_cross_pm (float, optional): Cross noise prefactor plus-minus. Defaults to 3.63e-5*1e-9.
            B_cross_lr (float, optional): Cross noise prefactor left-right. Defaults to 5.01e-5.
            B_cross_pm (float, optional): Cross noise prefactor plus-minus. Defaults to 5.01e-5.
            T1 (float, optional): Energy relaxation time (seconds). Defaults to 50e-6.
        """
        self.Ej = Ej
        self.El = El
        self.Er = Er
        self.N = N
        self.Ec = Ec
        self.Eplus = Er + El
        self.Eminus = Er - El
        self.d = self.Eminus / self.Eplus
        self.alpha_max = self.Eplus / self.Ej
        self.A_L = A_L
        self.B_L = B_L
        self.A_R = A_R
        self.B_R = B_R
        self.A_plus = A_plus
        self.B_plus = B_plus
        self.A_minus = A_minus
        self.B_minus = B_minus
        self.A_cross_lr = A_cross_lr
        self.A_cross_pm = A_cross_pm
        self.B_cross_lr = B_cross_lr
        self.B_cross_pm = B_cross_pm
        self.T1 = T1

    @staticmethod
    def get_phi_plus_minus(phi_L: Union[float, np.ndarray], phi_R: Union[float, np.ndarray]) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """Convert left/right phases to plus/minus basis.

        Args:
            phi_L (float or ndarray): Left phase.
            phi_R (float or ndarray): Right phase.

        Returns:
            tuple: (phi_plus, phi_minus)
        """
        return (phi_R + phi_L) / 2.0, (phi_R - phi_L) / 2.0

    def U_plus_minus_basis(self, phi: float, phi_plus: float, phi_minus: float) -> float:
        """Potential energy in the plus/minus basis.

        Args:
            phi (float): Phase variable.
            phi_plus (float): Plus phase.
            phi_minus (float): Minus phase.

        Returns:
            float: Potential energy.
        """
        term1 = -self.Eplus * np.cos(phi - phi_plus) * np.cos(phi_minus)
        term2 = -self.Eminus * np.sin(phi - phi_plus) * np.sin(phi_minus)
        term3 = -self.N * self.Ej * np.cos(phi / self.N)
        return term1 + term2 + term3

    def U_left_right_basis(self, phi: float, phi_L: float, phi_R: float) -> float:
        """Potential energy in the left/right basis.

        Args:
            phi (float): Phase variable.
            phi_L (float): Left phase.
            phi_R (float): Right phase.

        Returns:
            float: Potential energy.
        """
        return -self.Er * np.cos(phi - phi_R) \
             - self.El * np.cos(phi - phi_L) \
             - self.N * self.Ej * np.cos(phi / self.N)

    def di(self, order: int) -> Callable[[float, float, float], float]:
        """Return the i-th derivative function of the potential with respect to phi.

        Args:
            order (int): Order of the derivative.

        Returns:
            callable: Function that computes the i-th derivative.
        """
        f = self.U_plus_minus_basis
        for i in range(order):
            f = jax.grad(f, argnums=0)
        return f

    def Z(self, phi_min: float, phi_plus: float, phi_minus: float) -> float:
        """Calculate the Impedance.

        Args:
            phi_min (float): Phase at minimum.
            phi_plus (float): Plus phase.
            phi_minus (float): Minus phase.

        Returns:
            float: Impedance.
        """
        d2U = self.di(2)(phi_min, phi_plus, phi_minus)
        return np.sqrt(8*self.Ec / d2U)

    def gi(self, order: int, phi_min: float, phi_plus: float, phi_minus: float) -> float:
        """Calculate the i-th order nonlinearity coefficient.

        Args:
            order (int): Order of the nonlinearity.
            phi_min (float): Phase at minimum.
            phi_plus (float): Plus phase.
            phi_minus (float): Minus phase.

        Returns:
            float: Nonlinearity coefficient.
        """
        di = self.di(order)(phi_min, phi_plus, phi_minus)
        Z = self.Z(phi_min, phi_plus, phi_minus)
        return di * (Z / 2) ** (order / 2) / factorial(order)

    def omega_renormalised(self, phi_min: float, phi_plus: float, phi_minus: float) -> float:
        """Calculate the renormalized frequency including Kerr correction.

        Args:
            phi_min (float): Phase at minimum.
            phi_plus (float): Plus phase.
            phi_minus (float): Minus phase.

        Returns:
            float: Renormalized frequency.
        """
        d2U = self.di(2)(phi_min, phi_plus, phi_minus)
        g3 = self.gi(3, phi_min, phi_plus, phi_minus)
        g4 = self.gi(4, phi_min, phi_plus, phi_minus)
        omega = np.sqrt(8*self.Ec * d2U)
        kerr = 12 * (g4 - 5 * g3 ** 2 / omega)
        return omega + kerr

    def kerr(self, phi_min: float, phi_plus: float, phi_minus: float) -> float:
        """Calculate the Kerr nonlinearity.
        Kerr = 12 * (g4 - 5 * g3 ** 2 / omega)

        Args:
            phi_min (float): Phase at minimum.
            phi_plus (float): Plus phase.
            phi_minus (float): Minus phase.

        Returns:
            float: Kerr nonlinearity.
        """
        d2U = self.di(2)(phi_min, phi_plus, phi_minus)
        g3 = self.gi(3, phi_min, phi_plus, phi_minus)
        g4 = self.gi(4, phi_min, phi_plus, phi_minus)
        omega = np.sqrt(8*self.Ec * d2U)
        return 12 * (g4 - 5 * g3 ** 2 / omega)

    def g3_g4_g6(self, phi_L: float, phi_R: float, phi_min: Optional[float] = None) -> Tuple[float, float]:
        """Convenience helper to compute (g3, g4) at a given left/right flux bias.

        Args:
            phi_L (float): Left phase bias.
            phi_R (float): Right phase bias.
            phi_min (float, optional): Phase that minimizes the potential. If not provided,
                it is found via ``newton_minimize``.

        Returns:
            tuple: (g3, g4) nonlinearity coefficients.
        """
        phi_plus, phi_minus = self.get_phi_plus_minus(phi_L, phi_R)
        if phi_min is None:
            phi_min = newton_minimize(self.U_plus_minus_basis, phi_plus, phi_minus)
        g3 = self.gi(3, phi_min, phi_plus, phi_minus)
        g4 = self.gi(4, phi_min, phi_plus, phi_minus)
        g6 = self.gi(6, phi_min, phi_plus, phi_minus)
        return g3, g4,g6

    @staticmethod
    def _tilde_omega(omega_a: float, omega_b: float, omega_c: float) -> float:
        """Helper for the effective frequency combination used in cross-Kerr."""
        terms = [
            omega_a - omega_b - omega_c,
            -omega_a + omega_b - omega_c,
            omega_a + omega_b - omega_c,
            -omega_a - omega_b - omega_c,
        ]
        inv_sum = sum(1.0 / t for t in terms)
        return 1.0 / inv_sum

    def crosskerr(
        self,
        phi_L: float,
        phi_R: float,
        omega_a: float,
        omega_b: float,
        g_a: float,
        g_b: float,
        phi_min: Optional[float] = None,
    ) -> float:
        """Calculate cross-Kerr chi_ab using provided mode frequencies and couplings.

        chi_ab â‰ˆ (24 g4 + 36 g3^2 / w_tilde) * (g_a/Î”_a)^2 * (g_b/Î”_b)^2

        Args:
            phi_L (float): Left phase bias.
            phi_R (float): Right phase bias.
            omega_a (float): Frequency of mode a (rad/s).
            omega_b (float): Frequency of mode b (rad/s).
            omega_c (float, optional): ATS (coupler) frequency (rad/s). If None,
                it is computed from the ATS at the given bias.
            g_a (float): Coupling of mode a to the ATS.
            g_b (float): Coupling of mode b to the ATS.
            Delta_a (float): Detuning for mode a.
            Delta_b (float): Detuning for mode b.
            phi_min (float, optional): Phase that minimizes the potential. If omitted,
                it is found via ``newton_minimize``.

        Returns:
            float: Cross-Kerr chi_ab.
        """
        phi_plus, phi_minus = self.get_phi_plus_minus(phi_L, phi_R)
        if phi_min is None:
            phi_min = newton_minimize(self.U_plus_minus_basis, phi_plus, phi_minus)
        g3, g4, g6 = self.g3_g4_g6(phi_L, phi_R, phi_min=phi_min)
        omega_c = self.omega_renormalised(phi_min, phi_plus, phi_minus)
        Delta_a = omega_c - omega_a
        Delta_b = omega_c - omega_b
        w_tilde = self._tilde_omega(omega_a, omega_b, omega_c)
        prefactor = 24.0 * g4 + 36.0 * (g3 ** 2) / w_tilde
        g6_cont=180*g6*((g_a / Delta_a) ** 2 + (g_b / Delta_b) ** 2)
        prefactor+=g6_cont
        return prefactor * (g_a / Delta_a) ** 2 * (g_b / Delta_b) ** 2

    def gi_ac_plus(self, i: int, phi_min: float, phi_plus: float, phi_minus: float) -> float:
        """Calculate the i-th order AC nonlinearity coefficient with respect to phi_plus.

        Args:
            i (int): Order of the derivative.
            phi_min (float): Phase at minimum.
            phi_plus (float): Plus phase.
            phi_minus (float): Minus phase.

        Returns:
            float: AC nonlinearity coefficient for phi_plus.
        """
        def mixed_derive(phi, phi_plus, phi_minus):
            f = self.U_plus_minus_basis
            for _ in range(i):
                f = jax.grad(f, argnums=0)
            f = jax.grad(f, argnums=1)
            return f(phi, phi_plus, phi_minus)
        Z = self.Z(phi_min, phi_plus, phi_minus)
        factorial_term = factorial(i + 1)
        result = 1 / factorial_term * (Z / 2) ** (i / 2) * mixed_derive(phi_min, phi_plus, phi_minus)
        return result

    def gi_ac_minus(self, i: int, phi_min: float, phi_plus: float, phi_minus: float) -> float:
        """Calculate the i-th order AC nonlinearity coefficient with respect to phi_minus.

        Args:
            i (int): Order of the derivative.
            phi_min (float): Phase at minimum.
            phi_plus (float): Plus phase.
            phi_minus (float): Minus phase.

        Returns:
            float: AC nonlinearity coefficient for phi_minus.
        """
        def mixed_derive(phi, phi_plus, phi_minus):
            f = self.U_plus_minus_basis
            for _ in range(i):
                f = jax.grad(f, argnums=0)
            f = jax.grad(f, argnums=2)
            return f(phi, phi_plus, phi_minus)
        Z = self.Z(phi_min, phi_plus, phi_minus)
        factorial_term = factorial(i + 1)
        result = 1 / factorial_term * (Z / 2) ** (i / 2) * mixed_derive(phi_min, phi_plus, phi_minus)
        return result
    
    def get_sweet_spot(self, phi_L_range: Tuple[float, float] = (-2.5*onp.pi, 2.5*onp.pi),
                       phi_R_range: Tuple[float, float] = (-2.5*onp.pi, 2.5*onp.pi),
                       grid_points: int = 30,
                       n_initial_points: int = 15,
                       weight_kerr: float = 1.0,
                       weight_odd: float = 1.0,
                       weight_even: float = 1.0,
                       cost_threshold: float = 1e-3,
                       use_gradient: bool = True) -> pd.DataFrame:
        """Find sweet spots by minimizing a cost function containing Kerr, odd sum, and even sum.
        
        This function:
        1. Creates a cost function = weight_kerr*|Kerr|Â² + weight_odd*|odd_sum|Â² + weight_even*|even_sum|Â²
        2. Uses multiple initial guesses from a grid to find local minima
        3. Returns a table with optimized coordinates and circuit parameters
        
        Args:
            phi_L_range (tuple, optional): (min, max) for left phase search space. Defaults to (-2.5Ï€, 2.5Ï€).
            phi_R_range (tuple, optional): (min, max) for right phase search space. Defaults to (-2.5Ï€, 2.5Ï€).
            grid_points (int, optional): Number of grid points per dimension for initial search. Defaults to 30.
            n_initial_points (int, optional): Number of best initial points to start optimization from. Defaults to 15.
            weight_kerr (float, optional): Weight for Kerr in cost function. Defaults to 1.0.
            weight_odd (float, optional): Weight for odd sum in cost function. Defaults to 1.0.
            weight_even (float, optional): Weight for even sum in cost function. Defaults to 1.0.
            cost_threshold (float, optional): Threshold for accepting a sweet spot (normalized cost). Defaults to 1e-3.
            use_gradient (bool, optional): Use gradient-based optimization (faster but needs JAX). Defaults to True.
            
        Returns:
            pd.DataFrame: Table with sweet spot coordinates as columns and circuit parameters as rows.
        """
        
        # JIT-compiled cost function for speed
        @jit
        def cost_function_jax(phi_L: float, phi_R: float) -> float:
            """JAX-optimized cost function."""
            phi_plus, phi_minus = self.get_phi_plus_minus(phi_L, phi_R)
            
            # Find minimum of potential
            phi_min = newton_minimize(self.U_plus_minus_basis, phi_plus, phi_minus)
            
            # Calculate quantities
            kerr_val = self.kerr(phi_min, phi_plus, phi_minus)
            g5 = self.gi(5, phi_min, phi_plus, phi_minus)
            g7 = self.gi(7, phi_min, phi_plus, phi_minus)
            g9 = self.gi(9, phi_min, phi_plus, phi_minus)
            goddsum_val = g5 + g7 + g9
            
            g4 = self.gi(4, phi_min, phi_plus, phi_minus)
            g6 = self.gi(6, phi_min, phi_plus, phi_minus)
            g8 = self.gi(8, phi_min, phi_plus, phi_minus)
            gevensum_val = g4 + g6 + g8
            
            # Normalize by MHz scale (convert rad/s -> Hz first)
            kerr_norm = kerr_val / (2 * np.pi * 1e6)
            goddsum_norm = goddsum_val / (2 * np.pi * 1e6)
            gevensum_norm = gevensum_val / (2 * np.pi * 1e6)
            
            # Cost function
            cost = (weight_kerr * kerr_norm**2 + 
                   weight_odd * goddsum_norm**2 + 
                   weight_even * gevensum_norm**2)
            
            return cost
        
        # Wrapper for scipy that takes 1D array
        def cost_function_scipy(phi_lr: onp.ndarray) -> float:
            """Wrapper for scipy.optimize."""
            return float(cost_function_jax(phi_lr[0], phi_lr[1]))
        
        # Gradient function (if using gradient-based optimization)
        if use_gradient:
            grad_cost_jax = jit(grad(cost_function_jax, argnums=(0, 1)))
            
            def grad_cost_scipy(phi_lr: onp.ndarray) -> onp.ndarray:
                """Gradient wrapper for scipy."""
                grads = grad_cost_jax(phi_lr[0], phi_lr[1])
                return onp.array([float(grads[0]), float(grads[1])])
        else:
            grad_cost_scipy = None
        
        start_time = time.time()
        
        # Step 1: Create grid from ranges
        phi_vals_L = np.linspace(phi_L_range[0], phi_L_range[1], grid_points)
        phi_vals_R = np.linspace(phi_R_range[0], phi_R_range[1], grid_points)
        
        # Step 2: Coarse grid search to find promising initial points
        print(f"Performing initial grid search ({grid_points}x{grid_points} = {grid_points**2} points)...")
        grid_start = time.time()
        phi_L_grid, phi_R_grid, omega, kerr, goddsum, gevensum, g2ac_plus, g2ac_minus, g3ac_plus, g3ac_minus, T2_left_right, T2_plus_minus = self.grid_calculate(phi_vals_L, phi_vals_R)
        print(f"  Grid search completed in {time.time() - grid_start:.1f}s")
        
        # Calculate cost on grid
        kerr_norm = kerr / (2 * np.pi * 1e6)
        goddsum_norm = goddsum / (2 * np.pi * 1e6)
        gevensum_norm = gevensum / (2 * np.pi * 1e6)
        cost_grid = (weight_kerr * kerr_norm**2 + 
                    weight_odd * goddsum_norm**2 + 
                    weight_even * gevensum_norm**2)
        
        # Find the best initial points
        flat_costs = cost_grid.flatten()
        flat_indices = onp.argsort(flat_costs)[:n_initial_points]
        
        # Convert flat indices back to 2D
        initial_points = []
        for flat_idx in flat_indices:
            i, j = onp.unravel_index(flat_idx, cost_grid.shape)
            initial_points.append([float(phi_L_grid[i, j]), float(phi_R_grid[i, j])])
        
        # Step 3: Optimize from each initial point
        print(f"Optimizing from {n_initial_points} initial points...")
        optimized_points = []
        optimized_costs = []
        
        # JIT compile functions before optimization (warmup)
        print("Warming up JIT compilation...")
        warmup_start = time.time()
        _ = cost_function_scipy(onp.array([initial_points[0][0], initial_points[0][1]]))
        if use_gradient:
            _ = grad_cost_scipy(onp.array([initial_points[0][0], initial_points[0][1]]))
        print(f"  JIT compilation completed in {time.time() - warmup_start:.1f}s")
        
        opt_start = time.time()
        print(f"Starting optimization from {n_initial_points} initial points...")
        
        for idx, init_point in enumerate(initial_points):
            if (idx + 1) % 5 == 0 or idx == 0:
                elapsed = time.time() - opt_start
                print(f"  Progress: {idx + 1}/{n_initial_points} (elapsed: {elapsed:.1f}s)...")
            
            # Choose optimization method based on use_gradient
            if use_gradient:
                result = minimize(
                    cost_function_scipy,
                    init_point,
                    method='L-BFGS-B',
                    jac=grad_cost_scipy,
                    bounds=[(phi_L_range[0], phi_L_range[1]), (phi_R_range[0], phi_R_range[1])],
                    options={'maxiter': 100, 'ftol': 1e-9}
                )
            else:
                result = minimize(
                    cost_function_scipy,
                    init_point,
                    method='Nelder-Mead',
                    bounds=[(phi_L_range[0], phi_L_range[1]), (phi_R_range[0], phi_R_range[1])],
                    options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-9}
                )
            
            if result.success and result.fun < cost_threshold:
                optimized_points.append(result.x)
                optimized_costs.append(result.fun)
        
        print(f"  Optimization completed in {time.time() - opt_start:.1f}s")
        
        if len(optimized_points) == 0:
            print(f"No sweet spots found with cost < {cost_threshold}.")
            print("Try increasing cost_threshold or adjusting weights.")
            return pd.DataFrame()
        
        # Step 4: Remove duplicates (points very close to each other)
        unique_points = []
        unique_costs = []
        tolerance = 0.1  # tolerance in radians
        
        for point, cost in zip(optimized_points, optimized_costs):
            is_duplicate = False
            for existing_point in unique_points:
                if onp.linalg.norm(onp.array(point) - onp.array(existing_point)) < tolerance:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_points.append(point)
                unique_costs.append(cost)
        
        print(f"Found {len(unique_points)} unique sweet spots.")
        
        # Step 5: Calculate all parameters for each sweet spot
        sweet_spots_data = {}
        
        for idx, (point, cost) in enumerate(zip(unique_points, unique_costs)):
            phi_L_val, phi_R_val = point[0], point[1]
            phi_plus, phi_minus = self.get_phi_plus_minus(phi_L_val, phi_R_val)
            phi_min = newton_minimize(self.U_plus_minus_basis, phi_plus, phi_minus)
            
            d2U_dphi2 = self.di(2)(phi_min, phi_plus, phi_minus)
            # Calculate all quantities
            omega_val = self.omega_renormalised(phi_min, phi_plus, phi_minus)
            Z_val = self.Z(phi_min, phi_plus, phi_minus)
            kerr_val = self.kerr(phi_min, phi_plus, phi_minus)
            
            g5 = self.gi(5, phi_min, phi_plus, phi_minus)
            g7 = self.gi(7, phi_min, phi_plus, phi_minus)
            g9 = self.gi(9, phi_min, phi_plus, phi_minus)
            goddsum_val = g5 + g7 + g9
            
            g4 = self.gi(4, phi_min, phi_plus, phi_minus)
            g6 = self.gi(6, phi_min, phi_plus, phi_minus)
            g8 = self.gi(8, phi_min, phi_plus, phi_minus)
            gevensum_val = g4 + g6 + g8
            
            g2ac_plus_val = self.gi_ac_plus(2, phi_min, phi_plus, phi_minus)
            g2ac_minus_val = self.gi_ac_minus(2, phi_min, phi_plus, phi_minus)
            g3ac_plus_val = self.gi_ac_plus(3, phi_min, phi_plus, phi_minus)
            g3ac_minus_val = self.gi_ac_minus(3, phi_min, phi_plus, phi_minus)
            g4ac_plus_val = self.gi_ac_plus(4, phi_min, phi_plus, phi_minus)
            g4ac_minus_val = self.gi_ac_minus(4, phi_min, phi_plus, phi_minus)
            
            # Column name with coordinates rounded to 2 decimals
            col_name = f"({phi_L_val/onp.pi:.2f}pi, {phi_R_val/onp.pi:.2f}pi)"
            kerr_radius = np.sqrt((np.abs(phi_L_val/np.pi)-1)**2 + (np.abs(phi_R_val/np.pi)-1)**2)
            
            sweet_spots_data[col_name] = {
                'kerr_radius': round(float(kerr_radius), 3),
                'omega (GHz)': round(float(omega_val / 1e9 / 2 / onp.pi), 3),
                'Z (Ohm)': round(float(Z_val), 3),
                'Kerr (MHz)': round(float(kerr_val / (2 * onp.pi * 1e6)), 3),
                'Odd sum (MHz)': round(float(goddsum_val / (2 * onp.pi * 1e6)), 3),
                'Even sum (MHz)': round(float(gevensum_val / (2 * onp.pi * 1e6)), 3),
                'g2ac+ (MHz)': round(float(g2ac_plus_val / (2 * onp.pi * 1e6)), 3),
                'g2ac- (MHz)': round(float(g2ac_minus_val / (2 * onp.pi * 1e6)), 3),
                'g3ac+ (MHz)': round(float(g3ac_plus_val / (2 * onp.pi * 1e6)), 3),
                'g3ac- (MHz)': round(float(g3ac_minus_val / (2 * onp.pi * 1e6)), 3),
                'g4ac+ (MHz)': round(float(g4ac_plus_val / (2 * onp.pi * 1e6)), 3),
                'g4ac- (MHz)': round(float(g4ac_minus_val / (2 * onp.pi * 1e6)), 3),
                'Cost': round(float(cost), 3),
                'Lj (nH)': round(float(get_jj_inductance_from_energy(d2U_dphi2) * 1e9), 3),
            }
        
        # Create DataFrame
        df = pd.DataFrame(sweet_spots_data)
        
        total_time = time.time() - start_time
        print(f"\nTotal time: {total_time:.1f}s")
        print(f"  Grid search: {time.time() - grid_start:.1f}s")
        print(f"  Optimization: {time.time() - opt_start:.1f}s")
        
        return df

    def domega_dparam(self, U_func: Callable, phi_min: float, *args: float, param_index: int) -> Tuple[float, float]:
        """Compute the derivative of the frequency with respect to a parameter.

        Args:
            U_func (callable): Potential function.
            phi_min (float): Phase at minimum.
            *args: Additional arguments for U_func.
            param_index (int): Index of the parameter to differentiate with respect to.

        Returns:
            tuple: (domega/dparam, dphi_min/dparam)
        """
        d2U_dphi2 = grad(grad(U_func, argnums=0), argnums=0)(phi_min, *args)
        d2U_dphi_dparam = grad(grad(U_func, argnums=0), argnums=param_index+1)(phi_min, *args)
        d3U_dphi3 = grad(grad(grad(U_func, argnums=0), argnums=0), argnums=0)(phi_min, *args)
        d3U_dphi2_dparam = grad(grad(grad(U_func, argnums=0), argnums=0), argnums=param_index+1)(phi_min, *args)
        dphi_min_dparam = -d2U_dphi_dparam / d2U_dphi2
        omega = np.sqrt(self.Ec * d2U_dphi2)
        partial_omega_param = (self.Ec / omega) * d3U_dphi2_dparam
        partial_omega_phi = (self.Ec / omega) * d3U_dphi3
        domega_dparam_val = partial_omega_param + partial_omega_phi * dphi_min_dparam
        return domega_dparam_val, dphi_min_dparam

    def T2_basis_left_right(self, phi_min: float, phi_L: float, phi_R: float) -> float:
        """Calculate T2 dephasing time in the left/right basis.
        Notes : 
            White Noise : https://www.notion.so/Theory-of-white-noise-and-extension-to-2-loops-281c2ec4c368806ebbdaefd889f548e4?source=copy_link
            1/f Noise : https://www.notion.so/Theory-of-1-f-noise-and-extension-to-two-flux-loops-281c2ec4c368806896efd1f3c3eaf744?source=copy_link

        Args:
            phi_min (float): Phase at minimum.
            phi_L (float): Left phase.
            phi_R (float): Right phase.

        Returns:
            float: T2 dephasing time.
        """
        domega_left = self.domega_dparam(self.U_left_right_basis, phi_min, phi_L, phi_R, param_index=0)[0]
        domega_right = self.domega_dparam(self.U_left_right_basis, phi_min, phi_L, phi_R, param_index=1)[0]
        rate_white_noise = self.A_L*domega_left**2 + self.A_R*domega_right**2 + 2*self.A_cross_lr*domega_left*domega_right 
        rate_1overf_noise = np.sqrt(self.B_L*domega_left**2 + self.B_R*domega_right**2 + 2*self.B_cross_lr*domega_left*domega_right)
        rate_total = rate_white_noise + rate_1overf_noise

        Tphi = 1 / rate_total
        T2 = 1 / (1/(2*self.T1) + 1/Tphi)
        return T2

    def T2_basis_plus_minus(self, phi_min: float, phi_plus: float, phi_minus: float) -> float:
        """Calculate T2 dephasing time in the plus/minus basis.

        Args:
            phi_min (float): Phase at minimum.
            phi_plus (float): Plus phase.
            phi_minus (float): Minus phase.

        Returns:
            float: T2 dephasing time.
        """
        domega_plus = self.domega_dparam(self.U_plus_minus_basis, phi_min, phi_plus, phi_minus, param_index=0)[0]
        domega_minus = self.domega_dparam(self.U_plus_minus_basis, phi_min, phi_plus, phi_minus, param_index=1)[0]
        rate_white_noise = self.A_plus*domega_plus**2 + self.A_minus*domega_minus**2 + 2*self.A_cross_pm*domega_plus*domega_minus 
        rate_1overf_noise = np.sqrt(self.B_plus*domega_plus**2 + self.B_minus*domega_minus**2 + 2*self.B_cross_pm*domega_plus*domega_minus)
        rate_total = rate_white_noise + rate_1overf_noise
        # Tphi = np.where(rate_total > 1e-12, 1 / rate_total, 2 * self.T1)
        Tphi = 1 / rate_total
        # Tphi = np.where(rate_total > 1e-12, 1 / rate_total, 2 * self.T1)
        T2 = 1 / (1/(2*self.T1) + 1/Tphi)
        return T2

    def _vectorize_quantity(self, func: Callable, in_axes: Tuple[int, ...]) -> Callable:
        """Helper to vectorize a function over 2D grids using jax.vmap twice.

        Args:
            func (callable): Function to vectorize.
            in_axes (tuple): Axes specification for vmap.

        Returns:
            callable: Vectorized function.
        """
        return jax.vmap(jax.vmap(func, in_axes=in_axes), in_axes=in_axes)

    def _vectorize_quantity_with_order(self, func: Callable, in_axes: Tuple[int, ...]) -> Callable[[int], Callable]:
        """Helper to vectorize a function with an order argument over 2D grids.

        Args:
            func (callable): Function to vectorize, with order as first argument.
            in_axes (tuple): Axes specification for vmap.

        Returns:
            callable: Function that takes order and returns the vectorized function.
        """
        def vfunc(order):
            return self._vectorize_quantity(
                lambda phi_min, phi_plus, phi_minus: func(order, phi_min, phi_plus, phi_minus),
                in_axes
            )
        return vfunc

    def grid_calculate(self, phi_vals_L: np.ndarray, phi_vals_R: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate all relevant quantities on a grid of phi_L and phi_R.

        Args:
            phi_vals_L (ndarray): Array of left phase values.
            phi_vals_R (ndarray): Array of right phase values.

        Returns:
            tuple: Grids of phi_L, phi_R, omega, kerr, goddsum, gevensum, g2ac_plus, g2ac_minus, g3ac_plus, g3ac_minus, T2_left_right, T2_plus_minus.
        """
        phi_L_grid, phi_R_grid = np.meshgrid(phi_vals_L, phi_vals_R, indexing='ij')
        phi_plus_grid, phi_minus_grid = self.get_phi_plus_minus(phi_L_grid, phi_R_grid)

        # Vectorized newton minimizer for phi_min grid-wise
        v_find_min = self._vectorize_quantity(
            lambda p_plus, p_minus: newton_minimize(self.U_plus_minus_basis, p_plus, p_minus),
            in_axes=(0, 0)
        )
        phi_min_grid = v_find_min(phi_plus_grid, phi_minus_grid)

        # Vectorized quantities
        v_omega = self._vectorize_quantity(
            lambda phi_min, phi_plus, phi_minus: self.omega_renormalised(phi_min, phi_plus, phi_minus),
            in_axes=(0, 0, 0)
        )
        v_kerr = self._vectorize_quantity(
            lambda phi_min, phi_plus, phi_minus: self.kerr(phi_min, phi_plus, phi_minus),
            in_axes=(0, 0, 0)
        )
        v_gi = self._vectorize_quantity_with_order(self.gi, in_axes=(0, 0, 0))
        v_giac_plus = self._vectorize_quantity_with_order(self.gi_ac_plus, in_axes=(0, 0, 0))
        v_giac_minus = self._vectorize_quantity_with_order(self.gi_ac_minus, in_axes=(0, 0, 0))
        v_T2_left_right = self._vectorize_quantity(
            lambda phi_min, phi_L, phi_R: self.T2_basis_left_right(phi_min, phi_L, phi_R),
            in_axes=(0, 0, 0)
        )
        v_T2_plus_minus = self._vectorize_quantity(
            lambda phi_min, phi_plus, phi_minus: self.T2_basis_plus_minus(phi_min, phi_plus, phi_minus),
            in_axes=(0, 0, 0)
        )

        omega = v_omega(phi_min_grid, phi_plus_grid, phi_minus_grid)
        kerr = v_kerr(phi_min_grid, phi_plus_grid, phi_minus_grid)
        goddsum = v_gi(5)(phi_min_grid, phi_plus_grid, phi_minus_grid) \
                  + v_gi(7)(phi_min_grid, phi_plus_grid, phi_minus_grid) \
                  + v_gi(9)(phi_min_grid, phi_plus_grid, phi_minus_grid)
        gevensum = v_gi(4)(phi_min_grid, phi_plus_grid, phi_minus_grid) \
                   + v_gi(6)(phi_min_grid, phi_plus_grid, phi_minus_grid) \
                   + v_gi(8)(phi_min_grid, phi_plus_grid, phi_minus_grid)
        g2ac_plus = v_giac_plus(2)(phi_min_grid, phi_plus_grid, phi_minus_grid)
        g2ac_minus = v_giac_minus(2)(phi_min_grid, phi_plus_grid, phi_minus_grid)
        g3ac_plus = v_giac_plus(3)(phi_min_grid, phi_plus_grid, phi_minus_grid)
        g3ac_minus = v_giac_minus(3)(phi_min_grid, phi_plus_grid, phi_minus_grid)

        T2_left_right = v_T2_left_right(phi_min_grid, phi_L_grid, phi_R_grid)
        T2_plus_minus = v_T2_plus_minus(phi_min_grid, phi_plus_grid, phi_minus_grid)

        return phi_L_grid, phi_R_grid, omega, kerr, goddsum, gevensum, g2ac_plus, g2ac_minus, g3ac_plus, g3ac_minus, T2_left_right, T2_plus_minus

    def grid_calculate_nonlinearity(
        self,
        phi_vals_L: np.ndarray,
        phi_vals_R: np.ndarray,
        omega_a: float,
        omega_b: float,
        omega_c: Optional[float],
        g_a: float,
        g_b: float,
        Delta_a: Optional[float] = None,
        Delta_b: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate g3, g4, and cross-Kerr on a grid of phi_L and phi_R."""
        phi_L_grid, phi_R_grid = np.meshgrid(phi_vals_L, phi_vals_R, indexing='ij')
        phi_plus_grid, phi_minus_grid = self.get_phi_plus_minus(phi_L_grid, phi_R_grid)

        v_find_min = self._vectorize_quantity(
            lambda p_plus, p_minus: newton_minimize(self.U_plus_minus_basis, p_plus, p_minus),
            in_axes=(0, 0)
        )
        phi_min_grid = v_find_min(phi_plus_grid, phi_minus_grid)

        v_gi = self._vectorize_quantity_with_order(self.gi, in_axes=(0, 0, 0))
        g3_grid = v_gi(3)(phi_min_grid, phi_plus_grid, phi_minus_grid)
        g4_grid = v_gi(4)(phi_min_grid, phi_plus_grid, phi_minus_grid)
        g_6_grid = v_gi(6)(phi_min_grid, phi_plus_grid, phi_minus_grid)

        v_omega = self._vectorize_quantity(
            lambda phi_min, phi_plus, phi_minus: self.omega_renormalised(phi_min, phi_plus, phi_minus),
            in_axes=(0, 0, 0)
        )
        omega_c_grid = v_omega(phi_min_grid, phi_plus_grid, phi_minus_grid)

        Delta_a_grid = omega_c_grid - omega_a
        Delta_b_grid = omega_c_grid - omega_b

        w_tilde_grid = 1.0 / (
            1.0 / (omega_a - omega_b - omega_c_grid)
            + 1.0 / (-omega_a + omega_b - omega_c_grid)
            + 1.0 / (omega_a + omega_b - omega_c_grid)
            + 1.0 / (-omega_a - omega_b - omega_c_grid)
        )

        prefactor = 24.0 * g4_grid + 36.0 * (g3_grid ** 2) / w_tilde_grid
        crosskerr_grid = prefactor * (g_a / Delta_a_grid) ** 2 * (g_b / Delta_b_grid) ** 2

        return phi_L_grid, phi_R_grid, g3_grid, g4_grid,g_6_grid, crosskerr_grid

    def crosskerr_linecuts(
        self,
        phi_minus_vals: np.ndarray,
        phi_plus_vals: np.ndarray,
        omega_a: float,
        omega_b: float,
        g_a: float,
        g_b: float,
        plot: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute cross-Kerr along phi_+=0 (vs phi_-) and phi_-=0 (vs phi_+).

        Args:
            phi_minus_vals (ndarray): Sweep values for phi_- when phi_+ = 0 (phi_L = -phi_R).
            phi_plus_vals (ndarray): Sweep values for phi_+ when phi_- = 0 (phi_L = phi_R).
            omega_a (float): Frequency of mode a (rad/s).
            omega_b (float): Frequency of mode b (rad/s).
            g_a (float): Coupling of mode a to the ATS.
            g_b (float): Coupling of mode b to the ATS.
            plot (bool): If True, display a two-panel plot of the linecuts.

        Returns:
            tuple: (phi_minus_vals, chi_vs_phi_minus, phi_plus_vals, chi_vs_phi_plus)
                chi values are returned in rad/s.
        """
        phi_minus_vals = np.asarray(phi_minus_vals)
        phi_plus_vals = np.asarray(phi_plus_vals)

        v_crosskerr = jax.vmap(
            lambda phi_L, phi_R: self.crosskerr(phi_L, phi_R, omega_a, omega_b, g_a, g_b),
            in_axes=(0, 0)
        )

        chi_vs_phi_minus = v_crosskerr(-phi_minus_vals, phi_minus_vals)  # phi_+ = 0 -> phi_L = -phi_R
        chi_vs_phi_plus = v_crosskerr(phi_plus_vals, phi_plus_vals)      # phi_- = 0 -> phi_L = phi_R

        if plot:
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            ax[0].plot(phi_plus_vals / np.pi, chi_vs_phi_plus / (2 * np.pi), color='C1')

            ax[0].set_xlabel(r'$\phi_+ / \pi$')
            ax[0].set_ylabel(r'$\chi_{ab}$ (Hz)')
            ax[0].set_title(r'$\phi_+ sweep$')
            ax[0].axhline(0, color='k', linestyle='--', linewidth=0.8)
            ax[1].plot(phi_minus_vals / np.pi, chi_vs_phi_minus / (2 * np.pi))
            ax[1].set_xlabel(r'$\phi_- / \pi$')
            ax[1].set_ylabel(r'$\chi_{ab}$ (Hz)')
            ax[1].set_title(r'$\phi_- sweep$')
            ax[1].axhline(0, color='k', linestyle='--', linewidth=0.8)
            

            fig.tight_layout()
            plt.show()

        return phi_minus_vals, chi_vs_phi_minus, phi_plus_vals, chi_vs_phi_plus

    def crosskerr_linecuts_as_lists(
        self,
        phi_minus_vals: np.ndarray,
        phi_plus_vals: np.ndarray,
        omega_a: float,
        omega_b: float,
        g_a: float,
        g_b: float,
        in_hz: bool = True,
    ) -> dict:
        """Convenience wrapper to return linecut data as plain Python lists.

        Args:
            phi_minus_vals (ndarray): Sweep values for phi_- when phi_+ = 0.
            phi_plus_vals (ndarray): Sweep values for phi_+ when phi_- = 0.
            omega_a (float): Frequency of mode a (rad/s).
            omega_b (float): Frequency of mode b (rad/s).
            g_a (float): Coupling of mode a to the ATS.
            g_b (float): Coupling of mode b to the ATS.
            in_hz (bool): If True, chi is returned in Hz; otherwise rad/s.

        Returns:
            dict: {'phi_minus': [...], 'chi_phi_minus': [...], 'phi_plus': [...], 'chi_phi_plus': [...]}
        """
        phi_m, chi_m, phi_p, chi_p = self.crosskerr_linecuts(
            phi_minus_vals=phi_minus_vals,
            phi_plus_vals=phi_plus_vals,
            omega_a=omega_a,
            omega_b=omega_b,
            g_a=g_a,
            g_b=g_b,
            plot=False,
        )
        if in_hz:
            chi_m = chi_m / (2 * np.pi)
            chi_p = chi_p / (2 * np.pi)
        return {
            "phi_minus": onp.asarray(phi_m).tolist(),
            "chi_phi_minus": onp.asarray(chi_m).tolist(),
            "phi_plus": onp.asarray(phi_p).tolist(),
            "chi_phi_plus": onp.asarray(chi_p).tolist(),
        }

    def crosskerr_linecuts_to_json(
        self,
        phi_minus_vals: np.ndarray,
        phi_plus_vals: np.ndarray,
        omega_a: float,
        omega_b: float,
        g_a: float,
        g_b: float,
        path: Union[str, Path] = "crosskerr_linecuts.json",
        in_hz: bool = True,
    ) -> dict:
        """Compute linecuts and save them as JSON for use elsewhere.

        Args:
            phi_minus_vals (ndarray): Sweep values for phi_- when phi_+ = 0.
            phi_plus_vals (ndarray): Sweep values for phi_+ when phi_- = 0.
            omega_a (float): Frequency of mode a (rad/s).
            omega_b (float): Frequency of mode b (rad/s).
            g_a (float): Coupling of mode a to the ATS.
            g_b (float): Coupling of mode b to the ATS.
            path (str | Path): Output JSON path.
            in_hz (bool): If True, chi is stored in Hz; otherwise rad/s.

        Returns:
            dict: Same data that was saved to disk.
        """
        data = self.crosskerr_linecuts_as_lists(
            phi_minus_vals=phi_minus_vals,
            phi_plus_vals=phi_plus_vals,
            omega_a=omega_a,
            omega_b=omega_b,
            g_a=g_a,
            g_b=g_b,
            in_hz=in_hz,
        )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return data

    def grid_calculate_freq(self, phi_vals_L: np.ndarray, phi_vals_R: np.ndarray, grid: bool = True) -> Tuple[float, float, np.ndarray]:
        """Calculate the maximum and minimum frequencies on a grid.

        Args:
            phi_vals_L (ndarray): Array of left phase values.
            phi_vals_R (ndarray): Array of right phase values.
            grid (bool, optional): Whether to use meshgrid. Defaults to True.

        Returns:
            tuple: (max_omega, min_omega, omega_grid)
        """
        if grid:
            phi_L_grid, phi_R_grid = np.meshgrid(phi_vals_L, phi_vals_R, indexing='ij')
        else:
            phi_L_grid = phi_vals_L.reshape(-1, 1) if phi_vals_L.ndim == 1 else phi_vals_L
            phi_R_grid = phi_vals_R.reshape(-1, 1) if phi_vals_R.ndim == 1 else phi_vals_R

        phi_plus_grid, phi_minus_grid = self.get_phi_plus_minus(phi_L_grid, phi_R_grid)

        v_find_min = self._vectorize_quantity(
            lambda p_plus, p_minus: newton_minimize(self.U_plus_minus_basis, p_plus, p_minus),
            in_axes=(0, 0)
        )
        phi_min_grid = v_find_min(phi_plus_grid, phi_minus_grid)

        v_omega = self._vectorize_quantity(
            lambda phi_min, phi_plus, phi_minus: self.omega_renormalised(phi_min, phi_plus, phi_minus),
            in_axes=(0, 0, 0)
        )
        omega = v_omega(phi_min_grid, phi_plus_grid, phi_minus_grid)
        max_omega = np.nanmax(omega)
        min_omega = np.nanmin(omega)
        return max_omega, min_omega, omega

    def plot_results(
        self,
        phi_L_grid: np.ndarray,
        phi_R_grid: np.ndarray,
        omega: np.ndarray,
        kerr: np.ndarray,
        goddsum: np.ndarray,
        gevensum: np.ndarray,
        g2ac_plus: np.ndarray,
        g2ac_minus: np.ndarray,
        g3ac_plus: np.ndarray,
        g3ac_minus: np.ndarray,
        T2_left_right: np.ndarray,
        T2_plus_minus: np.ndarray,
        phi_vals_L: np.ndarray,
        phi_vals_R: np.ndarray,
        show_bias_point: bool = False
    ) -> None:
        """Plot the results of the grid calculation.

        Args:
            phi_L_grid (ndarray): Grid of left phase values.
            phi_R_grid (ndarray): Grid of right phase values.
            omega (ndarray): Frequency grid.
            kerr (ndarray): Kerr nonlinearity grid.
            goddsum (ndarray): Odd nonlinearity sum grid.
            gevensum (ndarray): Even nonlinearity sum grid.
            g2ac_plus (ndarray): 2nd order AC plus grid.
            g2ac_minus (ndarray): 2nd order AC minus grid.
            g3ac_plus (ndarray): 3rd order AC plus grid.
            g3ac_minus (ndarray): 3rd order AC minus grid.
            T2_left_right (ndarray): T2 left-right grid.
            T2_plus_minus (ndarray): T2 plus-minus grid.
            phi_vals_L (ndarray): Array of left phase values.
            phi_vals_R (ndarray): Array of right phase values.
            show_bias_point (bool): If True, mark the sweet spots with a star.
        """
        # -- Get sweet spots if requested --
        star_phiL, star_phiR = [], []

        if show_bias_point:
            try:
                sweet_spots = self.get_sweet_spot(
                    [float(phi_vals_L[0]), float(phi_vals_L[-1])],
                    [float(phi_vals_R[0]), float(phi_vals_R[-1])]
                )
            except Exception as e:
                print("Failed to call get_sweet_spot for plotting bias points. Exception:", e)
                sweet_spots = []

            # Try to handle sweet_spots in various possible forms (DataFrame, dict, ndarray/tuple/list)
            import pandas as pd
            try:
                if isinstance(sweet_spots, pd.DataFrame):
                    # Each column is a sweet spot, with phi_L and phi_R stored as row values
                    if {"phi_L", "phi_R"}.issubset(set(sweet_spots.index)):
                        phi_Ls = sweet_spots.loc["phi_L", :]
                        phi_Rs = sweet_spots.loc["phi_R", :]
                    elif {"phi_L", "phi_R"}.issubset(set(sweet_spots.columns)):
                        phi_Ls = sweet_spots["phi_L"]
                        phi_Rs = sweet_spots["phi_R"]
                    else:
                        phi_Ls = []
                        phi_Rs = []
                    star_phiL = [float(l)/np.pi for l in phi_Ls]
                    star_phiR = [float(r)/np.pi for r in phi_Rs]
                elif hasattr(sweet_spots, "__len__") and len(sweet_spots) > 0:
                    first = sweet_spots[0]
                    # If dict-like
                    if isinstance(first, dict):
                        star_phiL = [float(spot['phi_L'])/np.pi for spot in sweet_spots if 'phi_L' in spot and 'phi_R' in spot]
                        star_phiR = [float(spot['phi_R'])/np.pi for spot in sweet_spots if 'phi_L' in spot and 'phi_R' in spot]
                    # If DataFrame row
                    elif hasattr(first, "__getitem__") and len(first) == 2:
                        arr = []
                        try:
                            arr = np.array(sweet_spots)
                        except Exception:
                            arr = np.array([list(spot) for spot in sweet_spots])
                        if arr.ndim == 2 and arr.shape[1] == 2:
                            star_phiL = [float(x)/np.pi for x in arr[:, 0]]
                            star_phiR = [float(x)/np.pi for x in arr[:, 1]]
                        elif arr.ndim == 1 and arr.shape[0] == 2:
                            star_phiL = [float(arr[0])/np.pi]
                            star_phiR = [float(arr[1])/np.pi]
                elif hasattr(sweet_spots, "shape") and sweet_spots.shape[-1] == 2:
                    arr = np.array(sweet_spots)
                    star_phiL = [float(x)/np.pi for x in arr[:, 0]]
                    star_phiR = [float(x)/np.pi for x in arr[:, 1]]
            except Exception as e:
                print("Failed to process sweet spots for plotting. Exception:", e)
                star_phiL, star_phiR = [], []

        fig, ax = plt.subplots(3, 4, figsize=(15, 10))
        data = [
            (omega / 1e9 / 2 / np.pi, 'omega (GHz)'),
            (kerr / (2 * np.pi * 1e6), 'kerr (MHz)'),
            (goddsum / (2 * np.pi * 1e6), '$g_{odd}$ sum (MHz)'),
            (gevensum / (2 * np.pi * 1e6), '$g_{even}$ sum (MHz)'),
            (g2ac_plus / (2 * np.pi * 1e6), '$g_{2}$AC$^+$ (MHz)'),
            (g2ac_minus / (2 * np.pi * 1e6), '$g_{2}$AC$^-$ (MHz)'),
            (g3ac_plus / (2 * np.pi * 1e6), '$g_{3}$AC$^+$ (MHz)'),
            (g3ac_minus / (2 * np.pi * 1e6), '$g_{3}$AC$^-$ (MHz)'),
            (T2_left_right / 1e-6, r'$T_2$ left-right ($\mu$s)'),
            (T2_plus_minus / 1e-6, r'$T_2$ plus-minus ($\mu$s)'),
        ]
        
        extent = [phi_vals_L[0]/np.pi, phi_vals_L[-1]/np.pi, phi_vals_R[0]/np.pi, phi_vals_R[-1]/np.pi]

        for i in range(3):
            for j in range(4):
                if i == 2 and j > 1:
                    break
                idx = i * 4 + j
                im_data, title = data[idx]

                # Center colorbar at 0 for Kerr and g terms
                if idx in [1, 2, 3, 4, 5, 6, 7]:
                    vmax = np.nanmax(np.abs(im_data))
                    vmin = -vmax
                    cbar = ax[i, j].imshow(
                        im_data,
                        extent=extent,
                        origin='lower', aspect='auto',
                        cmap='seismic', vmin=vmin, vmax=vmax
                    )
                else:
                    cbar = ax[i, j].imshow(
                        im_data,
                        extent=extent,
                        origin='lower', aspect='auto',
                        cmap='seismic'
                    )

                ax[i, j].set_title(title)
                ax[i, j].set_xlabel(r'$\phi_L/\pi$')
                ax[i, j].set_ylabel(r'$\phi_R/\pi$')
                plt.colorbar(cbar, ax=ax[i, j])

                # add contour for Kerr free point
                ax[i, j].contour(phi_L_grid/np.pi, phi_R_grid/np.pi, kerr, levels=[0], colors='green')
                ax[i, j].contour(phi_L_grid/np.pi, phi_R_grid/np.pi, goddsum, levels=[0], colors='black', alpha=0.5)
                ax[i, j].contour(phi_L_grid/np.pi, phi_R_grid/np.pi, gevensum, levels=[0], colors='yellow', alpha=0.5)

                # Mark bias points with a star
                if show_bias_point and len(star_phiL) > 0 and len(star_phiL) == len(star_phiR):
                    ax[i, j].plot(
                        star_phiL, star_phiR,
                        marker='*', color='magenta', linestyle='None', markersize=12,
                        markeredgewidth=1, markeredgecolor='k', zorder=10
                    )

        ax[2, 2].axis('off')
        ax[2, 3].axis('off')

        # add common legend for all plots at bottom right
        legend_handles = [
            Line2D([0], [0], color="green", linestyle="--", label="K = 0"),
            Line2D([0], [0], color="black", linestyle="-", label="g_odd = 0"),
            Line2D([0], [0], color="yellow", linestyle="-", label="g_even = 0"),
        ]
        if show_bias_point:
            legend_handles.append(
                Line2D([0], [0], marker='*', color='magenta', linestyle='None',
                       label='Sweet spot', markersize=12, markeredgewidth=1, markeredgecolor='k')
            )
        fig.legend(handles=legend_handles, loc="lower right", frameon=False, fancybox=True, shadow=True)

        fig.tight_layout()
        plt.show()

    def plot_nonlinearity_maps(
        self,
        phi_vals_L: np.ndarray,
        phi_vals_R: np.ndarray,
        omega_a: float,
        omega_b: float,
        omega_c: Optional[float],
        g_a: float,
        g_b: float,
        Delta_a: Optional[float] = None,
        Delta_b: Optional[float] = None,
    ) -> None:
        """Plot g3, g4, and cross-Kerr on a phi_L/phi_R grid."""
        phi_L_grid, phi_R_grid, g3_grid, g4_grid,g6_grid, crosskerr_grid = self.grid_calculate_nonlinearity(
            phi_vals_L, phi_vals_R, omega_a, omega_b, omega_c, g_a, g_b, Delta_a, Delta_b
        )

        fig, ax = plt.subplots(1, 4, figsize=(15, 4))
        data = [
            (g3_grid / (2 * np.pi * 1e6), 'g3 (MHz)'),
            (g4_grid / (2 * np.pi * 1e6), 'g4 (MHz)'),
            (g6_grid / (2 * np.pi * 1e3), 'g6 (KHz)'),
            (crosskerr_grid / (2 * np.pi), r'$\chi_{ab}$ (Hz)'),  # convert from rad/s to Hz
        ]
        extent = [phi_vals_L[0]/np.pi, phi_vals_L[-1]/np.pi, phi_vals_R[0]/np.pi, phi_vals_R[-1]/np.pi]

        for idx, (im_data, title) in enumerate(data):
            vmax = np.nanmax(np.abs(im_data))
            vmin = -vmax
            cbar = ax[idx].imshow(
                im_data,
                extent=extent,
                origin='lower',
                aspect='auto',
                cmap='seismic',
                vmin=vmin,
                vmax=vmax
            )
            ax[idx].set_title(title)
            ax[idx].set_xlabel(r'$\phi_L/\pi$')
            ax[idx].set_ylabel(r'$\phi_R/\pi$')
            plt.colorbar(cbar, ax=ax[idx])

        fig.tight_layout()
        plt.show()
        
class ATS_linearInductance_all(ATS):
    """ATS model with linear inductances.
    Derived from the standard ATS model by adding effects of linear inductances in the circuit.
    See: https://www.notion.so/Real-ATS-potential-with-linear-inductances-239c2ec4c36880bb8000d5b62d5bdf5e?source=copy_link
    """

    def __init__(self, E_lin_l: float, E_lin_r: float, E_lin_c: float, **args) -> None:
        """Initialize the ATS_linearInductance_all model.

        Args:
            E_lin_l (float): Left inductive energy of linear inductor.
            E_lin_r (float): Right inductive energy of linear inductor.
            E_lin_c (float): Center inductive energy of linear inductor.
            **args: Additional arguments for ATS base class.
        """
        super().__init__(**args)
        self.E_lin_l = E_lin_l
        self.E_lin_r = E_lin_r
        self.E_lin_c = E_lin_c

    def U_plus_minus_basis(self, phi: float, phi_plus: float, phi_minus: float) -> float:
        """Potential energy in the plus/minus basis for the linear inductance model.

        Args:
            phi (float): Phase variable.
            phi_plus (float): Plus phase.
            phi_minus (float): Minus phase.

        Returns:
            float: Potential energy.
        """
        Ap = self.Er**2  / (4 * self.E_lin_r) + self.El**2 / (4 * self.E_lin_l)
        Am = self.Er**2  / (4 * self.E_lin_r) - self.El**2 / (4 * self.E_lin_l)
        basic_ats_term = super().U_plus_minus_basis(phi, phi_plus, phi_minus)
        linear_inductor_term = ((self.Ej * self.N)**2 / (4 * self.E_lin_c)) * np.cos(2 * phi / self.N) + Ap * np.cos(2 * (phi - phi_plus)) * np.cos(2 * phi_minus) + Am * np.sin(2 * (phi - phi_plus)) * np.sin(2 * phi_minus)
        return basic_ats_term + linear_inductor_term

    def U_left_right_basis(self, phi: float, phi_L: float, phi_R: float) -> float:
        """Potential energy in the left/right basis for the linear inductance model.

        Args:
            phi (float): Phase variable.
            phi_L (float): Left phase.
            phi_R (float): Right phase.

        Returns:
            float: Potential energy.
        """
        basic_ats_term = super().U_left_right_basis(phi, phi_L, phi_R)
        linear_inductor_term = ((self.Ej * self.N)**2 / (4 * self.E_lin_c)) * np.cos(2 * phi / self.N) + (self.El**2 / (4 * self.E_lin_l)) * np.cos(2 * (phi - phi_L)) + (self.Er**2 / (4 * self.E_lin_r)) * np.cos(2 * (phi - phi_R))
        return basic_ats_term + linear_inductor_term

### main

if __name__ == "__main__":

    
    def change_lr_to_pm(C_L:float,C_R:float,C_cross_lr:float):
        '''
        Function used to change A and B constants from left/right basis to plus/minus basis. Here C is A or B.
        '''
        C_plus = (C_R + C_L + 2*C_cross_lr) / 4
        C_minus = (C_R + C_L - 2*C_cross_lr) / 4
        C_cross_pm = (C_R - C_L) / 4
        return C_plus, C_minus, C_cross_pm
    
    def run_pure_ats_example(sweet_spot=False,static_nonlinearity=False) -> None:
        
        # Example 1: Pure ATS
        Ej = 409e9 *2.0* np.pi
        El = 8.44e9  *2.0* np.pi
        Er = 8.44e9 *2.0*np.pi
        Ec = get_Ec_from_C(C=540e-15)
        N = 3

        B = 5.01e-5/100 # 1% of value from SNAIL paper relating to 1/f noise
        A =  3.63e-5*1e-9/10 # 10% of value from SNAIL paper relating to white noise

        # Set A and B constants for left/right basis
        AL = A # Noise in left loop 
        AR = A # Noise in right loop
        A_cross_lr = A/2 # Correlated noise between left and right loop
        BL = B # Noise in left loop
        BR = B # Noise in right loop
        B_cross_lr = B/2 # Correlated noise between left and right loop

        AP, AM, A_cross_pm = change_lr_to_pm(AL,AR,A_cross_lr)
        BP, BM, B_cross_pm = change_lr_to_pm(BL,BR,B_cross_lr)


        ats_pure = ATS(
            Ej=Ej, El=El, Er=Er, Ec=Ec, N=N,
            A_L=AL , B_L=BL,
            A_R=AR , B_R=BR,
            A_plus=AP, B_plus=BP,
            A_minus=AM, B_minus=BM,
            A_cross_lr=A_cross_lr, A_cross_pm=A_cross_pm,
            B_cross_lr=B_cross_lr, B_cross_pm=B_cross_pm,
            T1=200e-6
        )

        phi_vals_L = np.linspace(-2.5 * np.pi, 2.5 * np.pi, 100)
        phi_vals_R = np.linspace(-2.5 * np.pi, 2.5 * np.pi, 100)

        phi_L_grid, phi_R_grid, omega, kerr, goddsum, gevensum, g2ac_plus, g2ac_minus, g3ac_plus, g3ac_minus, T2_left_right, T2_plus_minus = ats_pure.grid_calculate(phi_vals_L, phi_vals_R)
        ats_pure.plot_results(phi_L_grid, phi_R_grid, omega, kerr, goddsum, gevensum, g2ac_plus, g2ac_minus, g3ac_plus, g3ac_minus, T2_left_right, T2_plus_minus, phi_vals_L, phi_vals_R, show_bias_point=False)
        if sweet_spot:
            sweet_spot_table = ats_pure.get_sweet_spot(
                phi_L_range=(-2.5 * np.pi, 2.5 * np.pi),
                phi_R_range=(-2.5 * np.pi, 2.5 * np.pi),
                grid_points=30,  # Resolution for initial grid search (default: 30)
                n_initial_points=15,  # Number of starting points (default: 15)
                weight_kerr=1.0, 
                weight_odd=1.0, 
                weight_even=1.0,
                cost_threshold=1.0,  # Adjust this threshold based on your needs
                use_gradient=True  # Use analytical gradients (much faster!)
            )

            if not sweet_spot_table.empty:
                print("\nSweet Spot Analysis:")
                print("="*100)
                print(sweet_spot_table.to_string())
                print("="*100)
                print(f"\nNote: Cost = weight_kerr*|Kerr|Â² + weight_odd*|odd_sum|Â² + weight_even*|even_sum|Â²")
                print(f"      (all normalized to MHz scale)")
            else:
                print("\nNo sweet spots found. Try adjusting cost_threshold or weights.")

        
        # Example: plot g3, g4, and cross-Kerr over the bias grid
        if static_nonlinearity:
            omega_a = 2909e6 * 2.0 * np.pi  # rad/s
            omega_b = 3803e6 * 2.0 * np.pi  # rad/s
            g_a = 108e6 * 2.0 * np.pi       # rad/s
            g_b = 153e6 * 2.0 * np.pi       # rad/s
            # omega_c is the ATS frequency; detunings will be computed as (omega_c - omega_a/b) at each bias point
            ats_pure.plot_nonlinearity_maps(
                phi_vals_L,
                phi_vals_R,
                omega_a=omega_a,
                omega_b=omega_b,
                omega_c=None,
                g_a=g_a,
                g_b=g_b,
                )

            # Linecuts: phi_+ = 0 vs phi_- and phi_- = 0 vs phi_+
            linecut_range = np.linspace(-2.5 * np.pi, 2.5 * np.pi, 400)
            ats_pure.crosskerr_linecuts(
                phi_minus_vals=linecut_range,
                phi_plus_vals=linecut_range,
                omega_a=omega_a,
                omega_b=omega_b,
                g_a=g_a,
                g_b=g_b,
                plot=True,
            )
            # Save the same linecut data to JSON for reuse in other scripts
            ats_pure.crosskerr_linecuts_to_json(
                phi_minus_vals=linecut_range,
                phi_plus_vals=linecut_range,
                omega_a=omega_a,
                omega_b=omega_b,
                g_a=g_a,
                g_b=g_b,
                path="crosskerr_linecuts.json",
                in_hz=True,
            )

    # Example 2: ATS with Linear Inductance
    def run_linear_inductance_example() -> None:
        Ej = 409.6e9 * 2.0 * np.pi
        El = 8.344e9 * 2.0 * np.pi
        Er = 8.344e9 * 2.0 * np.pi
        Ec = get_Ec_from_C(C=540e-15)
        N = 3
        # E_center = 500000e9 * 2.0 * np.pi
        # E_side = 100000e9 * 2.0 * np.pi
        E_lin_l = 100000e9 * 2.0 * np.pi
        E_lin_r = 100000e9 * 2.0 * np.pi
        E_lin_c = 500000e9 * 2.0 * np.pi

        A = 3.63e-5*1e-9
        B = 5.01e-5
        ats_linear = ATS_linearInductance_all(
            Ej=Ej, El=El, Er=Er, Ec=Ec, N=N,
            E_lin_l=E_lin_l,
            E_lin_r=E_lin_r,
            E_lin_c=E_lin_c,
            A_L=A, B_L=B,
            A_R=A, B_R=B,
            A_plus=A, B_plus=B,
            A_minus=A, B_minus=B,
            T1=200e-6
        )
        phi_vals_L = np.linspace(-2.5 * np.pi, 2.5 * np.pi, 100)
        phi_vals_R = np.linspace(-2.5 * np.pi, 2.5 * np.pi, 100)

        phi_L_grid, phi_R_grid, omega, kerr, goddsum, gevensum, g2ac_plus, g2ac_minus, g3ac_plus, g3ac_minus, T2_left_right, T2_plus_minus = ats_linear.grid_calculate(phi_vals_L, phi_vals_R)
        ats_linear.plot_results(phi_L_grid, phi_R_grid, omega, kerr, goddsum, gevensum, g2ac_plus, g2ac_minus, g3ac_plus, g3ac_minus, T2_left_right, T2_plus_minus, phi_vals_L, phi_vals_R)


    # Example 3: Find sweet spots
    def run_sweet_spot_example() -> None:
        Ej = 409.6e9 * 2.0 * np.pi
        El = 8.344e9 * 2.0 * np.pi
        Er = 8.344e9 * 2.0 * np.pi
        Ec = get_Ec_from_C(C=1750e-15)
        N = 3

        B = 5.01e-5/100
        A = 3.63e-5*1e-9/10

        AL = A
        AR = A
        A_cross_lr = A/2
        BL = B
        BR = B
        B_cross_lr = B/2

        AP, AM, A_cross_pm = change_lr_to_pm(AL, AR, A_cross_lr)
        BP, BM, B_cross_pm = change_lr_to_pm(BL, BR, B_cross_lr)

        ats_pure = ATS(
            Ej=Ej, El=El, Er=Er, Ec=Ec, N=N,
            A_L=AL, B_L=BL,
            A_R=AR, B_R=BR,
            A_plus=AP, B_plus=BP,
            A_minus=AM, B_minus=BM,
            A_cross_lr=A_cross_lr, A_cross_pm=A_cross_pm,
            B_cross_lr=B_cross_lr, B_cross_pm=B_cross_pm,
            T1=200e-6
        )

        # Find sweet spots using optimization - just specify the search range!
        # The function uses JIT compilation and analytical gradients for speed
        sweet_spot_table = ats_pure.get_sweet_spot(
            phi_L_range=(-2.5 * np.pi, 2.5 * np.pi),
            phi_R_range=(-2.5 * np.pi, 2.5 * np.pi),
            grid_points=30,  # Resolution for initial grid search (default: 30)
            n_initial_points=15,  # Number of starting points (default: 15)
            weight_kerr=1.0, 
            weight_odd=1.0, 
            weight_even=1.0,
            cost_threshold=1.0,  # Adjust this threshold based on your needs
            use_gradient=True  # Use analytical gradients (much faster!)
        )
        
        if not sweet_spot_table.empty:
            print("\nSweet Spot Analysis:")
            print("="*100)
            print(sweet_spot_table.to_string())
            print("="*100)
            print(f"\nNote: Cost = weight_kerr*|Kerr|Â² + weight_odd*|odd_sum|Â² + weight_even*|even_sum|Â²")
            print(f"      (all normalized to MHz scale)")
        else:
            print("\nNo sweet spots found. Try adjusting cost_threshold or weights.")

    # run_pure_ats_example()
    # run_linear_inductance_example()
    run_sweet_spot_example()
    
    # Execute the basic ATS example with nonlinearity maps
    # run_pure_ats_example(static_nonlinearity=False)
