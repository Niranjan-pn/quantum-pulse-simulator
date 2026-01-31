import json
from dataclasses import dataclass, field
from functools import partial
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as ct
from matplotlib.axes import Axes
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit

import flux_tunable as ft
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class T2ModelData:
    a: Optional[float] = None
    b: Optional[float] = None
    T1: Optional[float | list] = None

    @property
    def sqrt_A_inv_f(self):
        return self.b * 1e3 / (2 * np.pi * np.sqrt(2 * np.log(2)))

    @property
    def sqrt_S_BB(self):
        return np.sqrt(self.a) / (2 * np.pi)


def T2_theory(flux, a, b, T1=None, deriv=None):
    """
    flux : flux [-np.pi np.pi]
    a : (uPhi_0)^2/Hz
    b : uPhi_0
    T1 : (us)
    deriv : (GHz/Phi_0)
    Returns T2 in us
    """
    return 1e-3 / (b * abs(deriv(flux)) + a * deriv(flux) ** 2 + 1 / (2 * T1 * 1e3))
Rq = ct.hbar / (2 * ct.e) ** 2


def cavity_freq_from_v(v, voff, dfluxdv, dl_Lj0, f_infty, α):
    out = ft.cavity_freq(((v - voff) * dfluxdv), dl_Lj0, f_infty, α=α)
    return out


@dataclass
class SNAILData:
    V: list[float]  # V
    f: list[float]  # GHz
    f_ramsey: list[float]  # GHz
    T1: list[float]  # us
    T2: list[float]  # us
    K: list[float]  # KHz

    def get_interpolated_at_V(self, v, parameter: str = "f", **spline_kwargs):
        return CubicSpline(self.V, getattr(self, parameter))(v, **spline_kwargs)


@dataclass
class SNAILParameters:
    off: Optional[float] = None  # Flux offset
    dfluxdv: Optional[float] = None  # 2*pi/ac.FLUX_QUANTA
    dl_Lj0: Optional[float] = None  # Lineic to Josephson inductance ratio (no units)
    f_infty: Optional[float] = None  # GHz
    alpha: Optional[float] = None  # SNAIL asymmetry (no units)
    Z: Optional[float] = None  # In Rq units (e.g 50 Ohm / Rq)
    Ej0: Optional[float] = 245 * 2 * np.pi  # GHz * 2pi. Obtained by room temp resistance.


@dataclass
class SNAILResonator(SNAILParameters):

    data: Optional[SNAILData] = None
    flux: Optional[np.ndarray] = None
    ac_couplings_orders: set[int] = (2, 3)
    dc_couplings_orders: set[int] = (3, 4, 5, 6)
    T2_model: T2ModelData = field(default_factory=T2ModelData)

    def V_to_flux(self, V):
        return (V - self.off) * self.dfluxdv

    def update_flux(self) -> None:
        if self.off and self.dfluxdv:
            self.flux = self.V_to_flux(self.data.V)

    @property
    def reduced_flux(self) -> np.ndarray:
        """Flux in phi0 units."""
        return self.flux / 2 / np.pi

    def update_f(self):
        self.f_fit_data = ft.cavity_freq(self.flux, self.dl_Lj0, self.f_infty, self.alpha)
        self.fcav_cs = CubicSpline(self.flux, self.f_fit_data)
        self.fcav_dcs = self.fcav_cs.derivative()
        self.fcav_d2cs = self.fcav_cs.derivative(nu=2)

        self.dl_Lj0 = self.Ej0 /(4 * self.f_infty / self.Z)
      # if self.Ej0 is not None:
      #     self.Z = 4 * self.f_infty / self.Ej0 * self.dl_Lj0
      # elif self.Z is not None:
      #     self.Ej0 = 4 * self.f_infty / self.Z * self.dl_Lj0
      # else:
      #     raise ValueError("Either the Josephson energy Ej0 or the linear impedance Z should be provided.")

        g3, g4, w0 = ft.couplings(
            self.flux, self.Ej0, self.dl_Lj0, 2 * np.pi * self.f_infty, self.Z, α=self.alpha, order=[3, 4]
        )
        self.Kerr = (g4 - 5 * g3**2 / (w0)) * 1e3 * 6/2/np.pi # In MHz

        ac_couplings = ft.ACcouplings(
            self.flux,
            self.Ej0,
            self.dl_Lj0,
            2 * np.pi * self.f_infty,
            self.Z,
            α=self.alpha,
            order=list(self.ac_couplings_orders),
        )
        self.ac_couplings = np.array(ac_couplings[:-1]) * 1e3/2/np.pi  # In MHz

        dc_couplings = ft.couplings(
            self.flux,
            self.Ej0,
            self.dl_Lj0,
            2 * np.pi * self.f_infty,
            self.Z,
            α=self.alpha,
            order=list(self.dc_couplings_orders),
        )
        self.dc_couplings = np.array(dc_couplings[:-1]) * 1e3 /2/np.pi # In MHz

    def update_T2(self):
        self.T2_fit_data = T2_theory(self.flux, self.T2_model.a, self.T2_model.b, self.T2_model.T1, self.fcav_dcs)

    def fit_f(self, p0: Optional[list] = None, fit_x: Literal["V", "flux"] = "V"):
        if fit_x == "V":
            (self.off, self.dfluxdv, dl_Lj0, f_infty, alpha), _ = curve_fit(
                cavity_freq_from_v, self.data.V, self.data.f, p0=p0, maxfev=300000
            )
            self.update_flux()
        else:
            (dl_Lj0, f_infty, alpha), _ = curve_fit(ft.cavity_freq, self.data.V, self.data.f, p0=p0, maxfev=300000)

        self.dl_Lj0, self.f_infty, self.alpha = dl_Lj0, f_infty, alpha
        self.update_f()

    def fit_T2(self, p0: Optional[list] = None, fit_T1: Optional[bool] = False):
        if fit_T1:
            bounds = [(0, 0, 0), (np.inf, np.inf, np.inf)]
            (a, b, T1_fit), _ = curve_fit(
                partial(T2_theory, deriv=self.fcav_dcs),
                self.flux,
                self.data.T2 * 1e3,
                p0=p0,
                maxfev=300000,
                bounds=bounds,
            )
            self.T2_model.T1 = T1_fit
        else:
            bounds = [(0, 0), (np.inf, np.inf)]
            (a, b), _ = curve_fit(
                partial(T2_theory, T1=np.mean(self.data.T1), deriv=self.fcav_dcs),
                self.flux,
                self.data.T2 * 1e3,
                p0=p0,
                maxfev=300000,
                bounds=bounds,
            )
            T1_fit = np.mean(self.data.T1)
            self.T2_model.T1 = T1_fit
        self.T2_model.a = a
        self.T2_model.b = b
        self.update_T2()

    def plot_f(self, plot_fit: bool = True, plot_data: bool = True, ax: Optional[Axes] = None):
        if ax is None:
            _, ax = plt.subplots()
        if plot_data:
            ax.plot(self.reduced_flux, self.data.f, ".", label="measured")
        if plot_fit:
            ax.plot(self.reduced_flux, self.f_fit_data, label="fit")
        ax.set_xlabel(r"$\Phi_{\mathrm{e}}/\phi_{\mathrm{q}}$")
        ax.set_ylabel(r"$f_{\mathrm{cavity}}$ (GHz)")
        ax.legend()
        return ax

    def plot_T2(self, plot_fit: bool = True, plot_data: bool = True, ax: Optional[Axes] = None):
        if ax is None:
            _, ax = plt.subplots()
        ax.set_xlabel(r"$\Phi_{\mathrm{e}}/\phi_{\mathrm{q}}$")
        ax.set_ylabel(r"$T_2$ ($\mu$s)")
        if plot_data:
            ax.scatter(self.reduced_flux, self.data.T2, label="measured")
        if plot_fit:
            ax.plot(self.reduced_flux, self.T2_fit_data, label="fit")
        ax.legend()
        return ax

    def plot_couplings(self, ax: Optional[Axes] = None):
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(self.reduced_flux, self.Kerr, label="Kerr")
        for i, ac in enumerate(self.ac_couplings):
            ax.plot(self.reduced_flux, ac, label=f"$g_{self.ac_couplings_orders[i]}$ AC")
        for i, dc in enumerate(self.dc_couplings):
            ax.plot(self.reduced_flux, dc, label=f"$g_{self.dc_couplings_orders[i]}$ DC")
            ax.axhline(0, ls="--", color="black")
        ax.set_xlabel(r"$\Phi_{\mathrm{e}}/\phi_{\mathrm{q}}$")
        ax.set_ylabel("couplings (KHz)")
        ax.legend()
        return ax

    def print_SNAIL_parameters(self):
        print(f"flux = (v-{self.off:.3g})*{self.dfluxdv:.3g}")
        print(f"f_∞ = {self.f_infty:.5g} GHz")
        print(f"α = {self.alpha:.5g}")
        print(f"resonator impedance : {self.Z*Rq:.5g} Ω")
        print(f"dl_Lj0 = {self.dl_Lj0:.5}")

    def print_T2_parameters(self):
        sqrt_A_inv_f = self.T2_model.sqrt_A_inv_f
        sqrt_S_BB = self.T2_model.sqrt_S_BB
        print(f"sqrt(A_1/f) = {sqrt_A_inv_f:.3g} $\\mu \\phi_0$")
        print(f"sqrt(S_BB)={sqrt_S_BB:.3g} $\\mu\\phi_0/\\sqrt(Hz)$")
        print(f"T1: {self.T2_model.T1:.3f} $\\mu$s")





def get_snail_couplings_at_zero_kerr(Ej0, dl_Lj0, f_infty, Z, alpha, n=3):
    """
    Calculates SNAIL couplings at the flux point where the Kerr nonlinearity vanishes (g4 - 5*g3^2/w0 = 0).
    
    Args:
        Ej0 (float): Josephson energy in angular frequency units (e.g. 245 * 2 * np.pi [GHz] or equivalent Hz).
        dl_Lj0 (float): Inductive participation parameter.
        f_infty (float): Frequency at infinity (Linear Frequency, GHz or Hz).
        Z (float): Impedance in units of Rq (dimensionless).
        alpha (float): SNAIL asymmetry parameter.
        n (int): Number of junctions in the SNAIL array (default 3).

    Returns: 
        tuple: (g3dc, g4dc, g5dc, g6dc, g3ac, w0) in Hz (Linear Frequency).
    """
    # Robust Unit Handling
    # 1. Normalize Ej0 to Angular Hz
    # Heuristic: If Ej0 < 1e6, assume it's Angular GHz, otherwise Angular Hz
    if abs(Ej0) < 1e6:
        Ej0_ang_hz = Ej0 * 1e9
    else:
        Ej0_ang_hz = Ej0

    # 2. Normalize f_infty to Angular Hz
    # Heuristic: If f_infty < 1e6, assume it's Linear GHz, otherwise Linear Hz
    if abs(f_infty) < 1e6:
        f_infty_lin_hz = f_infty * 1e9
    else:
        f_infty_lin_hz = f_infty
    
    w_infty_ang_hz = f_infty_lin_hz * 2 * np.pi

    # Find initial guess for zero Kerr flux (ignoring higher order corrections)
    try:
        flux_guess = ft.find_kerr_free(alpha, n=n)
    except Exception:
        flux_guess = 0.4 * 2 * np.pi 

    # Define function to minimize: Kerr proportional to g4 - 5*g3^2/w0
    # Calculations performed in Angular Hz
    def kerr_func(flux):
        res = ft.couplings(
            flux, Ej0_ang_hz, dl_Lj0, w_infty_ang_hz, Z, alpha, n=n, order=[3, 4]
        )
        g3, g4, w0 = res

        return g4 - 5 * g3**2 / w0

    # Solve for flux where Kerr = 0
    from scipy import optimize
    
    try:
        flux_zero_kerr = optimize.newton(kerr_func, flux_guess, maxiter=1000, tol=1e-10)
    except RuntimeError:
         raise RuntimeError("Could not find zero Kerr flux point.")

    # Calculate DC couplings at this flux (Angular Hz)
    dc_res = ft.couplings(
        flux_zero_kerr, Ej0_ang_hz, dl_Lj0, w_infty_ang_hz, Z, alpha, n=n, order=[3, 4, 5, 6]
    )
    g3dc, g4dc, g5dc, g6dc, w0 = dc_res[:5]

    # Calculate AC coupling g3 (Angular Hz)
    ac_res = ft.ACcouplings(
        flux_zero_kerr, Ej0_ang_hz, dl_Lj0, w_infty_ang_hz, Z, alpha, n=n, order=3
    )
    g3ac = ac_res[0]

    # Convert all outputs from Angular Hz to Linear Hz
    # Linear Hz = Angular Hz / (2 * pi)
    scale = 1.0 / (2 * np.pi)
    
    return (
        g3dc, 
        g4dc, 
        g5dc, 
        g6dc, 
        g3ac, 
        w0 
    )


from matplotlib.widgets import Slider, CheckButtons
from dataclasses import dataclass
from typing import Optional, Set
import matplotlib.gridspec as gridspec
plt.rcParams["font.size"]=18
# Assuming ft and other dependencies are imported
# and SNAILParameters/SNAILResonator classes are defined as shown

class InteractiveSNAIL:
    def __init__(self):
        # Create initial parameter values
        self.snail = SNAILResonator(
        off=-0.047,
        dfluxdv=2 * np.pi / 11.6,
        flux=np.linspace(0.3*np.pi*2, 0.62*2*np.pi, 200),
        dl_Lj0=3.007,
        f_infty=8.99,
        alpha=0.0971,
        T2_model=T2ModelData(a=4.18e-3, b=1.012e-07, T1=24),
        Z = 57.9 / Rq
    )
        self.snail.update_f()
        self.snail.update_T2()
        self.snail.print_SNAIL_parameters()
        self.snail.print_T2_parameters()

        # Create the main figure
        self.fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(3, 2, height_ratios=[4, 4, 1])

        # Create subplots
        self.ax_freq = plt.subplot(gs[0, 0])
        self.ax_kerr = plt.subplot(gs[0, 1])
        self.ax_ac = plt.subplot(gs[1, 0])
        self.ax_dc = plt.subplot(gs[1, 1])

        # Create slider axes
        slider_color = 'lightgoldenrodyellow'

        # Create parameter sliders
       # self.slider_dl = Slider(plt.axes([0.1, 0.05, 0.3, 0.03]), 'dl_Lj0', 0.5, 6,
       #                       valinit=self.snail.dl_Lj0, color=slider_color)
        self.slider_f = Slider(plt.axes([0.1, 0.1, 0.3, 0.03]), 'f_infty (GHz)', 5, 11.0,
                             valinit=self.snail.f_infty, color=slider_color)
        self.slider_alpha = Slider(plt.axes([0.5, 0.05, 0.3, 0.03]), 'alpha', 0.0, 0.5,
                                 valinit=self.snail.alpha, color=slider_color)
        self.slider_Ej0 = Slider(plt.axes([0.5, 0.1, 0.3, 0.03]), 'Ej0', 100, 400,
                             valinit=self.snail.Ej0/2/np.pi, color=slider_color)
        self.slider_Z = Slider(plt.axes([0.1, 0.05, 0.3, 0.03]), 'Z', 30, 80,
                             valinit=self.snail.Z*Rq, color=slider_color)

        # Create coupling order checkboxes
        self.ac_check = CheckButtons(
            plt.axes([0.85, 0.05, 0.1, 0.1]),
            [f'AC{i}' for i in range(2, 5)],
            [i in self.snail.ac_couplings_orders for i in range(2, 5)]
        )
        self.dc_check = CheckButtons(
            plt.axes([0.95, 0.05, 0.1, 0.1]),
            [f'DC{i}' for i in range(3, 7)],
            [i in self.snail.dc_couplings_orders for i in range(3, 7)]
        )

        # Connect callbacks
        #self.slider_dl.on_changed(self.update)
        self.slider_Z.on_changed(self.update)
        self.slider_f.on_changed(self.update)
        self.slider_alpha.on_changed(self.update)
        self.slider_Ej0.on_changed(self.update)
        self.ac_check.on_clicked(self.update_couplings)
        self.dc_check.on_clicked(self.update_couplings)

        # Initial plot
        self.update(None)

    def update_couplings(self, label):
        # Update coupling orders based on checkbox states
        ac_orders = {i for i in range(2, 5) if self.ac_check.get_status()[i-2]}
        dc_orders = {i for i in range(3, 7) if self.dc_check.get_status()[i-3]}

        self.snail.ac_couplings_orders = ac_orders
        self.snail.dc_couplings_orders = dc_orders
        self.update(None)

    def update(self, val):
        # Update SNAIL parameters
       # self.snail.dl_Lj0 = self.slider_dl.val
        self.snail.Z = self.slider_Z.val/Rq
        self.snail.f_infty = self.slider_f.val
        self.snail.alpha = self.slider_alpha.val
        self.snail.Ej0 = self.slider_Ej0.val * 2* np.pi

        # Update calculations
        self.snail.update_f()

        # Clear previous plots
        for ax in [self.ax_freq, self.ax_kerr, self.ax_ac, self.ax_dc]:
            ax.clear()

        # Plot frequency
        self.ax_freq.plot(self.snail.reduced_flux, self.snail.f_fit_data)
        self.ax_freq.set_xlabel(r'$\Phi_e/\phi_q$')
        self.ax_freq.set_ylabel('Frequency (GHz)')
        self.ax_freq.set_title('Cavity Frequency')

        # Plot Kerr
        self.ax_kerr.plot(self.snail.reduced_flux, self.snail.Kerr)
        self.ax_kerr.set_xlabel(r'$\Phi_e/\phi_q$')
        self.ax_kerr.set_ylabel('Kerr (MHz)')
        self.ax_kerr.set_title('Kerr Coefficient')

        # Plot AC couplings
        for i, ac in enumerate(self.snail.ac_couplings):
            order = list(self.snail.ac_couplings_orders)[i]
            self.ax_ac.plot(self.snail.reduced_flux, ac, label=f'g_{order} AC')
        self.ax_ac.set_xlabel(r'$\Phi_e/\phi_q$')
        self.ax_ac.set_ylabel('Coupling (MHz)')
        self.ax_ac.set_title('AC Couplings')
        self.ax_ac.legend()

        # Plot DC couplings
        for i, dc in enumerate(self.snail.dc_couplings):
            order = list(self.snail.dc_couplings_orders)[i]
            self.ax_dc.plot(self.snail.reduced_flux, dc, label=f'g_{order} DC')
        self.ax_dc.set_xlabel(r'$\Phi_e/\phi_q$')
        self.ax_dc.set_ylabel('Coupling (MHz)')
        self.ax_dc.set_title('DC Couplings')
        self.ax_dc.legend()

        plt.tight_layout()
        self.fig.canvas.draw_idle()

# Create and show the interactive plot
if __name__ == "__main__":
    g3dc, g4dc, g5dc, g6dc, g3ac ,w0 = get_snail_couplings_at_zero_kerr(
        Ej0=245*2*np.pi, dl_Lj0=2.4133, f_infty=6.99*1e9, Z=57.9/Rq, alpha=0.0971, n=5
    )
    # Inputs: g3dc...w0 are in Linear Hz
    
    # Kerr calculation: K = 12 * (g4 - 5*g3^2/w0). 
    # Use consistent units! 
    # g3dc is Linear Hz, w0 is Linear Hz -> g3^2/w0 is Linear Hz. g4dc is Linear Hz.
    # Result Kerr is Linear Hz.
    Kerr = 12 * (g4dc - 5 * g3dc**2 / w0)
    
    print(f'f = {w0/1e9} GHz')
    print(f'Kerr = {Kerr/1e6} MHz ')
    print(f'g3dc/2pi = {g3dc/1e6} MHz')
    print(f'g4dc/2pi = {g4dc/1e6} MHz')
    print(f'g5dc/2pi = {g5dc/1e6} MHz')
    print(f'g6dc/2pi = {g6dc/1e6} MHz')
    print(f'g3ac/2pi = {g3ac/1e6} MHz')