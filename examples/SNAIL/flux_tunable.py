from numpy import *
from math import factorial
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy import optimize as opt

def U_snail(φ, flux_e, α, n=3, **kwargs):
    '''adimensionnal SNAIL potential '''
    U = - α * cos(φ) - n*cos((flux_e - φ) / n)
    return U

def dderiv (flux_e, φ, α, n) :
    return α*cos(φ) + cos((flux_e - φ)/n)/n
def ddderiv (flux_e, φ, α, n):
    return -α*sin(φ) + sin((flux_e - φ)/n)/n**2
def dddderiv (flux_e, φ, α, n):
    return -α*cos(φ) - cos((flux_e - φ)/n)/n**3
def ddddderiv (flux_e, φ, α, n):
    return α*sin(φ) - sin((flux_e - φ)/n)/n**4
def dddddderiv (flux_e, φ, α, n):
    return α*cos(φ) + cos((flux_e - φ)/n)/n**5

def derivative (flux_e, φmin, α, n, order) :
    if order == 1 :
        return α*sin(φmin) - sin((flux_e - φmin)/n)
    elif order == 2 :
        return α*cos(φmin) + cos((flux_e - φmin)/n)/n
    elif order == 3 :
        return -α*sin(φmin) + sin((flux_e - φmin)/n)/n**2
    elif order == 4 :
        return -α*cos(φmin) - cos((flux_e - φmin)/n)/n**3
    elif order == 5 :
        return α*sin(φmin) - sin((flux_e - φmin)/n)/n**4
    elif order == 6 :
        return α*cos(φmin) + cos((flux_e - φmin)/n)/n**5

def ACderivative (flux_e, φmin, α, n, order) :
    if order == 1 :
        return - cos((flux_e - φmin)/n) / n
    elif order == 2 :
        return - sin((flux_e - φmin)/n) / n**2
    elif order == 3 :
        return cos((flux_e - φmin)/n) / n**3
    elif order == 4 :
        return sin((flux_e - φmin)/n) / n**4
    elif order == 5 :
        return - cos((flux_e - φmin)/n) / n**5

def find_U_minimum (flux_e, α, n=3, fast=False) :
    '''
    finds the minimum of U_snail for α < 1. fast=True to bypass zero finding.
    '''
    if α > 1 : print('Warning: α > 1. Other minima are probable.')
    guess = flux_e - n*α*sin(flux_e)
    if fast : return guess
    def func (φ, flux_e, α, n) :
        return α*sin(φ) - sin((flux_e-φ)/n)

    def deriv (φ, flux_e, α, n) :
        return α*cos(φ) + cos((flux_e-φ)/n)/n

    res = opt.newton(func, guess, fprime=deriv, args=(flux_e, α, n))
    return res

def Leff_SNAIL (flux_e, α, n=3, **kwargs) :
    '''
    return adimensional quadratic part of SNAIL potential. fast=True to bypass zero finding.
    '''
    φ0 = find_U_minimum(flux_e, α, n, **kwargs)
    return abs(1/dderiv(flux_e, φ0, α, n))

def Leff_SQUID (flux_e, d=0, **kwargs) :
    '''
    return adimensional quadratic part of SQUID potential
    '''
    return 1/(sqrt(cos(flux_e)**2 + d**2*sin(flux_e)**2))

def rootFunc (x, A, l) :
    return (arctan(A/x) + l*pi - x)

def boundary_solve (A, l=0) :
    '''
    finds frequency shift due to boundary L.

        Parameters:
            A : A = dl/L_J, d the resonator length, l its lineic inductance, L_J the boundary inductance.

        Returns:
            res: (π/2) * ω_0/ω_infty, the lowest mode frequency shift.
    '''
    res = opt.brentq(rootFunc, 1e-9 + l*pi, pi/2 + l*pi, args=(A, l))
    return res

V_boundary_solve = vectorize(boundary_solve)


def cavity_freq (fluxList, dl_Lj0, f_infty, α=None, l=0, **kwargs) :
    '''
    returns the predicted cavity angl. frequency

        Parameters:
            fluxList : list of admin external fluxes (angular flux)
            dl_Lj    : d*l_Lj0, resonator length * lineic inductance / Lj0
            f_infty  : cavity pulsation without boundary.
    '''
    if α :
        A = dl_Lj0/Leff_SNAIL(fluxList, α, **kwargs)
    else :
        A = dl_Lj0/Leff_SQUID(fluxList, **kwargs)

    return V_boundary_solve(A, l) * f_infty*2/pi



def fit_cavity_freq (vList, fList, model=('SNAIL', None), **kwargs) :
    '''
    fits measured frequencies to a model of boundary SNAIL or SQUID.

        parameters:
            vList : list of voltages
            fList : list of frequencies
            model : tuple (m, x). m is either 'SNAIL' or 'SQUID'. x is resp α or d. If x=None, its value is fitted.

        returns:
            (dfluxdv, vOff, Z, f_infty, x) :
                dfluxdv : prop. between adim. angl. flux and vList.
                voff : external flux calibrated as flux = dfluxdv(vList - voff).
                dl_Lj0  : resonator impedance (adim)
                f_infty : resonator bare frequency (unit of fList)
                x  : either α or d depending of the model.
    '''
    p0 = kwargs['p0'] if 'p0' in kwargs else [0, 1, 1, 10]
    if 'p0' in kwargs : del kwargs['p0']

    if model[0] == 'SNAIL' :
        if model[1] :
            (voff, dfluxdv, dl_Lj0, f0), _ = opt.curve_fit(
                lambda v, voff, dfluxdv, dl_Lj0, f0 :cavity_freq((v-voff)*dfluxdv, dl_Lj0, f0, α=model[1], **kwargs),
                vList, fList,
                bounds=([-inf,0,1e-3,0], [inf, inf, inf, inf]),
                p0=p0
            )
            return (voff, dfluxdv, dl_Lj0, f0, model[1])
        else :
            p0.append(0.1)
            (voff, dfluxdv, dl_Lj0, f0, α), _ = opt.curve_fit(
                lambda v, voff, dfluxdv, dl_Lj0, f0, α :cavity_freq((v-voff)*dfluxdv, dl_Lj0, f0, α=α, **kwargs),
                vList, fList,
                bounds=([-inf, 0,1e-3,0,1e-2], [inf, inf, inf, inf, 0.5]),
                p0=p0
            )
            return (voff, dfluxdv, dl_Lj0, f0, α)

    elif model[0] == 'SQUID' :
        if model[1] :
            d = model[1]
            (voff, dfluxdv, dl_Lj0, f0), _ = opt.curve_fit(
                lambda v, voff, dfluxdv, dl_Lj0, f0 :cavity_freq((v-voff)*dfluxdv, dl_Lj0, f0, d=model[1], **kwargs),
                vList, fList,
                bounds=([-inf,0,1e-3,0], [inf, inf, inf, inf]),
                p0=p0
            )
            return (voff, dfluxdv, dl_Lj0, f0, model[1])
        else :
            p0.append(0.01)
            (voff, dfluxdv, dl_Lj0, f0), _ = opt.curve_fit(
                lambda v, voff, dfluxdv, dl_Lj0, f0, d :cavity_freq((v-voff)*dfluxdv, dl_Lj0, f0, d=d, **kwargs),
                vList, fList,
                bounds=([-inf, 0,1e-3,0,0], [inf, inf, inf, inf, 1]),
                p0=p0
            )
            return (voff, dfluxdv, dl_Lj0, f0, d)


def find_kerr_free (α, n=3, **kwargs) :
    '''
    finds the kerr free point, assuming no effective kerr due to φ^3 interactions or qubit interactions.

        parameters:
            α
            n

        returns:
            kf_flux : the kerr free external flux (angular flux)
    '''

    a0 = arcsin(1/(α*n**3))
    if a0 != a0 : raise ValueError(f'No kfp for  1/n**3 = {1/n**3:.3g} > α = {α:.3g} ')
    root = opt.newton(
        lambda φ : a0 + n*α*cos(φ) - φ,
        a0 + n*α,
        fprime = lambda φ : -n*α*sin(φ) - 1,
        fprime2 = lambda φ : -n*α*cos(φ)
    )


    kf_flux = opt.newton(
        lambda flux_e : dddderiv(flux_e, find_U_minimum(flux_e, α, n, **kwargs), α, n),
        root + pi/2
    )

    return kf_flux

def couplings (flux_e, Ej0, dl_Lj0, w_infty, Z, α, n=3, order=[3,]) :

    if isinstance(order, int) : 
        order = [order,]

    if isinstance(order, list) and all(isinstance(elem, int) for elem in order):
    
        φmin = find_U_minimum(flux_e, α, n)
        w_0 = cavity_freq(flux_e, dl_Lj0, w_infty, α)

        δ_0 = pi * w_0/w_infty
        φ0 = sqrt(Z/δ_0 * (1+cos(δ_0))/(1+sin(δ_0)/δ_0) )

        out = [Ej0*derivative(flux_e, φmin, α, n, order=o) * φ0**o / factorial(o) for o in order]
        out.append(w_0)
        return out

    else : 
        raise ValueError('order value must be int of list of int.')

def ACcouplings (flux_e, Ej0, dl_Lj0, w_infty, Z, α, n=3, order=[1,]) :

    if isinstance(order, int) : 
        order = [order,]

    if isinstance(order, list) and all(isinstance(elem, int) for elem in order):
    
        φmin = find_U_minimum(flux_e, α, n)
        w_0 = cavity_freq(flux_e, dl_Lj0, w_infty, α)

        δ_0 = pi * w_0/w_infty
        φ0 = sqrt(Z/δ_0 * (1+cos(δ_0))/(1+sin(δ_0)/δ_0) )

        out = [Ej0*ACderivative(flux_e, φmin, α, n, order=o) * φ0**o / factorial(o) for o in order]
        out.append(w_0)
        return out

    else : 
        raise ValueError('order value must be int of list of int.')




def refine_kerr_free(Ej0, dl_Lj0, f_infty, Z, α, n=3) :
    '''
    finds the kerr free point, assuming g4' \propto g4 - 5 g3^2/w_0.
    '''
    def temp_func (flux_e, Ej0, dl_Lj0, f_infty, Z, α, n=3) :
        f0, g3, g4, _ = couplings(flux_e, Ej0, dl_Lj0, f_infty, Z, α, n)
        return g4 - 5* g3**2/(2*pi*f0)
    kf_flux = find_kerr_free(α, n)
    rkf_flux = opt.newton(
        temp_func,
        kf_flux,
        args=(Ej0, dl_Lj0, f_infty, Z, α)
    )
    return rkf_flux


def find_flux (freq, fcav_cs) :
    ''' inverts the freq=f(flux) relation given by spline interp fcav_cs.'''
    res = opt.brentq(lambda flux : fcav_cs(flux) - freq, 0, pi)
    return res
