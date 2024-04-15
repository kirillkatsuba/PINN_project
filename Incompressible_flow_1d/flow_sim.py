import numpy as np

import matplotlib
matplotlib.rcParams['image.cmap'] = 'jet'
import matplotlib.pyplot as plt

from numba import jit

import scipy.special as sp

### Corey model
@jit(nopython=True, fastmath=True)
def compute_rel_phase_perm_wat(s, p, k):
    ### s - water saturation
    ### p - power parameter
    ### k - maximum relative phase permeability
    eps = 1.0e-10
    sp = (s + eps) / (1.0 + eps)
    return k * (sp ** p)

@jit(nopython=True, fastmath=True)
def compute_rel_phase_perm_oil(s, p, k):
    ### s - oil saturation
    ### p - power parameter
    ### k - maximum relative phase permeability
    eps = 1.0e-10
    sp = (s + eps) / (1.0 + eps)
    return k * (sp ** p)

@jit(nopython=True, fastmath=True)
def compute_fractional_flow_wat(swat, pwat, kwat, koil, poil, vr):
    rel_perm_wat = compute_rel_phase_perm_wat(0.0 + swat, pwat, kwat)
    rel_perm_oil = compute_rel_phase_perm_wat(1.0 - swat, poil, koil)
    return rel_perm_wat / (rel_perm_wat + rel_perm_oil * vr)

@jit(nopython=True, fastmath=True)
def compute_fractional_flow_oil(swat, pwat, kwat, koil, poil, vr):
    rel_perm_wat = compute_rel_phase_perm_wat(0.0 + swat, pwat, kwat)
    rel_perm_oil = compute_rel_phase_perm_wat(1.0 - swat, poil, koil)
    return rel_perm_oil / (rel_perm_oil + rel_perm_wat / vr)


@jit(nopython=True, fastmath=True)
def compute_absolute_mobility(swat, pwat, kwat,
                              soil, poil, koil,
                              vr):
    ### vr - is the ratio of viscosities: oil / water
    term0 = 0.0 * swat
    term0 += compute_rel_phase_perm_wat(swat, pwat, kwat)
    term0 += compute_rel_phase_perm_oil(soil, poil, koil) * vr
    lmob = 1.0 / term0
    return lmob

@jit(nopython=True, fastmath=True)
def compute_mobility_integral(sdata, pwat, kwat, poil, koil, vr):
    ndata = sdata.size
    swat = sdata.copy()
    soil = 1.0 - swat
    lmob = compute_absolute_mobility(swat, pwat, kwat, soil, poil, koil, vr)
    return np.mean(lmob)

@jit(nopython=True, fastmath=True)
def compute_velocity(pmin, pmax, sdata,
                     pwat, kwat, poil, koil, vr):
    lmob = compute_mobility_integral(sdata, pwat, kwat, poil, koil, vr)
    return (pmax - pmin) / lmob

@jit(nopython=True, fastmath=True)
def update_saturations(s, dt, dx, poro, pmin, pmax,
                       pwat, kwat, poil, koil, vr):
    u = compute_velocity(pmin, pmax, s, pwat, kwat, poil, koil, vr)
    b = compute_fractional_flow_wat(s, pwat, kwat, koil, poil, vr)
    s_update = s.copy()
    u_flow = u / poro * dt / dx
    s_update -=  u_flow * b
    s_update[1:] += u_flow * b[:-1]
    s_update[0] += u_flow * 1.0
    s_update[0] = s[0]
    return s_update

@jit(nopython=True, fastmath=True)
def compute_saturation_data(nx, nt, tfinal, poro, pmin, pmax,
                            pwat, kwat, poil, koil, vr):
    dt = tfinal / nt
    dx = 1.0 / nx
    ### init saturations
    s0 = np.zeros((nx))
    s0[0] = 1.0
    for kt in range(nt):
        s1 = update_saturations(s0, dt, dx, poro, pmin, pmax, pwat, kwat, poil, koil, vr)
        s0 = s1.copy()
    return s0

@jit(nopython=True, fastmath=True)
def compute_pressure_gradient_data(pmin, pmax, sdata, pwat, kwat, poil, koil, vr):
    u = compute_velocity(pmin, pmax, sdata, pwat, kwat, poil, koil, vr)
    pgrad = -u * compute_absolute_mobility(sdata, pwat, kwat, 1.0 - sdata, poil, koil, vr)
    return pgrad

@jit(nopython=True, fastmath=True)
def compute_pressure(x, pmax, pgrad):
    nx = pgrad.size
    dx = 1.0 / nx
    idx = min(np.int64(x / dx), nx - 1)
    pres = pmax + np.sum(pgrad[:idx]) * dx
    pres += pgrad[idx] * (x - idx * dx)
    return pres

@jit(nopython=True, fastmath=True)
def compute_flow_velocity(x, pmin, pmax, sdata, pwat, kwat, poil, koil, vr):
    pgrad = compute_pressure_gradient_data(pmin, pmax, sdata, pwat, kwat, poil, koil, vr)
    r_wat = compute_rel_phase_perm_wat(0.0 + sdata, pwat, kwat)
    r_oil = compute_rel_phase_perm_wat(1.0 - sdata, poil, koil)
    u_wat = - r_wat * pgrad
    u_oil = - r_oil * pgrad    
    return (u_wat, u_oil)

@jit(nopython=True, fastmath=True)
def compute_solution(t, x, nx=100,
                     pmin=0.0, pmax=1.0,
                     poro=0.1, vr=3.0,
                     pwat=2.0, poil=4.0,
                     kwat=1.0, koil=0.1):
    ### t and x are real numbers
    dx = 1.0 / nx
    dt_min = 0.1 * dx  ### this is needed for stability
    nt = np.int64(t / dt_min) + 1

    s = compute_saturation_data(nx, nt, t, poro, pmin, pmax,
                                pwat, kwat, poil, koil, vr)

    ### saturations:
    idx = min(np.int64(x / dx), nx - 1)
    swat = s[idx]
    soil = 1.0 - swat
    ### array of pressure gradients
    pgrad_data = compute_pressure_gradient_data(pmin, pmax, s, pwat, kwat, poil, koil, vr)
    ### pressure gradient value at x:
    pgrad = pgrad_data[idx]
    ### preessure value:
    press = compute_pressure(x, pmax, pgrad_data)
    ### flow velocities
    u_wat, u_oil = compute_flow_velocity(x, pmin, pmax, s, pwat, kwat, poil, koil, vr)
    u_wat = u_wat[idx]
    u_oil = u_oil[idx]
    return (pgrad, press, swat, soil, u_wat, u_oil)


def numerical_solution_test():
    nx = 200
    nt = 2000
    tfinal = 1.0
    poro = 0.1
    pmin = 0.0
    pmax = 1.0
    pwat = 2.0
    poil = 4.0
    kwat = 1.0
    koil = 0.1
    vr = 3.0
    s = compute_saturation_data(nx, nt, tfinal, poro, pmin, pmax, pwat, kwat, poil, koil, vr)
    x = np.linspace(0.5 / nx, 1.0 - 0.5 / nx, nx)
    plt.figure()
    plt.title('saturation distribution')
    plt.plot(x, s, linewidth=4.0)
    plt.grid()
    plt.xlabel('coordinate')
    plt.ylabel('saturation')
    plt.show()
    return 0





