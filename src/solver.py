"""
Natural convection solver: 2D incompressible Navier-Stokes coupled with
temperature transport via the Boussinesq approximation.

Numerical method:
    - Staggered MAC grid (u on vertical faces, v on horizontal faces,
      p and T at cell centers).
    - Fractional-step (projection) time integration:
        1. Predictor: explicit u*, v* from convection + diffusion + buoyancy.
        2. Pressure Poisson solved by SOR until divergence is small.
        3. Corrector: u, v made divergence-free using grad(p).
        4. Temperature advected (1st-order upwind) and diffused (central).
    - Convection in momentum uses 2nd-order central differences.
    - Boundary conditions:
        - Bottom wall: localized hot patch on middle third, cold elsewhere.
        - Top wall: cold, optionally moving at Uwall (lid-driven).
        - Side walls: no-slip, adiabatic (Neumann on T).
"""

from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np


# Parameters
# ---------------------------------------------------------------------------

@dataclass
class Params:

    # Domain
    Lx: float = 3.0
    Ly: float = 2.0  # physical domain runs from y=-1 to y=+1

    # Grid
    Nx: int = 121
    Ny: int = 81

    # Fluid properties
    nu: float = 1.0e-4      # kinematic viscosity
    kappa: float = 1.0e-4   # thermal diffusivity
    Pr: float = 0.71        # Prandtl number (informational; nu/kappa is what's used)
    Ra: float = 1.0e6       # Rayleigh number — drives buoyancy strength

    # Temperatures
    T_hot: float = 4.0
    T_cold: float = 1.0

    # Boundary motion
    Uwall: float = 0.01     # top-wall (lid) velocity; set 0.0 for pure natural convection

    # Time control
    endT: float = 20.0      # end time in non-dimensional advective units
    CFL: float = 0.2        # convective CFL
    CFLv: float = 0.4       # viscous CFL

    # SOR pressure solver
    accel: float = 1.5
    err_tol: float = 1.0e-5
    max_sor_iters: int = 1000
    tiny: float = 1.0e-20


@dataclass
class Grid:
    """Cached grid quantities derived from Params."""
    dx: float
    dy: float
    dx2: float
    dy2: float
    x: np.ndarray   # face coords incl. ghosts
    y: np.ndarray
    xc: np.ndarray  # cell-center coords
    yc: np.ndarray

    @classmethod
    def from_params(cls, p: Params) -> "Grid":
        dx = p.Lx / float(p.Nx - 1)
        dy = p.Ly / float(p.Ny - 1)
        return cls(
            dx=dx,
            dy=dy,
            dx2=dx * dx,
            dy2=dy * dy,
            x=np.linspace(0.0, p.Lx, p.Nx + 2),
            y=np.linspace(-1.0, 1.0, p.Ny + 2),
            xc=np.linspace(0.0, p.Lx, p.Nx + 1),
            yc=np.linspace(-1.0, 1.0, p.Ny + 1),
        )


@dataclass
class State:
    """All simulation arrays, sized from params. Created via `State.zeros(p)`."""
    u: np.ndarray
    v: np.ndarray
    p: np.ndarray
    T: np.ndarray
    uaux: np.ndarray
    vaux: np.ndarray
    dive: np.ndarray

    @classmethod
    def zeros(cls, p: Params) -> "State":
        Nx, Ny = p.Nx, p.Ny
        return cls(
            u=np.zeros((Ny + 1, Nx + 2), dtype=np.float64),
            v=np.zeros((Ny + 2, Nx + 1), dtype=np.float64),
            p=np.zeros((Ny + 1, Nx + 1), dtype=np.float64),
            T=np.zeros((Ny + 1, Nx + 1), dtype=np.float64),
            uaux=np.zeros((Ny + 1, Nx + 2), dtype=np.float64),
            vaux=np.zeros((Ny + 2, Nx + 1), dtype=np.float64),
            dive=np.zeros((Ny + 1, Nx + 1), dtype=np.float64),
        )


# Time-step and buoyancy helpers


def reference_velocity(p: Params) -> float:
    """Reference velocity for non-dimensionalizing CFL."""
    return p.Uwall if p.Uwall != 0.0 else p.nu / p.Lx


def compute_dt(p: Params, g: Grid) -> float:
    """CFL-limited time step (convective and viscous limits)."""
    Uref = reference_velocity(p)
    return min(p.CFL * g.dx / Uref, p.CFLv * g.dx2 / p.nu)


def buoyancy_coefficient(p: Params) -> float:
    """
    Boussinesq buoyancy coefficient g*beta consistent with the chosen Ra.

    Definition: Ra = g * beta * dT * L^3 / (nu * kappa)
    so         g * beta = Ra * nu * kappa / (dT * L^3).

    Here L = Ly (the height) and dT = T_hot - T_cold.
    """
    dT = p.T_hot - p.T_cold
    return p.Ra * p.nu * p.kappa / (dT * p.Ly ** 3)


def initial_temperature(p: Params, g: Grid) -> np.ndarray:
    """Linear temperature profile from T_hot at bottom to T_cold at top."""
    T = np.zeros((p.Ny + 1, p.Nx + 1), dtype=np.float64)
    for j in range(p.Ny + 1):
        T[j, :] = p.T_hot + (p.T_cold - p.T_hot) * (g.yc[j] + 1.0) / 2.0
    return T



# Boundary conditions

def set_bc_temp(T: np.ndarray, p: Params, g: Grid) -> None:
    """Heated middle-third on bottom; cold top; adiabatic sides."""
    for i in range(p.Nx + 1):
        if p.Lx / 3.0 <= g.xc[i] <= 2.0 * p.Lx / 3.0:
            T[0, i] = p.T_hot
        else:
            T[0, i] = p.T_cold
    T[p.Ny, :] = p.T_cold
    T[1:p.Ny, 0] = T[1:p.Ny, 1]              # left wall, Neumann
    T[1:p.Ny, p.Nx] = T[1:p.Ny, p.Nx - 1]    # right wall, Neumann


def set_bc_u(u: np.ndarray, p: Params) -> None:
    u[:, 1] = 0.0                # left wall
    u[:, p.Nx] = 0.0             # right wall
    u[0, :] = -u[1, :]                            # bottom: no-slip via ghost
    u[p.Ny, :] = -u[p.Ny - 1, :] + 2.0 * p.Uwall  # top: lid


def set_bc_v(v: np.ndarray, p: Params) -> None:
    v[:, 0] = -v[:, 1]                # left wall: no-slip via ghost
    v[:, p.Nx] = -v[:, p.Nx - 1]      # right wall
    v[1, :] = 0.0                     # bottom
    v[p.Ny, :] = 0.0                  # top


def set_bc_pressure(pres: np.ndarray, p: Params) -> None:
    pres[0, 1:p.Nx] = pres[1, 1:p.Nx]
    pres[p.Ny, 1:p.Nx] = pres[p.Ny - 1, 1:p.Nx]
    pres[1:p.Ny, 0] = pres[1:p.Ny, 1]
    pres[1:p.Ny, p.Nx] = pres[1:p.Ny, p.Nx - 1]


# Predictor: u* and v* with convection, diffusion, and buoyancy


def calc_aux_u(uaux, u, v, p: Params, g: Grid, dt: float) -> None:
    """Explicit predictor for u-velocity. No body force in x-direction."""
    Nx, Ny = p.Nx, p.Ny
    dx, dy, dx2, dy2 = g.dx, g.dy, g.dx2, g.dy2
    for jc in range(1, Ny):
        for i in range(1, Nx + 1):
            visc = ((u[jc, i - 1] - 2.0 * u[jc, i] + u[jc, i + 1]) / dx2
                    + (u[jc - 1, i] - 2.0 * u[jc, i] + u[jc + 1, i]) / dy2)
            conv = ((+(u[jc, i - 1] + u[jc, i]) / 2.0
                     * (-u[jc, i - 1] + u[jc, i]) / dx
                     + (u[jc, i] + u[jc, i + 1]) / 2.0
                     * (-u[jc, i] + u[jc, i + 1]) / dx) / 2.0
                    + (+(v[jc, i - 1] + v[jc, i]) / 2.0
                       * (-u[jc - 1, i] + u[jc, i]) / dy
                       + (v[jc + 1, i - 1] + v[jc + 1, i]) / 2.0
                       * (-u[jc, i] + u[jc + 1, i]) / dy) / 2.0)
            uaux[jc, i] = u[jc, i] + dt * (-conv + p.nu * visc)


def calc_aux_v(vaux, u, v, T, p: Params, g: Grid, dt: float, gbeta: float, T_ref: float) -> None:
    """
    Explicit predictor for v-velocity, including Boussinesq buoyancy:
        dv/dt += g*beta*(T - T_ref)

    T lives at cell centers (jc, ic). v at (j, ic) sits between cell-centers
    (j-1, ic) and (j, ic), so we average T to the v-location.
    """
    Nx, Ny = p.Nx, p.Ny
    dx, dy, dx2, dy2 = g.dx, g.dy, g.dx2, g.dy2
    for j in range(1, Ny + 1):
        for ic in range(1, Nx):
            visc = ((v[j - 1, ic] - 2.0 * v[j, ic] + v[j + 1, ic]) / dy2
                    + (v[j, ic - 1] - 2.0 * v[j, ic] + v[j, ic + 1]) / dx2)
            conv = ((+(u[j - 1, ic] + u[j, ic]) / 2.0
                     * (-v[j, ic - 1] + v[j, ic]) / dx
                     + (u[j - 1, ic + 1] + u[j, ic + 1]) / 2.0
                     * (-v[j, ic] + v[j, ic + 1]) / dx) / 2.0
                    + (+(v[j - 1, ic] + v[j, ic]) / 2.0
                       * (-v[j - 1, ic] + v[j, ic]) / dy
                       + (v[j, ic] + v[j + 1, ic]) / 2.0
                       * (-v[j, ic] + v[j + 1, ic]) / dy) / 2.0)
            # Buoyancy: temperature averaged to v-location
            T_at_v = 0.5 * (T[j - 1, ic] + T[j, ic])
            buoy = gbeta * (T_at_v - T_ref)
            vaux[j, ic] = v[j, ic] + dt * (-conv + p.nu * visc + buoy)



# Pressure Poisson via SOR
# ---------------------------------------------------------------------------

def divergence(div, u, v, p: Params, g: Grid, dt: float) -> None:
    """Divergence of u*, scaled by 1/dt (RHS of pressure Poisson)."""
    Nx, Ny = p.Nx, p.Ny
    for jc in range(1, Ny):
        for ic in range(1, Nx):
            div[jc, ic] = ((-u[jc, ic] + u[jc, ic + 1]) / g.dx
                           + (-v[jc, ic] + v[jc + 1, ic]) / g.dy) / dt


def sor_sweep(pres, div, p: Params, g: Grid) -> float:
    """One Gauss-Seidel sweep of SOR. Returns relative residual."""
    Nx, Ny = p.Nx, p.Ny
    dx2, dy2 = g.dx2, g.dy2
    err_n = 0.0
    err_d = 0.0
    denom = (dx2 + dy2) * 2.0
    for jc in range(1, Ny):
        for ic in range(1, Nx):
            d_pres = (dy2 * (pres[jc, ic - 1] + pres[jc, ic + 1])
                      + dx2 * (pres[jc - 1, ic] + pres[jc + 1, ic])
                      - (dx2 * dy2 * div[jc, ic])) / denom - pres[jc, ic]
            pres[jc, ic] += p.accel * d_pres
            err_n += d_pres * d_pres
            err_d += pres[jc, ic] * pres[jc, ic]
    set_bc_pressure(pres, p)
    if err_d < p.tiny:
        err_d = 1.0
    return float(np.sqrt(err_n / err_d))


def solve_pressure(pres, div, p: Params, g: Grid) -> int:
    """Iterate SOR until residual < err_tol or max_sor_iters reached."""
    err_r = 1.0
    n = 0
    while err_r > p.err_tol and n < p.max_sor_iters:
        n += 1
        err_r = sor_sweep(pres, div, p, g)
    return n

# Corrector

def correct_u(u, uaux, pres, p: Params, g: Grid, dt: float) -> None:
    Nx, Ny = p.Nx, p.Ny
    for jc in range(1, Ny):
        for i in range(1, Nx + 1):
            u[jc, i] = uaux[jc, i] - dt * (-pres[jc, i - 1] + pres[jc, i]) / g.dx


def correct_v(v, vaux, pres, p: Params, g: Grid, dt: float) -> None:
    Nx, Ny = p.Nx, p.Ny
    for j in range(1, Ny + 1):
        for ic in range(1, Nx):
            v[j, ic] = vaux[j, ic] - dt * (-pres[j - 1, ic] + pres[j, ic]) / g.dy
# Temperature transport (1st-order upwind convection + central diffusion)

def solve_temperature(T, u, v, p: Params, g: Grid, dt: float) -> np.ndarray:
    Nx, Ny = p.Nx, p.Ny
    dx, dy, dx2, dy2 = g.dx, g.dy, g.dx2, g.dy2
    Tnew = T.copy()
    max_dT = 0.1 * (p.T_hot - p.T_cold)  # NOTE: stability limiter; remove if scheme is improved
    T_lo, T_hi = p.T_cold * 0.9, p.T_hot * 1.1
    for j in range(1, Ny):
        for i in range(1, Nx):
            # Upwind convection — uses u, v at the same index, an approximation
            # since they live on faces (kept as in original code for fidelity).
            if u[j, i] > 0:
                dTdx = (T[j, i] - T[j, i - 1]) / dx
            else:
                dTdx = (T[j, i + 1] - T[j, i]) / dx

            if v[j, i] > 0:
                dTdy = (T[j, i] - T[j - 1, i]) / dy
            else:
                dTdy = (T[j + 1, i] - T[j, i]) / dy

            d2Tdx2 = (T[j, i + 1] - 2.0 * T[j, i] + T[j, i - 1]) / dx2
            d2Tdy2 = (T[j + 1, i] - 2.0 * T[j, i] + T[j - 1, i]) / dy2

            dT = dt * (p.kappa * (d2Tdx2 + d2Tdy2)
                       - (u[j, i] * dTdx + v[j, i] * dTdy))
            dT = np.clip(dT, -max_dT, max_dT)
            Tnew[j, i] = np.clip(T[j, i] + dT, T_lo, T_hi)
    return Tnew


# One full time step


def step(state: State, p: Params, g: Grid, dt: float, gbeta: float, T_ref: float) -> int:
    """Advance the simulation by one dt. Returns the SOR iteration count."""
    # 1. Momentum predictor
    calc_aux_u(state.uaux, state.u, state.v, p, g, dt)
    set_bc_u(state.uaux, p)
    calc_aux_v(state.vaux, state.u, state.v, state.T, p, g, dt, gbeta, T_ref)
    set_bc_v(state.vaux, p)

    # 2. Pressure Poisson
    divergence(state.dive, state.uaux, state.vaux, p, g, dt)
    sor_iters = solve_pressure(state.p, state.dive, p, g)

    # 3. Velocity correction
    correct_u(state.u, state.uaux, state.p, p, g, dt)
    set_bc_u(state.u, p)
    correct_v(state.v, state.vaux, state.p, p, g, dt)
    set_bc_v(state.v, p)

    # 4. Temperature
    state.T = solve_temperature(state.T, state.u, state.v, p, g, dt)
    set_bc_temp(state.T, p, g)

    return sor_iters


# Diagnostics (post-processing helpers — not used inside the time loop)


def heat_flux(T: np.ndarray, g: Grid, kappa: float):
    """Cell-centered heat flux components q = -kappa * grad(T)."""
    qx = np.zeros_like(T)
    qy = np.zeros_like(T)
    qx[1:-1, 1:-1] = -kappa * (T[1:-1, 2:] - T[1:-1, :-2]) / (2.0 * g.dx)
    qy[1:-1, 1:-1] = -kappa * (T[2:, 1:-1] - T[:-2, 1:-1]) / (2.0 * g.dy)
    return qx, qy


def linear_reference_T(p: Params, g: Grid) -> np.ndarray:
    """Pure-conduction reference profile (linear in y) for L2 comparison."""
    Tref = np.zeros((p.Ny + 1, p.Nx + 1), dtype=np.float64)
    for j in range(p.Ny + 1):
        Tref[j, :] = p.T_cold + (p.T_hot - p.T_cold) * (1.0 - (g.yc[j] + 1.0) / 2.0)
    return Tref


def l2_error(T: np.ndarray, T_ref: np.ndarray) -> float:
    return float(np.sqrt(np.mean((T - T_ref) ** 2)))


# Top-level driver


def run(
    p: Optional[Params] = None,
    callback: Optional[Callable[[int, float, "State"], None]] = None,
    callback_every: int = 100,
):
    """

    Args:
        p: simulation parameters (defaults to Params()).
        callback: optional function called as `callback(step_idx, time, state)`
                  every `callback_every` steps. Use this for plotting,
                  logging, or recording diagnostics — keeps solver.py free
                  of plotting concerns.
        callback_every: stride for callback invocation.

    Returns:
        (state, params, grid, dt) at the final time.
    """
    if p is None:
        p = Params()
    g = Grid.from_params(p)
    state = State.zeros(p)
    state.T = initial_temperature(p, g)
    set_bc_temp(state.T, p, g)

    dt = compute_dt(p, g)
    Uref = reference_velocity(p)
    Nt = int(p.endT * p.Lx / Uref / dt)

    gbeta = buoyancy_coefficient(p)
    T_ref = 0.5 * (p.T_hot + p.T_cold)  # Boussinesq reference temperature

    for itr in range(Nt):
        step(state, p, g, dt, gbeta, T_ref)
        if callback is not None and itr % callback_every == 0:
            callback(itr, itr * dt, state)

    return state, p, g, dt
