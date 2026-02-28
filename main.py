"""
Triple Pendulum Simulation
===========================
A graphical simulation of a triple (3-link) pendulum using Lagrangian mechanics.

The equations of motion are derived from the Lagrangian L = T - V for three
point masses connected by rigid, massless rods. The resulting system of 6
first-order ODEs is solved with scipy.integrate.solve_ivp (RK45) and animated
in real time with matplotlib.

Controls:
    - Close the window to stop the simulation.

Dependencies: numpy, scipy, matplotlib
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ─────────────────────────── Physical parameters ───────────────────────────

G = 9.81  # gravitational acceleration (m/s²)

# Lengths of the three rods (m)
L1, L2, L3 = 1.0, 1.0, 1.0

# Masses of the three bobs (kg)
M1, M2, M3 = 1.0, 1.0, 1.0

# ──────────────────────── Initial conditions (rad, rad/s) ──────────────────

theta1_0 = np.radians(120)   # initial angle of rod 1
theta2_0 = np.radians(-10)   # initial angle of rod 2
theta3_0 = np.radians(25)    # initial angle of rod 3
omega1_0 = 0.0               # initial angular velocity of rod 1
omega2_0 = 0.0               # initial angular velocity of rod 2
omega3_0 = 0.0               # initial angular velocity of rod 3

# ──────────────────────── Simulation parameters ────────────────────────────

T_MAX = 60.0    # total simulation time (s)
DT = 0.005      # time step for the ODE solver output (s)
FPS = 50        # animation frames per second
TRAIL_LEN = 600 # number of past positions to draw in the trail


# ─────────────────── Equations of motion (mass-matrix form) ────────────────

def derivs(t, state):
    """
    Compute derivatives for the triple pendulum.

    State vector: [theta1, theta2, theta3, omega1, omega2, omega3]

    We form the mass matrix  M * [alpha1, alpha2, alpha3]^T = F
    where alpha_i = d(omega_i)/dt, then solve for the accelerations.
    """
    th1, th2, th3, w1, w2, w3 = state

    # Shorthand for trig
    c12 = np.cos(th1 - th2)
    c13 = np.cos(th1 - th3)
    c23 = np.cos(th2 - th3)
    s12 = np.sin(th1 - th2)
    s13 = np.sin(th1 - th3)
    s23 = np.sin(th2 - th3)
    s1 = np.sin(th1)
    s2 = np.sin(th2)
    s3 = np.sin(th3)

    # ── Mass matrix M (3x3, symmetric) ──
    # Derived from the Lagrangian for three point masses on rigid rods.
    #
    #   x1 = L1 sin(th1)
    #   y1 = -L1 cos(th1)
    #   x2 = x1 + L2 sin(th2)
    #   y2 = y1 - L2 cos(th2)
    #   x3 = x2 + L3 sin(th3)
    #   y3 = y2 - L3 cos(th3)

    m11 = (M1 + M2 + M3) * L1 ** 2
    m12 = (M2 + M3) * L1 * L2 * c12
    m13 = M3 * L1 * L3 * c13
    m22 = (M2 + M3) * L2 ** 2
    m23 = M3 * L2 * L3 * c23
    m33 = M3 * L3 ** 2

    M_mat = np.array([
        [m11, m12, m13],
        [m12, m22, m23],
        [m13, m23, m33],
    ])

    # ── Right-hand side (Coriolis / centripetal + gravity) ──
    f1 = (-(M2 + M3) * L1 * L2 * s12 * w2 ** 2
          - M3 * L1 * L3 * s13 * w3 ** 2
          - (M1 + M2 + M3) * G * L1 * s1)

    f2 = ((M2 + M3) * L1 * L2 * s12 * w1 ** 2
          - M3 * L2 * L3 * s23 * w3 ** 2
          - (M2 + M3) * G * L2 * s2)

    f3 = (M3 * L1 * L3 * s13 * w1 ** 2
          + M3 * L2 * L3 * s23 * w2 ** 2
          - M3 * G * L3 * s3)

    F = np.array([f1, f2, f3])

    # Solve  M * alpha = F  for angular accelerations
    alpha = np.linalg.solve(M_mat, F)

    return [w1, w2, w3, alpha[0], alpha[1], alpha[2]]


# ─────────────────────────── Integrate the ODE ─────────────────────────────

t_span = (0, T_MAX)
t_eval = np.arange(0, T_MAX, DT)
y0 = [theta1_0, theta2_0, theta3_0, omega1_0, omega2_0, omega3_0]

print("Solving ODE … ", end="", flush=True)
sol = solve_ivp(derivs, t_span, y0, method="RK45", t_eval=t_eval,
                rtol=1e-9, atol=1e-9, max_step=0.01)
print("done.")

th1_sol = sol.y[0]
th2_sol = sol.y[1]
th3_sol = sol.y[2]
w1_sol = sol.y[3]
w2_sol = sol.y[4]
w3_sol = sol.y[5]

# Convert to Cartesian coordinates
x1 = L1 * np.sin(th1_sol)
y1 = -L1 * np.cos(th1_sol)
x2 = x1 + L2 * np.sin(th2_sol)
y2 = y1 - L2 * np.cos(th2_sol)
x3 = x2 + L3 * np.sin(th3_sol)
y3 = y2 - L3 * np.cos(th3_sol)


# ─────────────────────────── Energy computation ────────────────────────────

def compute_energy(th1, th2, th3, w1, w2, w3):
    """Return (kinetic, potential, total) energy of the system."""
    # Velocities of each bob (Cartesian)
    vx1 = L1 * w1 * np.cos(th1)
    vy1 = L1 * w1 * np.sin(th1)

    vx2 = vx1 + L2 * w2 * np.cos(th2)
    vy2 = vy1 + L2 * w2 * np.sin(th2)

    vx3 = vx2 + L3 * w3 * np.cos(th3)
    vy3 = vy2 + L3 * w3 * np.sin(th3)

    T = 0.5 * M1 * (vx1 ** 2 + vy1 ** 2)
    T += 0.5 * M2 * (vx2 ** 2 + vy2 ** 2)
    T += 0.5 * M3 * (vx3 ** 2 + vy3 ** 2)

    V = M1 * G * (-L1 * np.cos(th1))
    V += M2 * G * (-L1 * np.cos(th1) - L2 * np.cos(th2))
    V += M3 * G * (-L1 * np.cos(th1) - L2 * np.cos(th2) - L3 * np.cos(th3))

    return T, V, T + V


KE, PE, TE = compute_energy(th1_sol, th2_sol, th3_sol,
                            w1_sol, w2_sol, w3_sol)

# ─────────────────────────── Animation setup ───────────────────────────────

# Subsample: pick every n-th frame so we hit the target FPS
frame_step = max(1, int(1.0 / (FPS * DT)))
n_frames = len(t_eval) // frame_step

fig, ax = plt.subplots(figsize=(8, 8))
fig.patch.set_facecolor("#1a1a2e")
ax.set_facecolor("#1a1a2e")

lim = (L1 + L2 + L3) * 1.15
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_aspect("equal")
ax.grid(True, color="#2a2a4a", linewidth=0.5, alpha=0.6)
ax.tick_params(colors="#888888")
for spine in ax.spines.values():
    spine.set_color("#2a2a4a")

title = ax.set_title("Triple Pendulum", fontsize=16, color="#e0e0ff",
                      fontweight="bold", pad=12)

# Rods
line, = ax.plot([], [], "o-", color="#ccccee", lw=2.2, markersize=8,
                markerfacecolor="#ff6f61", markeredgecolor="white",
                markeredgewidth=0.8, zorder=5)

# Trails (one per bob, with fading colour)
trail_colors = ["#ff6f61", "#6fc3ff", "#a8ff6f"]
trail_lines = []
for c in trail_colors:
    tl, = ax.plot([], [], "-", color=c, lw=0.8, alpha=0.55, zorder=2)
    trail_lines.append(tl)

# Energy text
energy_text = ax.text(0.02, 0.97, "", transform=ax.transAxes, fontsize=9,
                       color="#b0b0d0", verticalalignment="top",
                       fontfamily="monospace")
time_text = ax.text(0.98, 0.97, "", transform=ax.transAxes, fontsize=9,
                     color="#b0b0d0", verticalalignment="top",
                     horizontalalignment="right", fontfamily="monospace")

# Pivot marker
ax.plot(0, 0, "s", color="#ffffff", markersize=6, zorder=6)


def init():
    line.set_data([], [])
    for tl in trail_lines:
        tl.set_data([], [])
    energy_text.set_text("")
    time_text.set_text("")
    return (line, *trail_lines, energy_text, time_text)


# Trail history buffers
trail_x = [[], [], []]
trail_y = [[], [], []]


def animate(frame):
    i = frame * frame_step

    # Current positions
    pts_x = [0, x1[i], x2[i], x3[i]]
    pts_y = [0, y1[i], y2[i], y3[i]]
    line.set_data(pts_x, pts_y)

    # Update trails
    for k, (tx, ty) in enumerate(zip(trail_x, trail_y)):
        tx.append(pts_x[k + 1])
        ty.append(pts_y[k + 1])
        if len(tx) > TRAIL_LEN:
            tx.pop(0)
            ty.pop(0)
        trail_lines[k].set_data(tx, ty)

    # Energy readout
    energy_text.set_text(
        f"KE  = {KE[i]:+8.3f} J\n"
        f"PE  = {PE[i]:+8.3f} J\n"
        f"Tot = {TE[i]:+8.3f} J"
    )
    time_text.set_text(f"t = {t_eval[i]:6.2f} s")

    return (line, *trail_lines, energy_text, time_text)


ani = animation.FuncAnimation(fig, animate, init_func=init,
                              frames=n_frames, interval=1000 // FPS,
                              blit=True)

plt.tight_layout()
plt.show()
