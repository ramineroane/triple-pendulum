# Triple Pendulum Simulation

A physics-based simulation of a triple pendulum using Lagrangian mechanics, numerically integrated and rendered as an animated GIF.

👉 **Live demo:**  
https://ramineroane.github.io/triple-pendulum/

---

## Overview

This project simulates a system of three masses connected by rigid rods. The motion is highly nonlinear and exhibits chaotic behavior, making it a classic example of a coupled dynamical system.

The simulation:
- derives equations of motion from the Lagrangian \( L = T - V \)
- solves the resulting system of ordinary differential equations (ODEs)
- renders the motion into an animated visualization

---

## Preview

![Triple Pendulum](triple_pendulum.gif)

---

## Physics

Instead of directly applying Newton’s laws to each mass, this simulation uses **Lagrangian mechanics**, which is better suited for systems with constraints like pendulums.

- Generalized coordinates: \( \theta_1, \theta_2, \theta_3 \)
- State vector: angles and angular velocities
- Energies:
  - Kinetic energy \( T \)
  - Potential energy \( V \)

The equations of motion are derived from:

\[
L = T - V
\]

This leads to a coupled nonlinear system of equations that can be written in matrix form:

\[
M(\theta) \cdot \alpha = F(\theta, \omega)
\]

Where:
- \( M \) is the mass matrix  
- \( \alpha \) are angular accelerations  
- \( F \) includes gravity and nonlinear coupling terms  

---

## Numerical Method

The system is integrated using:

- `scipy.integrate.solve_ivp`
- Method: **RK45 (Runge–Kutta)**

This method provides stable and accurate time evolution, even for chaotic motion.

---

## Implementation

- **numpy** → numerical computations  
- **scipy** → ODE solving  
- **matplotlib** → animation and rendering  
- **pillow** → GIF export  

Pipeline:
1. Solve the ODE system over time
2. Convert angular positions to Cartesian coordinates
3. Render each frame
4. Export the animation as a GIF

---

## Why this is interesting

- Demonstrates chaotic dynamics in a simple mechanical system  
- Shows the power of Lagrangian mechanics for constrained systems  
- Combines physics, numerical methods, and visualization  

---

## Run locally

```bash
pip install numpy scipy matplotlib pillow
python your_script.py
