#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 19:42:43 2023

@author: judewh
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the system of ODEs with conservation law constraint


def dPdt(t, P, a, g, gamma, r, d):
    M, C, T = P
    T = 1 - M - C
    dMdt = a*M*C - g*M/(M+T) + gamma*M*T
    dCdt = r*T*C - d*C - a*M*C
    dTdt = g*M/(M+T) - gamma*M*T - r*T*C + d*C
    return [dMdt, dCdt, dTdt]


# def dPdt(t, P, a, g, gamma, r, d,):
#     M, C, T = P
#     dMdt = a*M*C - g*M/(M+T) + gamma*M*T
#     dCdt = r*T*C - d*C - a*M*C
#     dTdt = g*M/(M+T) - gamma*M*T - r*T*C + d*C
#     return [dMdt, dCdt, dTdt]


# Parameter values
a = 0.1
g = 0.4
gamma = 0.8
r = 1
d = 0.44

# Initial conditions
M0 = 1/3
C0 = 1/3
T0 = 1/3
P0 = [M0, C0, T0]

# Time range
t_start = 0
t_end = 100
t = np.linspace(t_start, t_end, 1000)

# Solve the ODEs
sol = solve_ivp(dPdt, [t_start, t_end], P0, args=(a, g, gamma, r, d), t_eval=t)

# Plot the results

fig = plt.figure()
axes_data = fig.add_subplot(4, 1, (1, 3))
axes_data.plot(sol.t, sol.y[0], label='M(t) - Macroalgae')
axes_data.plot(sol.t, sol.y[1], label='C(t) - Coral')
axes_data.plot(sol.t, sol.y[2],
               label='T(t) - Algal Turfs')
axes_data.set_xlabel('Time')
axes_data.set_ylabel('Population')
axes_data.set_title('Population Change for Coral Reef System',
                    fontname='Arial', fontsize='14')
axes_data.grid(dashes=[4, 2], linewidth=1.2)
axes_data.legend()
axes_data.annotate((r'Parameters: a = {}, g = {}, $\gamma$ = {}, r = {}, d = {}'
                    .format(a, g, gamma, r, d)),
                   (0, 0), (25, -55), xycoords='axes fraction', va='top',
                   textcoords='offset points', fontsize='10')
axes_data.annotate((r'Initial Conditions: M(0) = {0:.2f}, C(0) = {0:.2f}, T(0) = {0:.2f}'
                    .format(M0, C0, T0)),
                   (0, 0), (25, -70), xycoords='axes fraction', va='top',
                   textcoords='offset points', fontsize='10')
plt.savefig('coral_reef_population_dynamics_0.4.png', dpi=300,
            bbox_inches='tight')
plt.show()
