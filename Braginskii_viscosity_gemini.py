#! /usr/bin/env python3

import numpy as np

class BraginskiiViscositySI:
    def __init__(self):
        # Physical Constants in CGS (Gaussian) units
        self.e = 4.8032e-10        # Elementary charge (statC)
        self.k_B = 1.3807e-16      # Boltzmann constant (erg/K)
        self.c = 2.9979e10         # Speed of light (cm/s)
        self.m_p = 1.6726e-24      # Proton mass (g)
        self.eV_to_erg = 1.6022e-12 

    def calculate(self, n_cm3, T_eV, B_G, Z=1, mu=1, lambda_log=16.5):
        # 1. Setup variables in CGS
        m_i = mu * self.m_p
        kT_erg = T_eV * self.eV_to_erg
        
        # 2. Calculate Frequencies and Times
        # omega_ci = Z * e * B / (m_i * c)
        # omega_ci = (Z * self.e * B_G) / (m_i * self.c)
        omega_ci = 9.58e3 * Z * B_G / mu  # in rad/s
        
        # tau_i = (3 * sqrt(m_i) * (kT)^3/2) / (4 * sqrt(pi) * n * lambda * e^4)
        # numerator = 3 * np.sqrt(m_i) * np.power(kT_erg, 1.5)
        # denominator = 4 * np.sqrt(np.pi) * n_cm3 * lambda_log * np.power(self.e, 4)
        # tau_i = numerator / denominator
        tau_i = 2.09e7 * (T_eV**1.5) * np.sqrt(mu) / (n_cm3 * lambda_log)  # in seconds
        
        # 3. Calculate Dynamic Viscosities in CGS (Poise)
        nkT = n_cm3 * kT_erg
        
        # Dictionary to store eta (CGS)
        eta_cgs = {}
        eta_cgs['eta_0'] = 0.96 * nkT * tau_i
        eta_cgs['eta_1'] = (3 * nkT) / (10 * (omega_ci**2) * tau_i)
        eta_cgs['eta_2'] = (6 * nkT) / (5 * (omega_ci**2) * tau_i)
        eta_cgs['eta_3'] = nkT / (2 * omega_ci)
        eta_cgs['eta_4'] = nkT / omega_ci # As per image 1
        
        # 4. Convert to SI Units
        # Dynamic Viscosity: 1 Poise = 0.1 kg/(m*s)
        eta_si = {k: v * 0.1 for k, v in eta_cgs.items()}
        
        # Mass Density
        rho_cgs = n_cm3 * m_i         # g/cm^3
        rho_si = rho_cgs * 1000.0     # kg/m^3
        
        # Kinematic Viscosity: nu = eta / rho (m^2/s)
        nu_si = {k: v / rho_si for k, v in eta_si.items()}
        
        return {
            "parameters": {"n": n_cm3, "T": T_eV, "B": B_G},
            "derived": {"tau_i": tau_i, "omega_ci": omega_ci, "rho_si": rho_si},
            "eta_si": eta_si,
            "nu_si": nu_si
        }

# --- Example Usage ---
# Using typical Tokamak edge parameters
calc = BraginskiiViscositySI()
res = calc.calculate(n_cm3=1.01e14, T_eV=4573.63, B_G=25700.0, Z=1.0, mu=1.0)

print(f"--- Plasma Parameters ---")
print(f"Density: {res['parameters']['n']:.1e} cm^-3")
print(f"Temp:    {res['parameters']['T']} eV")
print(f"B Field: {res['parameters']['B']} G")
print(f"Mass Density: {res['derived']['rho_si']:.4e} kg/m^3")
print(f"Ion Collision Time: {res['derived']['tau_i']:.4e} s")
print(f"Ion Cyclotron Frequency: {res['derived']['omega_ci']:.4e} rad/s")
print("-" * 65)
print(f"{'Viscosity':<10} | {'Dynamic (kg/(m*s))':<20} | {'Kinematic (m^2/s)':<20}")
print("-" * 65)
keys = ['eta_0', 'eta_1', 'eta_2', 'eta_3', 'eta_4']
for k in keys:
    val_eta = res['eta_si'][k]
    val_nu = res['nu_si'][k]
    print(f"{k:<10} | {val_eta:<20.4e} | {val_nu:<20.4e}")