#! /usr/bin/env python3

"""
Braginskii Viscosity Calculator
================================
Calculates ion viscosity coefficients for magnetized plasmas.
Units: CGS with temperature in eV
"""

import numpy as np

# Physical constants (CGS units)
E_CHARGE = 4.803e-10  # esu (statcoulomb)
M_PROTON = 1.673e-24  # g
K_BOLTZMANN = 1.381e-16  # erg/K
EV_TO_ERG = 1.602e-12  # erg/eV
C_LIGHT = 2.998e10  # cm/s

def calculate_collision_time(T_eV, n, Z=1, A=1, ln_Lambda=15):
    """
    Calculate ion collision time τ_i
    
    Parameters:
    -----------
    T_eV : float
        Temperature in eV
    n : float
        Number density in cm^-3
    Z : int
        Ion charge number (default: 1 for hydrogen)
    A : float
        Ion mass number (default: 1 for hydrogen)
    ln_Lambda : float
        Coulomb logarithm (default: 15)
    
    Returns:
    --------
    tau_i : float
        Collision time in seconds
    """
    # Convert temperature to ergs
    T_erg = T_eV * EV_TO_ERG
    
    # Ion mass
    m_i = A * M_PROTON
    
    # Calculate collision time using the formula from image 1
    # τ_i = 3√(m_i(kT_i)^3/2) / (4√(πn)λe^4)
    # Simplified form: τ_i = 2.09 × 10^7 × T^(3/2) / (nλ) × μ^(1/2) sec
    
    # Using the numerical coefficient form
    mu = A  # atomic mass number
    tau_i = 2.09e7 * (T_eV**(3/2)) / (n * ln_Lambda) * np.sqrt(mu)
    
    return tau_i

def calculate_cyclotron_frequency(B, Z=1, A=1):
    """
    Calculate ion cyclotron frequency ω_ci
    
    Parameters:
    -----------
    B : float
        Magnetic field in Gauss
    Z : int
        Ion charge number (default: 1)
    A : float
        Ion mass number (default: 1)
    
    Returns:
    --------
    omega_ci : float
        Cyclotron frequency in rad/s
    """
    # Ion mass
    m_i = A * M_PROTON
    
    # Calculate cyclotron frequency using formula from image 2
    # ω_ci = ZeB/(m_i c) = 9.58 × 10^3 × Z × μ^(-1) × B rad/sec
    
    mu = A  # atomic mass number
    omega_ci = 9.58e3 * Z / mu * B
    
    return omega_ci

def calculate_braginskii_viscosities(T_eV, n, B, Z=1, A=1, ln_Lambda=15):
    """
    Calculate all Braginskii ion viscosity coefficients
    
    Parameters:
    -----------
    T_eV : float
        Temperature in eV
    n : float
        Number density in cm^-3
    B : float
        Magnetic field in Gauss
    Z : int
        Ion charge number (default: 1 for hydrogen)
    A : float
        Ion mass number (default: 1 for hydrogen)
    ln_Lambda : float
        Coulomb logarithm (default: 15)
    
    Returns:
    --------
    dict : Dictionary containing all viscosity coefficients in g/(cm·s)
    """
    # Calculate collision time and cyclotron frequency
    tau_i = calculate_collision_time(T_eV, n, Z, A, ln_Lambda)
    omega_ci = calculate_cyclotron_frequency(B, Z, A)
    
    # Convert temperature to ergs for viscosity calculation
    T_erg = T_eV * EV_TO_ERG
    
    # Calculate viscosities using formulas from image 3
    # η_0^i = 0.96 n k T_i τ_i
    eta_0 = 0.96 * n * T_erg * tau_i
    
    # η_1^i = 3nkT_i / (10 ω_ci^2 τ_i)
    eta_1 = (3 * n * T_erg) / (10 * omega_ci**2 * tau_i)
    
    # η_2^i = 6nkT_i / (5 ω_ci^2 τ_i)
    eta_2 = (6 * n * T_erg) / (5 * omega_ci**2 * tau_i)
    
    # η_3^i = nkT_i / (2 ω_ci)
    eta_3 = (n * T_erg) / (2 * omega_ci)
    
    # η_4^i = nkT_i / ω_ci
    eta_4 = (n * T_erg) / omega_ci
    
    return {
        'eta_0': eta_0,
        'eta_1': eta_1,
        'eta_2': eta_2,
        'eta_3': eta_3,
        'eta_4': eta_4,
        'tau_i': tau_i,
        'omega_ci': omega_ci,
        'omega_ci_tau_i': omega_ci * tau_i
    }

def print_results(results, T_eV, n, B, Z=1, A=1):
    """
    Print formatted results
    """
    print("\n" + "="*60)
    print("BRAGINSKII ION VISCOSITY COEFFICIENTS")
    print("="*60)
    print(f"\nInput Parameters:")
    print(f"  Temperature:       T = {T_eV:.2e} eV")
    print(f"  Density:           n = {n:.2e} cm⁻³")
    print(f"  Magnetic field:    B = {B:.2e} G")
    print(f"  Ion charge:        Z = {Z}")
    print(f"  Ion mass number:   A = {A}")
    
    print(f"\nDerived Quantities:")
    print(f"  Collision time:    τᵢ = {results['tau_i']:.3e} s")
    print(f"  Cyclotron freq:    ωcᵢ = {results['omega_ci']:.3e} rad/s")
    print(f"  Hall parameter:    ωcᵢτᵢ = {results['omega_ci_tau_i']:.3e}")
    
    print(f"\nViscosity Coefficients (g/(cm·s)):")
    print(f"  η₀ = {results['eta_0']:.3e}  (parallel)")
    print(f"  η₁ = {results['eta_1']:.3e}  (perpendicular 1)")
    print(f"  η₂ = {results['eta_2']:.3e}  (perpendicular 2)")
    print(f"  η₃ = {results['eta_3']:.3e}  (gyroviscous 1)")
    print(f"  η₄ = {results['eta_4']:.3e}  (gyroviscous 2)")
    print("="*60 + "\n")

# Example usage
if __name__ == "__main__":
    # Example: Hydrogen plasma
    T_eV = 4573.63     # Temperature in eV
    n = 1.01e14        # Number density in cm^-3
    B = 25700.0        # Magnetic field in Gauss
    Z = 1              # Hydrogen ion charge
    A = 1              # Hydrogen mass number
    ln_Lambda = 16.5   # Coulomb logarithm
    
    # Calculate viscosities
    results = calculate_braginskii_viscosities(T_eV, n, B, Z, A, ln_Lambda)
    
    # Print results
    print_results(results, T_eV, n, B, Z, A)
    
    # You can also access individual components:
    print("Individual access example:")
    print(f"η₀ = {results['eta_0']:.3e} g/(cm·s)")
    print(f"ωcᵢτᵢ = {results['omega_ci_tau_i']:.3f}")
