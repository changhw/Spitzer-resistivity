#! /usr/bin/env python3

import numpy as np

def braginskii_ion_viscosities(Z, mu, T_eV, n_cm3, Lambda, B_gauss):
    """
    Calculate Braginskii's ion viscosity coefficients in cgs units.
    
    Parameters
    ----------
    Z : float or int
        Ion charge number.
    mu : float
        Ion mass in proton masses (mass number).
    T_eV : float
        Ion temperature in electronvolts.
    n_cm3 : float
        Ion density in cm⁻³.
    Lambda : float
        Coulomb logarithm (ln Λ).
    B_gauss : float
        Magnetic field strength in Gauss.
    
    Returns
    -------
    dict
        Dictionary with keys 'eta0', 'eta1', 'eta2', 'eta3', 'eta4'
        containing the viscosities in g/(cm·s) (poise).
    """
    # Convert temperature to erg (kT in erg)
    # 1 eV = 1.602e-12 erg (standard value)
    eV_to_erg = 1.602e-12
    kT_erg = T_eV * eV_to_erg

    # Ion collision time (seconds) – Braginskii's numerical formula
    tau_i = 2.09e7 * (T_eV ** 1.5) / (n_cm3 * Lambda) * np.sqrt(mu)

    # Ion cyclotron frequency (rad/sec) – Braginskii's numerical formula
    omega_ci = 9.58e3 * Z / mu * B_gauss

    # Viscosity coefficients
    eta0 = 0.96 * n_cm3 * kT_erg * tau_i
    eta1 = 3.0 * n_cm3 * kT_erg / (10.0 * omega_ci**2 * tau_i)
    eta2 = 6.0 * n_cm3 * kT_erg / (5.0 * omega_ci**2 * tau_i)
    eta3 = n_cm3 * kT_erg / (2.0 * omega_ci)
    eta4 = n_cm3 * kT_erg / omega_ci

    return {
        'eta0': eta0,
        'eta1': eta1,
        'eta2': eta2,
        'eta3': eta3,
        'eta4': eta4
    }

# ---------------------------------------------------------------------
# Example usage
if __name__ == "__main__":
    # Typical parameters for a hydrogen plasma (Z=1, mu=1)
    Z = 1
    mu = 1
    T_eV = 4573.63        # 4573.63 eV
    n_cm3 = 1.01e14       # 1.01 × 10^14 cm⁻³
    Lambda = 16.5         # Coulomb logarithm
    B_gauss = 25700.0     # 25.7 kG = 2.57 T

    viscosities = braginskii_ion_viscosities(Z, mu, T_eV, n_cm3, Lambda, B_gauss)
    
    print("Braginskii Ion Viscosities (poise):")
    for key, val in viscosities.items():
        print(f"  {key} = {val:.4e}")