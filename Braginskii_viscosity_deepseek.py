#! /usr/bin/env python3

import numpy as np

# Physical constants for SI conversions
M_PROTON_SI = 1.673e-27  # kg

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
    
    # Convert viscosities from CGS [g/(cm·s)] to SI [kg/(m·s)]
    # Conversion factor: 1 g/(cm·s) = 0.1 kg/(m·s)
    CGS_TO_SI = 0.1
    eta0_SI = eta0 * CGS_TO_SI
    eta1_SI = eta1 * CGS_TO_SI
    eta2_SI = eta2 * CGS_TO_SI
    eta3_SI = eta3 * CGS_TO_SI
    eta4_SI = eta4 * CGS_TO_SI
    
    # Calculate mass density for kinematic viscosity
    # ρ = n * m_i
    # Convert n from cm^-3 to m^-3: multiply by 10^6
    n_SI = n_cm3 * 1e6  # m^-3
    m_i_SI = mu * M_PROTON_SI  # kg
    rho_SI = n_SI * m_i_SI  # kg/m^3
    
    # Calculate kinematic viscosities ν = η / ρ [m^2/s]
    nu0 = eta0_SI / rho_SI
    nu1 = eta1_SI / rho_SI
    nu2 = eta2_SI / rho_SI
    nu3 = eta3_SI / rho_SI
    nu4 = eta4_SI / rho_SI

    return {
        'eta0': eta0,
        'eta1': eta1,
        'eta2': eta2,
        'eta3': eta3,
        'eta4': eta4,
        'eta0_SI': eta0_SI,
        'eta1_SI': eta1_SI,
        'eta2_SI': eta2_SI,
        'eta3_SI': eta3_SI,
        'eta4_SI': eta4_SI,
        'nu0': nu0,
        'nu1': nu1,
        'nu2': nu2,
        'nu3': nu3,
        'nu4': nu4,
        'rho_SI': rho_SI,
        'tau_i': tau_i,
        'omega_ci': omega_ci,
        'omega_ci_tau_i': omega_ci * tau_i
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
    
    print("="*60)
    print("BRAGINSKII ION VISCOSITY COEFFICIENTS")
    print("="*60)
    print(f"\nInput Parameters:")
    print(f"  Temperature:       T = {T_eV:.2e} eV")
    print(f"  Density:           n = {n_cm3:.2e} cm⁻³")
    print(f"  Magnetic field:    B = {B_gauss:.2e} G")
    print(f"  Ion charge:        Z = {Z}")
    print(f"  Ion mass number:   μ = {mu}")
    print(f"  Coulomb logarithm: Λ = {Lambda}")
    
    print(f"\nDerived Quantities:")
    print(f"  Collision time:    τᵢ = {viscosities['tau_i']:.3e} s")
    print(f"  Cyclotron freq:    ωcᵢ = {viscosities['omega_ci']:.3e} rad/s")
    print(f"  Hall parameter:    ωcᵢτᵢ = {viscosities['omega_ci_tau_i']:.3e}")
    
    print(f"\nViscosity Coefficients (CGS) [g/(cm·s)]:")
    print(f"  η₀ = {viscosities['eta0']:.3e}  (parallel)")
    print(f"  η₁ = {viscosities['eta1']:.3e}  (perpendicular 1)")
    print(f"  η₂ = {viscosities['eta2']:.3e}  (perpendicular 2)")
    print(f"  η₃ = {viscosities['eta3']:.3e}  (gyroviscous 1)")
    print(f"  η₄ = {viscosities['eta4']:.3e}  (gyroviscous 2)")
    
    print(f"\nViscosity Coefficients (SI) [kg/(m·s)]:")
    print(f"  η₀ = {viscosities['eta0_SI']:.3e}  (parallel)")
    print(f"  η₁ = {viscosities['eta1_SI']:.3e}  (perpendicular 1)")
    print(f"  η₂ = {viscosities['eta2_SI']:.3e}  (perpendicular 2)")
    print(f"  η₃ = {viscosities['eta3_SI']:.3e}  (gyroviscous 1)")
    print(f"  η₄ = {viscosities['eta4_SI']:.3e}  (gyroviscous 2)")
    
    print(f"\nMass Density:")
    print(f"  ρ = {viscosities['rho_SI']:.3e} kg/m³")
    
    print(f"\nKinematic Viscosity Coefficients [m²/s]:")
    print(f"  ν₀ = {viscosities['nu0']:.3e}  (parallel)")
    print(f"  ν₁ = {viscosities['nu1']:.3e}  (perpendicular 1)")
    print(f"  ν₂ = {viscosities['nu2']:.3e}  (perpendicular 2)")
    print(f"  ν₃ = {viscosities['nu3']:.3e}  (gyroviscous 1)")
    print(f"  ν₄ = {viscosities['nu4']:.3e}  (gyroviscous 2)")
    print("="*60)