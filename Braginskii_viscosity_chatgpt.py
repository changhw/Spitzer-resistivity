#! /usr/bin/env python3

import numpy as np

# =========================
# Physical constants (cgs)
# =========================
e_cgs = 4.80320427e-10       # statcoulomb
c = 2.99792458e10            # cm/s
m_p = 1.67262192369e-24      # g
erg_per_eV = 1.602176634e-12 # erg

# =========================
# Physical constants (SI)
# =========================
M_PROTON_SI = 1.67262192369e-27  # kg

# =========================
# Braginskii ion viscosities
# =========================
def braginskii_ion_viscosity(
        n,          # density [cm^-3]
        Ti_eV,      # ion temperature [eV]
        B,          # magnetic field [Gauss]
        lnLambda,   # Coulomb logarithm
        Z=1,        # ion charge
        mu=1.0      # ion mass ratio m_i/m_p
    ):
    """
    Returns:
        Dictionary with viscosity coefficients in CGS, SI units,
        and kinematic viscosities
    """

    # Ion mass
    m_i = mu * m_p

    # Convert T to erg (k_B T)
    kT = Ti_eV * erg_per_eV

    # Ion collision time (given formula)
    tau_i = 2.09e7 * (Ti_eV**1.5) * np.sqrt(mu) / (n * lnLambda)

    # Ion cyclotron frequency (cgs)
    omega_ci = Z * e_cgs * B / (m_i * c)

    # Common factor
    nkT = n * kT

    # Braginskii viscosities
    eta0 = 0.96 * nkT * tau_i
    eta1 = 3.0 * nkT / (10.0 * omega_ci**2 * tau_i)
    eta2 = 6.0 * nkT / (5.0 * omega_ci**2 * tau_i)
    eta3 = nkT / (2.0 * omega_ci)
    eta4 = nkT / omega_ci
    
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
    n_SI = n * 1e6  # m^-3
    m_i_SI = mu * M_PROTON_SI  # kg
    rho_SI = n_SI * m_i_SI  # kg/m^3
    
    # Calculate kinematic viscosities ν = η / ρ [m^2/s]
    nu0 = eta0_SI / rho_SI
    nu1 = eta1_SI / rho_SI
    nu2 = eta2_SI / rho_SI
    nu3 = eta3_SI / rho_SI
    nu4 = eta4_SI / rho_SI

    return {
        'tau_i': tau_i,
        'omega_ci': omega_ci,
        'omega_ci_tau_i': omega_ci * tau_i,
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
        'rho_SI': rho_SI
    }


# =========================
# Example usage
# =========================
if __name__ == "__main__":

    # Typical tokamak core parameters (cgs)
    n = 1.01e14          # cm^-3
    Ti = 4573.63         # eV
    B = 25700            # Gauss
    lnLambda = 16.5
    Z = 1
    mu = 1.0          # hydrogen

    results = braginskii_ion_viscosity(n, Ti, B, lnLambda, Z, mu)

    print("="*60)
    print("BRAGINSKII ION VISCOSITY COEFFICIENTS")
    print("="*60)
    print(f"\nInput Parameters:")
    print(f"  Temperature:       T = {Ti:.2e} eV")
    print(f"  Density:           n = {n:.2e} cm⁻³")
    print(f"  Magnetic field:    B = {B:.2e} G")
    print(f"  Ion charge:        Z = {Z}")
    print(f"  Ion mass number:   μ = {mu}")
    print(f"  Coulomb logarithm: Λ = {lnLambda}")
    
    print(f"\nDerived Quantities:")
    print(f"  Collision time:    τᵢ = {results['tau_i']:.3e} s")
    print(f"  Cyclotron freq:    ωcᵢ = {results['omega_ci']:.3e} rad/s")
    print(f"  Hall parameter:    ωcᵢτᵢ = {results['omega_ci_tau_i']:.3e}")
    
    print(f"\nViscosity Coefficients (CGS) [g/(cm·s)]:")
    print(f"  η₀ = {results['eta0']:.3e}  (parallel)")
    print(f"  η₁ = {results['eta1']:.3e}  (perpendicular 1)")
    print(f"  η₂ = {results['eta2']:.3e}  (perpendicular 2)")
    print(f"  η₃ = {results['eta3']:.3e}  (gyroviscous 1)")
    print(f"  η₄ = {results['eta4']:.3e}  (gyroviscous 2)")
    
    print(f"\nViscosity Coefficients (SI) [kg/(m·s)]:")
    print(f"  η₀ = {results['eta0_SI']:.3e}  (parallel)")
    print(f"  η₁ = {results['eta1_SI']:.3e}  (perpendicular 1)")
    print(f"  η₂ = {results['eta2_SI']:.3e}  (perpendicular 2)")
    print(f"  η₃ = {results['eta3_SI']:.3e}  (gyroviscous 1)")
    print(f"  η₄ = {results['eta4_SI']:.3e}  (gyroviscous 2)")
    
    print(f"\nMass Density:")
    print(f"  ρ = {results['rho_SI']:.3e} kg/m³")
    
    print(f"\nKinematic Viscosity Coefficients [m²/s]:")
    print(f"  ν₀ = {results['nu0']:.3e}  (parallel)")
    print(f"  ν₁ = {results['nu1']:.3e}  (perpendicular 1)")
    print(f"  ν₂ = {results['nu2']:.3e}  (perpendicular 2)")
    print(f"  ν₃ = {results['nu3']:.3e}  (gyroviscous 1)")
    print(f"  ν₄ = {results['nu4']:.3e}  (gyroviscous 2)")
    print("="*60)
