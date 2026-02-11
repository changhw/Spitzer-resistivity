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
        tau_i  [s]
        omega_ci [1/s]
        eta0, eta1, eta2, eta3, eta4  [g / (cm s)]
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

    return tau_i, omega_ci, eta0, eta1, eta2, eta3, eta4


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

    names = ["tau_i [s]",
             "omega_ci [1/s]",
             "eta0 [g/(cm s)]",
             "eta1 [g/(cm s)]",
             "eta2 [g/(cm s)]",
             "eta3 [g/(cm s)]",
             "eta4 [g/(cm s)]"]

    for name, val in zip(names, results):
        print(f"{name:15s} = {val:.3e}")
