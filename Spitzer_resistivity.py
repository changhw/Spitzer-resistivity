#! /usr/bin/env python3
#
# Purpose: Calculate resistivity, viscosity,
# magnetic Prandtl number, Hartmann number, etc.
# Date: 2023-09
# Author: Haowei Zhang, IPP Garching
# Please modify the parameters in # plasma parameters

# References:
# https://farside.ph.utexas.edu/teaching/plasma/Plasma/node41.html
# https://descanso.jpl.nasa.gov/SciTechBook/series1/Goebel_03_Chap3_plasphys.pdf
# https://doi.org/10.1088/1742-6596/2397/1/012010
# https://farside.ph.utexas.edu/teaching/plasma/Plasmahtml/Plasmahtml.html
# http://hsxie.me/fusionbook/fusion230103.pdf#page=59.32
# https://library.psfc.mit.edu/catalog/online_pubs/NRL_FORMULARY_19.pdf

import numpy as np
import math
import matplotlib.pyplot as plt

# some constants
pi = np.pi
epsilon0 = 8.854188e-12  # F/m
mu0 = 4 * pi * 1e-7  # H/m
kB = 1.380649e-23  # J/K
e = 1.602177e-19  # C or J/eV
gamma = 5.0 / 3.0

# plasma parameters
B0_axis = 2.57       # T
central_mass = 1.0   # hydrogen
Zeff = central_mass
Mie = 1836 * Zeff    # mass ratio
me = 9.109384e-31    # kg
mi = me * Mie        # kg
Ne = 1.01 * 1.0e+20  # m-3
Te = 5033.16         # eV (for base case = 4.92e+04 * 1.023e-01)
Ti = 4573.632        # eV (for base case = 4.92e+04 * 9.296e-02)

# JOREK normalization factors
rho0 = Ne * mi  # kg/m^3
sqrt_mu0_rho0 = np.sqrt(mu0 * rho0)
sqrt_mu0_over_rho0 = np.sqrt(mu0 / rho0)

L0_norm = 1  # m
B0_norm = 1  # T
VA_norm = B0_norm / (mu0 * rho0) ** 0.5
P0_norm = B0_norm ** 2 / mu0

T_norm = 1 / e / mu0 / Ne
T_norm2 = 1 / kB / mu0 / Ne

OUT_PUT_HARTMANN_PRANDTL_ONLY = False # output all parameters if set it to False

if __name__ == '__main__':

    clog=np.log(4.0*pi*(epsilon0*e*Te)**1.5/e**3/(Ne)**(0.5)) # Coulomb logarithm # log(4 * pi * ne * lambda_D), lambda_D is the Debye length, # https://www.zgbk.com/ecph/words?SiteID=1&ID=104679&Type=bkzyb
    # clog_Huba = 24.0 - np.log(Ne**0.5/Te) # valid for Ti*me/mi < 10 Z^2 eV < Te, from Huba-2016-NRL Page 34
    # clog_ee_Huba = 23.5 - np.log(Ne ** 0.5 * Te ** -1.25) - (1.e-5 + (np.log(Te) - 2)**2 / 16.0) ** 0.5 # from Huba-2016-NRL Page 34
    eta_spz=0.51 * (8*pi*e**2*me**0.5)/((4*pi*epsilon0)**2*3*(2*pi)**0.5*(e*Te)**1.5)*clog*Zeff
    eta_spz_vivenzi = 0.06 / pi**1.5 * me**0.5 * e**2 / epsilon0**2 * Zeff * clog / (e*Te)**1.5 # based on N Vivenzi et al 2022 J. Phys.: Conf. Ser. 2397 012010
    lnA = 14.9 - 0.5*math.log( Ne/1.e20 ) + math.log( Te/1.e3 )
    eta_spz_in_Jorek=1.65e-9/((Te/1.e3)**1.5)*lnA*Zeff

    # calculate the ion-electron energy transfer rate
    # from jorek, model 711
    lambda_e_bg = 23. - np.log((Ne * 1.e-6) ** 0.5 * Te ** (-1.5)) # Assuming bg_charge is 1!
    nu_e_bg = 1.8e-19 * (1.e6 * me * mi * central_mass) ** 0.5 \
              * (1.e14 * Ne / 1.e20) * lambda_e_bg   \
              / (1.e3 * (me * Ti / T_norm +Te / T_norm * mi * central_mass) \
              / (e * mu0 * Ne)) ** 1.5
    dTi_e_norm = nu_e_bg / VA_norm * (Te - Ti) / T_norm * Ne / 1e20
    dTi_e = nu_e_bg * (Te - Ti) * e * Ne
    dTi_e2 = dTi_e_norm * (T_norm2 * VA_norm * 1e20) * kB

    # from NRL 2019, mass in (g), density in (cm^-3), T in (eV)
    nu_e_NRL = 1.8e-19 * np.sqrt(mi*me * 1e6) * (Ne / 1e6) * lambda_e_bg / (mi * 1e3 * Ti + me * 1e3 * Te) ** 1.5
    nu_e_NRLapprox = 3.2483624174976183e-9 * lambda_e_bg / 1 / (Te)**1.5 * (Ne/1e6)
    dTi_e_norm_NRL = nu_e_NRL / VA_norm * (Te - Ti) / T_norm * Ne / 1e20
    dTi_e_NRL = nu_e_NRL * (Te - Ti) * e * Ne
    dTi_e2_NRL = dTi_e_norm_NRL * (T_norm2 * VA_norm * 1e20) * kB

    # from HS Xie: http://hsxie.me/fusionbook/fusion230103.pdf#page=59.32
    lambda_xie = 12 * np.pi / (Ne ** 0.5) * (epsilon0 * e * Te / e ** 2) ** 1.5
    log_lambda_xie = np.log(lambda_xie)
    tau_ei_xie = mi/2/me*1.09e16 * (Te/1e3)**1.5/Ne/log_lambda_xie
    v_ei_xie = 1/tau_ei_xie

    # from Jincai Ren
    ve = np.sqrt(e * Te * 2 / me)
    vi = np.sqrt(e * Ti * 2 / mi)
    nu_ei_ren = Ne * e ** 4 * lambda_e_bg / (2 * np.pi * epsilon0**2 * me ** 2 * ve ** 3) * (1 - 1 / (2 * (ve/vi)**2))
    nu_ie_ren = Ne * e ** 4 * lambda_e_bg / (2 * np.pi * epsilon0**2 * mi ** 2 * vi ** 3) * 4 / (3*np.pi**0.5) * (vi/ve)

    # from Richard Fitzpatrick
    # https://farside.ph.utexas.edu/teaching/plasma/Plasmahtml/Plasmahtml.html
    tau_ei_Fitz = 6 * np.sqrt(2) * np.pi ** 1.5 * epsilon0 **2 * me ** 0.5 * (e * Te) ** 1.5 / (lambda_e_bg * e**4 * Ne)
    tau_ie_Fitz = 6 * np.sqrt(2) * np.pi ** 1.5 * epsilon0 **2 * mi        * (e * Te) ** 1.5 / (lambda_e_bg * e**4 * Ne * me**0.5)
    nu_ei_Fitz = 1 / tau_ei_Fitz
    nu_ie_Fitz = 1 / tau_ie_Fitz

    # from Haowei Zhang Thesis
    nu_ie_zhang = np.sqrt(2)/(3*np.sqrt(np.pi))*Ne*(e**2/4/np.pi/epsilon0)**2 * 4*np.pi/np.sqrt(me * (e*Te)**3) * lambda_e_bg

    # eta_spz_simple = me / Ne / e**2 * 2.9e-12 * Ne * clog / (Te)**1.5 # https://descanso.jpl.nasa.gov/SciTechBook/series1/Goebel_03_Chap3_plasphys.pdf
    # eta_spz_huba = 1.03e-2 * clog / (Te) ** 1.5

    if not OUT_PUT_HARTMANN_PRANDTL_ONLY:
        # print(f'T_norm                   = {"{:.02e}".format(T_norm)} eV')
        # print(f'dTi_e_norm (JOREK units) = {"{:.02e}".format(dTi_e_norm)}')
        print(f'lambda_e_bg              = {"{:.02e}".format(lambda_e_bg)}')
        print(f'log_lambda_xie           = {"{:.02e}".format(log_lambda_xie)}')
        print(f'Te-Ti                    = {"{:.02e}".format(Te-Ti)} eV')
        print(f'nu_ei_bg                 = {"{:.02e}".format(nu_e_bg)} s^-1')
        print(f'nu_e_NRL                 = {"{:.02e}".format(nu_e_NRL)} s^-1')
        print(f'nu_e_NRLapprox           = {"{:.02e}".format(nu_e_NRLapprox)} s^-1')
        print(f'v_ei_xie                   = {"{:.02e}".format(v_ei_xie)} s^-1')
        print(f'nu_ei_ren                  = {"{:.02e}".format(nu_ei_ren)} s^-1')
        # print(f'nu_ie_ren                  = {"{:.02e}".format(nu_ie_ren)} s^-1')
        # print(f'nu_ei_Fitz                 = {"{:.02e}".format(nu_ei_Fitz)} s^-1')
        # print(f'nu_ie_Fitz                 = {"{:.02e}".format(nu_ie_Fitz)} s^-1')
        # print(f'nu_ie_zhang                = {"{:.02e}".format(nu_ie_zhang)} s^-1')

        print(f'dTi_e (e -> i)           = {"{:.02e}".format(dTi_e)} W/m^3')
        print(f'dTi_e2                   = {"{:.02e}".format(dTi_e2)} W/m^3')

        print(f'dTi_e_NRL (e -> i)       = {"{:.02e}".format(dTi_e_NRL)} W/m^3')
        print(f'dTi_e2_NRL               = {"{:.02e}".format(dTi_e2_NRL)} W/m^3')
        # print(f'eta_spz_simple           = {"{:02e}".format(eta_spz_simple)} Ohm m')
        # print(f'eta_spz_huba             = {"{:02e}".format(eta_spz_huba)} Ohm m')

    S_lundquist_spz=mu0*L0_norm*VA_norm/eta_spz
    eta_spz_norm=eta_spz/sqrt_mu0_over_rho0
    eta_spz_norm_Jorek=eta_spz_in_Jorek/sqrt_mu0_over_rho0 # same as the JOREK code

    if not OUT_PUT_HARTMANN_PRANDTL_ONLY:
        print('--------------------------------')
        print(f'clog                = {"{:.02e}".format(clog)}')
        # print(f'clog_Huba           = {"{:.02e}".format(clog_Huba)}')
        # print(f'clog_ee_Huba        = {"{:.02e}".format(clog_ee_Huba)}')
        print(f'lnA (in Jorek)      = {"{:.02e}".format(lnA)}')
        print('--------------------------------')
        print(f'eta_spz             = {"{:.02e}".format(eta_spz)} Ohm m')
        print(f'eta_spz_vivenzi     = {"{:.02e}".format(eta_spz_vivenzi)} Ohm m')
        print(f'eta_spz_in_Jorek    = {"{:.02e}".format(eta_spz_in_Jorek)} Ohm m')
        print(f'S_lundquist_spz     = {"{:.02e}".format(S_lundquist_spz)}')
        print(f'*eta_spz_norm       = {"{:.02e}".format(eta_spz_norm)}') # = 1.0/S_lundquist
        print(f'*eta_spz_norm_Jorek = {"{:.02e}".format(eta_spz_norm_Jorek)}') # = 1.0/S_lundquist

    # calculate the kinematic viscosities based on N Vivenzi et al 2022 J. Phys.: Conf. Ser. 2397 012010
    gamma_viscous = 1
    visco_para = 34.6 * pi**1.5 * epsilon0**2 / (e**4 * mi ** 0.5) * (e * Ti) ** 2.5 / (Zeff**4 * clog * gamma_viscous**0.5 * Ne)
    visco_perp = 1.0 / (8 * pi**1.5) * mi**0.5 * e**2 / epsilon0**2 * gamma_viscous**0.5 * Ne * clog / (e*Ti)**0.5 / B0_axis**2
    visco_gyro = 1.25 / e * (e * Ti) / Zeff / B0_axis
    aT = 1 #m
    aB = 1.7 #m
    visco_ITG  = 1.08e-4 * gamma_viscous**0.5 * (1 * Te) * (1 * Ti)**0.5 / (Zeff * aT**0.75 * aB**0.25 * B0_axis**2)
    gamma_e = 1
    gamma_i = 3
    Lc      = 1.7 # m
    visco_Finn = ((gamma_e * Zeff * e * Te + gamma_i * e * Ti) / mi)**0.5 * Lc * (1e-3)**2
    viscosity_in_Jorek = visco_perp + visco_ITG + visco_Finn


    # χ⊥ perpendicular heat conductivity, typically 1m2/s; lower in pedestal by one order of magnitude
    # χ|| parallel heat conductivity can be approximated by the Spitzer-Haerm formula
    Te_keV = Te/1000.0
    Ti_keV = Ti/1000.0
    chi_parallel_SH = 3.6 * 1.e29 * Te_keV ** 2.5  / Ne # m^2/s
    zk_parallel_SH = chi_parallel_SH * rho0  # K_si = rho_si * chi_si   kg m^-1 s^-1
    reduction_factor = 30.0

    # At high temperature, SH is typically overestimating the parallel conductivity (heat flux limit, see literature)
    # e.g., reducing by a factor ≈ 30, compared to SH seemed to make sense for tearing modes in TEXTOR and ASDEX Upgrade (PhD M. Hoelzl)
    ZK_e_par_SpitzerHaerm = 5.5789e+0 * mi /(me*lnA) * Te_keV**(2.5e+0) * (gamma-1.e0) * sqrt_mu0_over_rho0        # Same as the JORKE code
    ZK_i_par_SpitzerHaerm = 5.8410e+2 * math.sqrt(central_mass/2.e+0)/lnA * Ti_keV**(2.5e+0) * (gamma-1.e0) * sqrt_mu0_over_rho0  # Same as the JORKE code

    if not OUT_PUT_HARTMANN_PRANDTL_ONLY:
        print('--------------------------------')
        # print('overestimated? zk_parallel_SH_e = ', '{:02e}'.format(zk_parallel_SH))
        # print('reduced, zk_parallel_SH_e = ', '{:02e}'.format(zk_parallel_SH / 30))
        print(f'overestimated, zk_parallel_SH_e (Jorek unit)       = {"{:.02e}".format(zk_parallel_SH * (gamma - 1.e0) * sqrt_mu0_over_rho0)}')
        print(f'*overestimated, ZK_e_par_SpitzerHaerm (Jorek unit) = {"{:.02e}".format(ZK_e_par_SpitzerHaerm)}')
        print(f'*overestimated, ZK_i_par_SpitzerHaerm (Jorek unit) = {"{:.02e}".format(ZK_i_par_SpitzerHaerm)}')
        print(f'*reduced, ZK_e_par_SpitzerHaerm (Jorek unit) = {"{:.02e}".format(ZK_e_par_SpitzerHaerm / reduction_factor)}')
        print(f'*reduced, ZK_i_par_SpitzerHaerm (Jorek unit) = {"{:.02e}".format(ZK_i_par_SpitzerHaerm / reduction_factor)}')

    Hartmann = np.sqrt(mu0/(eta_spz_in_Jorek*viscosity_in_Jorek)) * L0_norm * VA_norm * B0_axis
    magnetic_Prandtl = mu0 * viscosity_in_Jorek / eta_spz_in_Jorek
    print('--------------different viscosity models from N Vivenzi et al 2022 J. Phys.: Conf. Ser. 2397 012010------------------')
    print(f'1. Braginskii visco_para          = {"{:.02e}".format(visco_para)} m^2/s')
    print(f'2. Braginskii visco_perp          = {"{:.02e}".format(visco_perp)} m^2/s')
    print(f'3. Braginskii visco_gyro          = {"{:.02e}".format(visco_gyro)} m^2/s')
    print(f'4. visco_ITG                      = {"{:.02e}".format(visco_ITG) } m^2/s')
    print(f'5. visco_Finn                     = {"{:.02e}".format(visco_Finn)} m^2/s')

    print('--------------------------------')
    print(f'*estimated viscosity in Jorek (models 2-5) = {"{:.02e}".format(viscosity_in_Jorek)} m^2/s')
    print(f'*estimated eta Spitzer in Jorek            = {"{:.02e}".format(eta_spz_in_Jorek)} Ohm m')

    print(f'*magnetic Prandtl number                   = {"{:.02e}".format(magnetic_Prandtl)}')
    print(f'*Hartmann number (L = {L0_norm} m)                 = {"{:.02e}".format(Hartmann)}')


    #####################################################
    #
    #      parameter ranges of flux pumping
    #
    #####################################################


    # Set global font properties using rcParams
    plt.rcParams.update({
        'axes.formatter.limits': (-3, 3),
        'axes.formatter.useoffset': True,  # Use scientific notation
        'font.size': 18,                # Global font size
        'lines.linewidth': 3.0,         # Global line width
        'axes.linewidth': 3.0,           # Set line width for both x and y axes
        # 'font.family': 'serif',         # Use serif fonts, which includes Times New Roman
        # 'font.serif': ['Times New Roman'],  # Specifically set Times New Roman
        # 'font.weight': 'bold'           # Font weight for all text elements
        'legend.fontsize': 18,          # Font size for the legend
        # 'axes.labelsize': 14,           # Font size for axis labels
        'axes.titlesize': 18,           # Font size for title
        # 'xtick.labelsize': 12,          # Font size for x-tick labels
        # 'ytick.labelsize': 12,          # Font size for y-tick labels
    })

    magnetic_Prandtl_Finn = mu0 * visco_Finn / eta_spz_in_Jorek
    Hartmann_Finn = np.sqrt(mu0/(eta_spz_in_Jorek*visco_Finn)) * L0_norm * VA_norm * B0_axis
    print(f'*magnetic Prandtl number Finn              = {"{:.02e}".format(magnetic_Prandtl_Finn)}')
    print(f'*Hartmann number Finn (L = {L0_norm} m)            = {"{:.02e}".format(Hartmann_Finn)}')


    Ne_min = 1e19
    Ne_max = 1e21   # m^{-3}
    Te_min = 500
    Te_max = 50000  # ev

    # Ne_min = 0.1* Ne
    # Ne_max = 10 * Ne
    # Te_min = 0.1* Te
    # Te_max = 10 * Te

    Te_1d = np.logspace(np.log10(Te_min), np.log10(Te_max), 1000)
    Ne_1d = np.logspace(np.log10(Ne_min), np.log10(Ne_max), 1000)

    Te_2d, Ne_2d = np.meshgrid(Te_1d, Ne_1d)
    Ti_2d = Te_2d

    H_c = 3.e7
    beta_Nc = 2.0
    
    beta_N_2d = Te_2d / Te * Ne_2d / Ne * 3.0

    clog_2d=np.log(4.0*pi*(epsilon0*e*Te_2d)**1.5/e**3/(Ne_2d)**(0.5))

    eta_2d = 1.65e-9/((Te_2d/1.e3)**1.5)*clog_2d*Zeff
    visco_Finn_2d = ((gamma_e * Zeff * e * Te_2d + gamma_i * e * Ti_2d) / mi)**0.5 * Lc * (1e-3)**2
    visco_ITG_2d  = 1.08e-4 * gamma_viscous**0.5 * (1 * Te_2d) * (1 * Ti_2d)**0.5 / (Zeff * aT**0.75 * aB**0.25 * B0_axis**2)

    Hartmann_Finn_2d = np.sqrt(mu0/( eta_2d * visco_Finn_2d )) * L0_norm * B0_axis * B0_norm / (mu0 * Ne_2d * mi) ** 0.5

    Hartmann_ITG_2d  = np.sqrt(mu0/( eta_2d * visco_ITG_2d )) * L0_norm * B0_axis * B0_norm / (mu0 * Ne_2d * mi) ** 0.5

    Hartmann_total_2d = np.sqrt(mu0/(eta_2d * (visco_Finn_2d + visco_ITG_2d) )) * L0_norm * B0_axis * B0_norm / (mu0 * Ne_2d * mi) ** 0.5

    fig, ax = plt.subplots(figsize=(8, 6))

    mask = (
    (beta_N_2d > beta_Nc) &
    (Hartmann_ITG_2d > H_c) &
    (Hartmann_Finn_2d > H_c) & 
    (Hartmann_total_2d > H_c)
    )

    # filled contour
    cf = ax.contourf(Te_2d/1e3, Ne_2d/1e20, np.log10(Hartmann_total_2d), 20, cmap='bwr')

    # Overlay hatched region where mask == 1
    cs_hatch = ax.contourf(
        Te_2d / 1e3,
        Ne_2d / 1e20,
        mask,
        levels=[0.5, 1.5],
        colors='none',          # <-- no color fill
        hatches=['..'],        # <-- shaded lines
    )

    # 2. Update the color for modern Matplotlib
    # In newer versions, we iterate directly over the ContourSet
    for path_collection in cs_hatch.get_paths():
        # We apply the edge color to the segments within the path
        cs_hatch.set_edgecolor('black')
        cs_hatch.set_linewidth(0)

    # ITG critical contour
    cs_itg = ax.contour(
        Te_2d/1e3, Ne_2d/1e20, Hartmann_ITG_2d,
        levels=[H_c],
        colors='black',
        linestyles='-',
        linewidths=2
    )

    # Finn critical contour
    cs_finn = ax.contour(
        Te_2d/1e3, Ne_2d/1e20, Hartmann_Finn_2d,
        levels=[H_c],
        colors='black',
        linestyles='--',
        linewidths=2,
    )

    # total critical contour
    cs_tot = ax.contour(
        Te_2d/1e3, Ne_2d/1e20, Hartmann_total_2d,
        levels=[H_c],
        colors='black',
        linestyles='-.',
        linewidths=2
    )

    # critical beta
    cs_beta = ax.contour(
        Te_2d/1e3, Ne_2d/1e20, beta_N_2d,
        levels=[beta_Nc],
        colors='black',
        linestyles=':',
        linewidths=2
    )

    sc_base = ax.scatter(Te/1e3, Ne/1e20, marker = 'o', s = 200, edgecolor = 'black', facecolor='none', linewidth = 2)

    # After all your contour calls:
    lines = [cs_itg.legend_elements()[0][0], 
            cs_finn.legend_elements()[0][0], 
            cs_tot.legend_elements()[0][0],
            cs_beta.legend_elements()[0][0],
            cs_hatch.legend_elements()[0][0],
            sc_base]

    labels = [r'$H_\text{ITG}=H_c$', r'$H_\text{Finn}=H_c$', r'$H_\text{ITG+Finn}=H_c$', r'$\beta_p=\beta_{p,c}$', 'Flux pumping', 'Base case']

    leg = ax.legend(lines, labels, loc='lower left')
    # Make the legend draggable
    leg.set_draggable(True)
    leg.get_frame().set_alpha(1)

    # log scales for axes
    plt.xscale('log')
    plt.yscale('log')

    # labels
    plt.xlabel(r'$T_e$ [keV]')
    plt.ylabel(r'$N_e$ [10$^{20}$ m$^{-3}$]')

    # colorbar
    cbar = plt.colorbar(cf)
    cbar.set_ticks(np.arange(7., 8.5, 0.5))   # integer ticks from 0 to 10

    
    cbar.set_label(r'$\log_{10}{(H_\text{ITG+Finn})}$')

    plt.title('estimated $N_e-T_e$ window of flux pumping')

    plt.tight_layout()
    plt.show()
