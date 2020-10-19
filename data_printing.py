import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
from deconvolve import predict_j, basic_j
from scipy.special import lambertw

# basic setup

#pd.set_option('display.float_format', '{:.2g}'.format)
#mpl.use('pgf')
size=8
mpl.rc('font', family='serif')
mpl.rc('font', size=12)
mpl.rc('axes', titlesize=size)
mpl.rc('axes', labelsize=size)
mpl.rc('xtick', labelsize=size)
mpl.rc('ytick', labelsize=size)
mpl.rc('legend', fontsize=int(size/2))

# import all the data

params_df_304SS_h2so4 = pd.read_csv('./Processed Data/304SS_h2so4_fit_params.csv', index_col='reaction')
params_df_304SS_hcl = pd.read_csv('./Processed Data/304SS_hcl_fit_params.csv', index_col='reaction')
params_df_1018MS_lpr = pd.read_csv('./Processed Data/linear1018MS_fit_params.csv', index_col=['scan', 'direction'])
params_df_1018MS_anocat = pd.read_csv('./Processed Data/nonlinear1018MS_fit_params.csv', index_col=['scan', 'direction'])

ss_h2so4_df = pd.read_csv('./Processed Data/304SS_h2so4_merged_scan.csv')
ss_hcl_df = pd.read_csv('./Processed Data/304SS_hcl_merged_scan.csv')
ss_h2so4_df['progress'] = np.linspace(0, 1, ss_h2so4_df.shape[0])
ss_hcl_df['progress'] = np.linspace(0, 1, ss_hcl_df.shape[0])

lpr_u_dfs = []
lpr_d_dfs = []
anocat_u_dfs = []
anocat_d_dfs = []

for n in range(8):
    lpr_u_dfs.append(pd.read_csv(f'./Processed Data/linear1018MS_scan{n}u.csv'))
    lpr_d_dfs.append(pd.read_csv(f'./Processed Data/linear1018MS_scan{n}d.csv'))
    anocat_u_dfs.append(pd.read_csv(f'./Processed Data/nonlinear1018MS_scan{n}u.csv'))
    anocat_d_dfs.append(pd.read_csv(f'./Processed Data/nonlinear1018MS_scan{n}d.csv'))

# utility functions to produce j data using a row of the params dataframes:

def j_lpr(sr, phi):
    C = 2*96485/8.314/298.15
    eta = phi - sr['phi_corr']
    j0 = sr['j0']
    return C*eta*j0

def j_anocat(sr, phi):
    a1 = sr['a1']
    a2 = sr['a2']
    j0 = sr['j0']
    eta = phi - sr['phi_corr']
    return j0 * (np.exp(a1*eta) - np.exp(-a2*eta))

def j_deconv_basic(sr, phi):
    eta = phi - sr['phi0']
    j0 = sr['j0']
    a0 = sr['a0']
    return j0 * np.exp(a0*eta)

def j_deconv_full(sr, phi, rev=False):

    eta = phi - sr['phi0']
    j0 = sr['j0']
    a0 = sr['a0']
    rho_lim = sr['rho_lim']

    eta_pass = phi-sr['phi_pass'] if not rev else sr['phi_pass']-phi
    alpha_pass = sr['alpha_pass']
    rho_pass = sr['rho_pass']

    with np.errstate(over='ignore', invalid='ignore'):

        rho_jmak_raw = rho_pass*(1-np.exp(-alpha_pass*eta_pass**3))
        rho_jmak = np.where(eta_pass>0, rho_jmak_raw, 0)

        rho = rho_lim + rho_jmak

        x = a0*j0*rho*np.exp(a0*eta)
        w = np.real(lambertw(x))

        j_pred_w = w/a0/rho
        j_pred_p = eta/(rho_pass+rho_lim)
        j_pred = np.where(np.isinf(j_pred_w), j_pred_p, j_pred_w)

    return j_pred





#
# 1 - description of potentiostat --> no data analysis necessary
#




#
# 2a - reaction identification
# plotting the deconvolutions of the 304SS curves and tabling the fitting parameters
#

fig_2a1, axes_2a1 = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
fig_2a1.set_size_inches(6, 5)
ax_2a1 = axes_2a1.ravel()
ax_2a1[0].set_yscale('log')
ax_2a1[1].set_yscale('log')

ss_h2so4_df.plot.scatter(x='potential_V', y='abs_j_A/mm2', c='progress', cmap='viridis', colorbar=None, ax=ax_2a1[0], s=1)
ss_h2so4_df.plot.scatter(x='potential_V', y='abs_j_A/mm2', c='progress', cmap='viridis', colorbar=None, ax=ax_2a1[1], s=1)

ax_2a1[0].autoscale(enable=False)
ax_2a1[1].autoscale(enable=False)

ax_2a1[1].set_xlabel(r'Potential ($\phi$) vs. SCE')
ax_2a1[0].set_ylabel(r'|j| (A/mm$^2$)')
ax_2a1[1].set_ylabel(r'|j| (A/mm$^2$)')

phi_h2so4 = ss_h2so4_df['potential_V'].values
j_h2so4_asc = np.zeros_like(phi_h2so4)
j_h2so4_dec = np.zeros_like(phi_h2so4)
h2so4_split_idx = np.argmax(phi_h2so4)

h2so4_rxn_keys = {
        'hydrogen_reduction': r'H$^+$ reduction',
        'iron_ox_passivation': r'Fe oxidation (passivation-limited)',
        'cr_breakdown_asc': r'Cr$_2$O$_3$ barrier breakdown (asc)',
        'h2o_breakdown_asc': r'H$_2$O breakdown (asc)',
        'singularity_asc': 'unknown reduction reaction',
        'cr_deposition': r'Cr$_2$O$_3$ solution deposition',
        'cr_breakdown_dec': r'Cr$_2$O$_3$ barrier breakdown (desc)',
        'h2o_breakdown_dec': r'H$_2$O breakdown (desc)',
        }

for rxn in params_df_304SS_h2so4.index.values[np.r_[:5]]:
    rev = True if rxn=='singularity_asc' else False
    j_add = j_deconv_full(params_df_304SS_h2so4.loc[rxn, :], phi_h2so4, rev=rev)
    if rxn=='singularity_asc':
        j_add[phi_h2so4 < -0.28] = np.nan
    j_h2so4_asc += np.where(np.isnan(j_add), 0, j_add)
    ax_2a1[1].plot(phi_h2so4, np.abs(j_add), label=h2so4_rxn_keys[rxn])

for rxn in params_df_304SS_h2so4.index.values[np.r_[0, 5, 6, 7]]:
    rev = False
    j_add = j_deconv_full(params_df_304SS_h2so4.loc[rxn, :], phi_h2so4, rev=rev)
    j_h2so4_dec += np.where(np.isnan(j_add), 0, j_add)
    ax_2a1[1].plot(phi_h2so4, np.abs(j_add), label=h2so4_rxn_keys[rxn])

ax_2a1[0].plot(phi_h2so4, np.abs(j_h2so4_asc), label='Reconstructed Ascending Polarization')
ax_2a1[0].plot(phi_h2so4[h2so4_split_idx:], np.abs(j_h2so4_dec[h2so4_split_idx:]), label='Reconstructed Descending Polarization')
ax_2a1[0].legend()
ax_2a1[1].legend()

#plt.show()
plt.savefig(fname='Writeup/resources/fig_2a1.png', format='png', dpi=100)
plt.close(fig_2a1)


print_df_2a1 = params_df_304SS_h2so4.copy()
print_df_2a1.reset_index(inplace=True)
print_df_2a1['a0'] = np.log(10) / print_df_2a1['a0']
print_df_2a1.columns=['Reaction',  r'$\phi_0$ (V)', r'$j_0$ (A/mm$^2$)', r'$A$ (V)', r'$\rho_{lim}$ ($\Omega \cdot$mm)',
        r'$\phi_{pass}$ (V)', r'$\alpha_{pass}$ (V$^{-3}$)', r'$\rho_{pass}$ ($\Omega \cdot$mm)']
print_df_2a1['Reaction'] = print_df_2a1['Reaction'].replace(h2so4_rxn_keys)
print_df_2a1.set_index(['Reaction'], inplace=True)
print_df_2a1_barriers = print_df_2a1[print_df_2a1[r'$\alpha_{pass}$ (V$^{-3}$)'] != 0].copy()
print_df_2a1.drop(columns=[r'$\phi_{pass}$ (V)', r'$\alpha_{pass}$ (V$^{-3}$)', r'$\rho_{pass}$ ($\Omega \cdot$mm)'], inplace=True)
print_df_2a1_barriers.drop(columns=[r'$\phi_0$ (V)', r'$j_0$ (A/mm$^2$)', r'$A$ (V)', r'$\rho_{lim}$ ($\Omega \cdot$mm)'], inplace=True)

print_df_2a1.to_latex(buf='Writeup/resources/table_2a1.tex', float_format='%.3e', escape=False)
print_df_2a1_barriers.to_latex(buf='Writeup/resources/table_2a1_barriers.tex', float_format='%.3e', escape=False)



fig_2a2, axes_2a2 = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
fig_2a2.set_size_inches(6, 5)
ax_2a2 = axes_2a2.ravel()
ax_2a2[0].set_yscale('log')
ax_2a2[1].set_yscale('log')

ss_hcl_df.plot.scatter(x='potential_V', y='abs_j_A/mm2', c='progress', cmap='viridis', colorbar=None, ax=ax_2a2[0], s=1)
ss_hcl_df.plot.scatter(x='potential_V', y='abs_j_A/mm2', c='progress', cmap='viridis', colorbar=None, ax=ax_2a2[1], s=1)

ax_2a2[1].set_xlabel(r'Potential ($\phi$) vs. SCE')
ax_2a2[0].set_ylabel(r'|j| (A/mm$^2$)')
ax_2a2[1].set_ylabel(r'|j| (A/mm$^2$)')

ax_2a2[0].autoscale(enable=False)
ax_2a2[1].autoscale(enable=False)

phi_hcl = ss_hcl_df['potential_V'].values
j_hcl_asc = np.zeros_like(phi_hcl)
j_hcl_dec = np.zeros_like(phi_hcl)
hcl_split_idx = np.argmax(phi_hcl)

hcl_rxn_keys = {
        'hydrogen_reduction':r'H$^+$ reduction',
        'iron_oxidation':'Fe oxidation (diffusion-limited)',
        'iron_ox_passivation':r'Fe oxidation (passivation-limited)',
        'cl_pitting':r'Cl$^-$ ion pitting'
        }

for rxn in params_df_304SS_hcl.index.values[np.r_[0, 2, 3]]:
    rev = False
    j_add = j_deconv_full(params_df_304SS_hcl.loc[rxn, :], phi_hcl, rev=rev)
    j_hcl_asc += np.where(np.isnan(j_add), 0, j_add)
    ax_2a2[1].plot(phi_hcl, np.abs(j_add), label=hcl_rxn_keys[rxn])

for rxn in params_df_304SS_hcl.index.values[np.r_[0, 1]]:
    rev = False
    j_add = j_deconv_full(params_df_304SS_hcl.loc[rxn, :], phi_hcl, rev=rev)
    j_hcl_dec += np.where(np.isnan(j_add), 0, j_add)
    ax_2a2[1].plot(phi_hcl, np.abs(j_add), label=hcl_rxn_keys[rxn])

ax_2a2[0].plot(phi_hcl, np.abs(j_hcl_asc), label='Reconstructed Ascending Polarization')
ax_2a2[0].plot(phi_hcl[hcl_split_idx:], np.abs(j_hcl_dec[hcl_split_idx:]), label='Reconstructed Descending Polarization')
ax_2a2[0].legend()
ax_2a2[1].legend()

#plt.show()
plt.savefig(fname='Writeup/resources/fig_2a2.png', format='png', dpi=100)
plt.close(fig_2a2)



print_df_2a2 = params_df_304SS_hcl.copy()
print_df_2a2.reset_index(inplace=True)
print_df_2a2['a0'] = np.log(10) / print_df_2a2['a0']
print_df_2a2.columns=['Reaction',  r'$\phi_0$ (V)', r'$j_0$ (A/mm$^2$)', r'$A$ (V)', r'$\rho_{lim}$ ($\Omega \cdot$mm)',
        r'$\phi_{pass}$ (V)', r'$\alpha_{pass}$ (V$^{-3}$)', r'$\rho_{pass}$ ($\Omega \cdot$mm)']
print_df_2a2['Reaction'] = print_df_2a2['Reaction'].replace(hcl_rxn_keys)
print_df_2a2.set_index(['Reaction'], inplace=True)
print_df_2a2_barriers = print_df_2a2[print_df_2a2[r'$\rho_{pass}$ ($\Omega \cdot$mm)'] != 0].copy()
print_df_2a2.drop(columns=[r'$\phi_{pass}$ (V)', r'$\alpha_{pass}$ (V$^{-3}$)', r'$\rho_{pass}$ ($\Omega \cdot$mm)'], inplace=True)
print_df_2a2_barriers.drop(columns=[r'$\phi_0$ (V)', r'$j_0$ (A/mm$^2$)', r'$A$ (V)', r'$\rho_{lim}$ ($\Omega \cdot$mm)'], inplace=True)

print_df_2a2.to_latex(buf='Writeup/resources/table_2a2.tex', float_format='%.3e', escape=False)
print_df_2a2_barriers.to_latex(buf='Writeup/resources/table_2a2_barriers.tex', float_format='%.3e', escape=False)








#
# 2b - plot log|icell| versus SCE potential for 1018MS in H2SO4 and HCl
# (implication: use this to determinine the corrosion potential and current)
#


fig_2b, axes_2b = plt.subplots(nrows=4, ncols=2, sharex=False, sharey=True)
fig_2b.set_size_inches(6, 7)
ax_2b = axes_2b.ravel()
labels_2b = {
        0:r'$H_2SO_4$ Anodic/Cathodic 1 mV/sec',
        1:r'$H_2SO_4$ LPR 1 mV/sec',
        2:r'$H_2SO_4$ Anodic/Cathodic 10 mV/sec',
        3:r'$H_2SO_4$ LPR 10 mV/sec',
        4:r'$HCl$ Anodic/Cathodic 1 mV/sec',
        5:r'$HCl$ LPR 1 mV/sec',
        6:r'$HCl$ Anodic/Cathodic 10 mV/sec',
        7:r'$HCl$ LPR 10 mV/sec',
        }

for n in range(8):

    anocat_u_df = anocat_u_dfs[n]
    anocat_d_df = anocat_d_dfs[n]
    anocat_u_df.plot.scatter(x='potential_V', y='abs_j_A/mm2', c='C0', ax=ax_2b[n], s=1)
    anocat_d_df.plot.scatter(x='potential_V', y='abs_j_A/mm2', c='C3', ax=ax_2b[n], s=1)

    phi_u = anocat_u_df['potential_V']
    phi_d = anocat_d_df['potential_V']

    anocat_row_u = params_df_1018MS_anocat.loc[(n, 'up')]
    anocat_row_d = params_df_1018MS_anocat.loc[(n, 'down')]
    
    j_u = j_anocat(anocat_row_u, phi_u)
    j_d = j_anocat(anocat_row_d, phi_d)

    ax_2b[n].plot(phi_u, j_u, c='C0', label=labels_2b[n]+' (Ascending Sweep)')
    ax_2b[n].plot(phi_d, j_d, c='C3', label=labels_2b[n]+' (Descending Sweep)')
    ax_2b[n].set_yscale('log')
    ax_2b[n].legend()
    ax_2b[n].set_xlabel(r'Potential ($\phi$) vs. SCE')
    ax_2b[n].set_ylabel(r'|j| (A/mm$^2$)')

plt.savefig(fname='Writeup/resources/fig_2b.png', format='png', dpi=100)
plt.close(fig_2b)



print_df_2b = params_df_1018MS_anocat.copy()
print_df_2b.reset_index(inplace=True)
print_df_2b['a1_var'] = np.log(10)**2 / print_df_2b['a1_var']**4
print_df_2b['a2_var'] = np.log(10)**2 / print_df_2b['a2_var']**4
print_df_2b['a1'] = np.log(10) / print_df_2b['a1']
print_df_2b['a2'] = np.log(10) / print_df_2b['a2']
print_df_2b.columns=['Scan', 'Dir', '$A_H$ (V)', '$A_{Fe}$ (V)', '$j_{corr}$ (A/mm$^2$)', '$\Delta \phi_{corr}$ (V)',
        '$\sigma^2(A_H)$', '$\sigma^2(A_{Fe})$', '$\sigma^2(j_{corr})$', '$\sigma^2(\Delta \phi_{corr})$', 'n']
print_df_2b['Rate'] = print_df_2b['Scan'].replace({
    0: '1 mV/sec', 1: '1 mV/sec', 2: '10 mV/sec', 3: '10 mV/sec', 4: '1 mV/sec', 5: '1 mV/sec', 6: '10 mV/sec', 7: '10 mV/sec', })
print_df_2b['Type'] = np.where(print_df_2b['Scan']%2==0, 'Ano/Cat', 'LPR')
print_df_2b['Soln'] = np.where(np.arange(16)<8, 'H$_2$SO$_4$', 'HCl')
print_df_2b['Dir'].replace({'up':'Asc', 'down':'Des'}, inplace=True)
print_df_2b = print_df_2b.reindex(columns=['Scan', 'Type', 'Soln', 'Rate', 'Dir', '$A_H$ (V)', '$A_{Fe}$ (V)', '$j_{corr}$ (A/mm$^2$)', '$\Delta \phi_{corr}$ (V)',
        '$\sigma^2(A_H)$', '$\sigma^2(A_{Fe})$', '$\sigma^2(j_{corr})$', '$\sigma^2(\Delta \phi_{corr})$', 'n'])
print_df_2b.set_index(['Scan', 'Type', 'Soln', 'Rate', 'Dir'], inplace=True)

print_df_2b_vals = print_df_2b.iloc[:, :4]
print_df_2b_vars = print_df_2b.iloc[:, 4:]

print_df_2b_anocat = print_df_2b.loc[pd.IndexSlice[:, 'Ano/Cat', :, :, :], :].copy()
print_df_2b_anocat.reset_index(level='Type', drop=True, inplace=True)
print_df_2b_anocat_vals = print_df_2b_anocat.iloc[:, :4]
print_df_2b_anocat_vars = print_df_2b_anocat.iloc[:, 4:]

print_df_2b_lpr = print_df_2b.loc[pd.IndexSlice[:, 'LPR', :, :, :], :].copy()
print_df_2b_lpr.reset_index(level='Type', drop=True, inplace=True)
print_df_2b_lpr_vals = print_df_2b_lpr.iloc[:, :4]
print_df_2b_lpr_vars = print_df_2b_lpr.iloc[:, 4:]

print_df_2b_vals.to_latex(buf='Writeup/resources/table_2b_values.tex', float_format='%.3e', escape=False)
print_df_2b_vars.to_latex(buf='Writeup/resources/table_2b_variances.tex', float_format='%.3e', escape=False)
print_df_2b.to_latex(buf='Writeup/resources/table_2b_full.tex', float_format='%.3e', escape=False)

print_df_2b_anocat_vals.to_latex(buf='Writeup/resources/table_2b_anocat_values.tex', float_format='%.3e', escape=False)
print_df_2b_anocat_vars.to_latex(buf='Writeup/resources/table_2b_anocat_variances.tex', float_format='%.3e', escape=False)
print_df_2b_anocat.to_latex(buf='Writeup/resources/table_2b_anocat_full.tex', float_format='%.3e', escape=False)

print_df_2b_lpr_vals.to_latex(buf='Writeup/resources/table_2b_lpr_values.tex', float_format='%.3e', escape=False)
print_df_2b_lpr_vars.to_latex(buf='Writeup/resources/table_2b_lpr_variances.tex', float_format='%.3e', escape=False)
print_df_2b_lpr.to_latex(buf='Writeup/resources/table_2b_lpr_full.tex', float_format='%.3e', escape=False)




#
# 2c - linear slope measurements
# use the measured value of the LPR slopes to calculate the corrosion rate
#


fig_2c, axes_2c = plt.subplots(nrows=4, ncols=2, sharex=False, sharey=True)
fig_2c.set_size_inches(6, 7)
ax_2c = axes_2c.ravel()
labels_2c = {
        0:r'$H_2SO_4$ Anodic/Cathodic 1 mV/sec',
        1:r'$H_2SO_4$ LPR 1 mV/sec',
        2:r'$H_2SO_4$ Anodic/Cathodic 10 mV/sec',
        3:r'$H_2SO_4$ LPR 10 mV/sec',
        4:r'$HCl$ Anodic/Cathodic 1 mV/sec',
        5:r'$HCl$ LPR 1 mV/sec',
        6:r'$HCl$ Anodic/Cathodic 10 mV/sec',
        7:r'$HCl$ LPR 10 mV/sec',
        }

for n in range(8):

    lpr_u_df = lpr_u_dfs[n]
    lpr_d_df = lpr_d_dfs[n]
    lpr_u_df.plot.scatter(x='potential_V', y='j_A/mm2', c='C0', ax=ax_2c[n], s=1)
    lpr_d_df.plot.scatter(x='potential_V', y='j_A/mm2', c='C3', ax=ax_2c[n], s=1)

    phi_u = lpr_u_df['potential_V']
    phi_d = lpr_d_df['potential_V']

    lpr_row_u = params_df_1018MS_lpr.loc[(n, 'up')]
    lpr_row_d = params_df_1018MS_lpr.loc[(n, 'down')]
    
    j_u = j_lpr(lpr_row_u, phi_u)
    j_d = j_lpr(lpr_row_d, phi_d)

    ax_2c[n].plot(phi_u, j_u, c='C0', label=labels_2c[n]+' (Ascending Sweep)')
    ax_2c[n].plot(phi_d, j_d, c='C3', label=labels_2c[n]+' (Descending Sweep)')
    ax_2c[n].legend()
    ax_2c[n].set_xlabel(r'Potential ($\phi$) vs. SCE')
    ax_2c[n].set_ylabel(r'|j| (A/mm$^2$)')

plt.savefig(fname='Writeup/resources/fig_2c.png', format='png', dpi=100)
plt.close(fig_2c)

print_df_2c = params_df_1018MS_lpr.copy()
print_df_2c.reset_index(inplace=True)
print_df_2c.columns=['Scan', 'Dir', '$j_{corr}$ (A/mm$^2$)', '$\Delta \phi_{corr}$ (V)',
        '$\sigma^2(j_{corr})$', '$\sigma^2(\Delta \phi_{corr})$', 'C', 'n']
print_df_2c.drop(columns='C', inplace=True)
print_df_2c['Rate'] = print_df_2c['Scan'].replace({
    0: '1 mV/sec', 1: '1 mV/sec', 2: '10 mV/sec', 3: '10 mV/sec', 4: '1 mV/sec', 5: '1 mV/sec', 6: '10 mV/sec', 7: '10 mV/sec', })
print_df_2c['Type'] = np.where(print_df_2c['Scan']%2==0, 'Ano/Cat', 'LPR')
print_df_2c['Soln'] = np.where(np.arange(16)<8, 'H$_2$SO$_4$', 'HCl')
print_df_2c['Dir'].replace({'up':'Asc', 'down':'Des'}, inplace=True)
print_df_2c = print_df_2c.reindex(columns=['Scan', 'Type', 'Soln', 'Rate', 'Dir', '$j_{corr}$ (A/mm$^2$)', '$\Delta \phi_{corr}$ (V)',
        '$\sigma^2(j_{corr})$', '$\sigma^2(\Delta \phi_{corr})$', 'n'])
print_df_2c.set_index(['Scan', 'Type', 'Soln', 'Rate', 'Dir'], inplace=True)
print_df_2c['$\sigma^2(\Delta \phi_{corr})$'] = '~0'

print_df_2c.drop(columns='$\sigma^2(\Delta \phi_{corr})$', inplace=True)

print_df_2c_vals = print_df_2c.iloc[:, :1]
print_df_2c_vars = print_df_2c.iloc[:, 1:]

print_df_2c_anocat = print_df_2c.loc[pd.IndexSlice[:, 'Ano/Cat', :, :, :], :].copy()
print_df_2c_anocat.reset_index(level='Type', drop=True, inplace=True)
print_df_2c_anocat_vals = print_df_2c_anocat.iloc[:, :1]
print_df_2c_anocat_vars = print_df_2c_anocat.iloc[:, 1:]

print_df_2c_lpr = print_df_2c.loc[pd.IndexSlice[:, 'LPR', :, :, :], :].copy()
print_df_2c_lpr.reset_index(level='Type', drop=True, inplace=True)
print_df_2c_lpr_vals = print_df_2c_lpr.iloc[:, :1]
print_df_2c_lpr_vars = print_df_2c_lpr.iloc[:, 1:]

print_df_2c_vals.to_latex(buf='Writeup/resources/table_2c_values.tex', float_format='%.3e', escape=False)
print_df_2c_vars.to_latex(buf='Writeup/resources/table_2c_variances.tex', float_format='%.3e', escape=False)
print_df_2c.to_latex(buf='Writeup/resources/table_2c_full.tex', float_format='%.3e', escape=False)

print_df_2c_anocat_vals.to_latex(buf='Writeup/resources/table_2c_anocat_values.tex', float_format='%.3e', escape=False)
print_df_2c_anocat_vars.to_latex(buf='Writeup/resources/table_2c_anocat_variances.tex', float_format='%.3e', escape=False)
print_df_2c_anocat.to_latex(buf='Writeup/resources/table_2c_anocat_full.tex', float_format='%.3e', escape=False)

print_df_2c_lpr_vals.to_latex(buf='Writeup/resources/table_2c_lpr_values.tex', float_format='%.3e', escape=False)
print_df_2c_lpr_vars.to_latex(buf='Writeup/resources/table_2c_lpr_variances.tex', float_format='%.3e', escape=False)
print_df_2c_lpr.to_latex(buf='Writeup/resources/table_2c_lpr_full.tex', float_format='%.3e', escape=False)






#
# 2d - maximum active dissolution rate of 304SS
# determine either as local maximum of data or as local maximum of fit
#

h2so4_sl = np.logical_and(ss_h2so4_df['potential_V'] > -0.32, ss_h2so4_df['potential_V'] < -0.24)
h2so4_idxmax_disol = ss_h2so4_df.loc[h2so4_sl, 'j_A/mm2'].argmax()
h2so4_max_phi = ss_h2so4_df.loc[h2so4_sl, 'potential_V'].iloc[h2so4_idxmax_disol]
h2so4_max_j = ss_h2so4_df.loc[h2so4_sl, 'j_A/mm2'].iloc[h2so4_idxmax_disol]
print(f'  [2c] Max dissolution rate of 304SS in H2SO4 is {h2so4_max_j} A/mm2 at {h2so4_max_phi} V.')

hcl_sl = np.logical_and(ss_hcl_df['potential_V'] > -0.20, ss_hcl_df['potential_V'] < -0.12)
hcl_idxmax_disol = ss_hcl_df.loc[hcl_sl, 'j_A/mm2'].argmax()
hcl_max_phi = ss_hcl_df.loc[hcl_sl, 'potential_V'].iloc[hcl_idxmax_disol]
hcl_max_j = ss_hcl_df.loc[hcl_sl, 'j_A/mm2'].iloc[hcl_idxmax_disol]
print(f'  [2c] Max dissolution rate of 304SS in HCl is {hcl_max_j} A/mm2 at {hcl_max_phi} V.')

fig_2d, ax_2d = plt.subplots(1, 1)
ax_2d.set_yscale('log')
ss_h2so4_df.plot.scatter(x='potential_V', y='abs_j_A/mm2', ax=ax_2d, c='C0', s=1)
ss_hcl_df.plot.scatter(x='potential_V', y='abs_j_A/mm2', ax=ax_2d, c='C3', s=1)
#plt.show()
plt.close(fig_2d)


