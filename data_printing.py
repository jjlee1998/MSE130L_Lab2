import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
from deconvolve import predict_j, basic_j

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
    eta = phi - dr['phi_corr']
    j0 = df['j0']
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
# no graph necessary, but plotting everything on the same set(s) of axes to ensure that they work...
#

fig_2a, ax_2a = plt.subplots(nrows=1, ncols=1)
ax_2a.set_yscale('log')

for n in range(8):

    lpr_u_dfs[n].plot(x='potential_V', y='abs_j_A/mm2', c=f'C{n}', ax=ax_2a)
    lpr_d_dfs[n].plot(x='potential_V', y='abs_j_A/mm2', c=f'C{n}', ax=ax_2a)
    anocat_u_dfs[n].plot(x='potential_V', y='abs_j_A/mm2', c=f'C{n}', ax=ax_2a)
    anocat_d_dfs[n].plot(x='potential_V', y='abs_j_A/mm2', c=f'C{n}', ax=ax_2a)
    ss_h2so4_df.plot(x='potential_V', y='abs_j_A/mm2', c=f'C{n}', ax=ax_2a)
    ss_hcl_df.plot(x='potential_V', y='abs_j_A/mm2', c=f'C{n}', ax=ax_2a)

#plt.show()
plt.close(fig_2a)

# 2b - plot log|icell| versus SCE potential for 1018MS in H2SO4 and HCl
# (implication: use this to determinine the corrosion potential and current)

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

plt.savefig(fname='Writeup/resources/fig_2b.png', format='png', dpi=100)

print_df_2b = params_df_1018MS_anocat.copy()
print_df_2b.reset_index(inplace=True)
print_df_2b['a1_var'] = np.log(10)**2 / print_df_2b['a1_var']**4
print_df_2b['a2_var'] = np.log(10)**2 / print_df_2b['a2_var']**4
print_df_2b['a1'] = np.log(10) / print_df_2b['a1']
print_df_2b['a2'] = np.log(10) / print_df_2b['a2']
print_df_2b.columns=['Scan', 'Dir', '$A_H$ (V)', '$A_{Fe}$ (V)', '$j_{corr}$ (A/mm$^2$)', '$\Delta \phi_{corr}$ (V)',
        '$\sigma^2(A_H)$', '$\sigma^2(A_{Fe})$', '$\sigma^2(j_0)$', '$\sigma^2(\Delta \phi_{corr})$', 'n']
print_df_2b['Rate'] = print_df_2b['Scan'].replace({
    0: '1 mV/sec', 1: '1 mV/sec', 2: '10 mV/sec', 3: '10 mV/sec', 4: '1 mV/sec', 5: '1 mV/sec', 6: '10 mV/sec', 7: '10 mV/sec', })
print_df_2b['Type'] = np.where(print_df_2b['Scan']%2==0, 'Ano/Cat', 'LPR')
print_df_2b['Soln'] = np.where(np.arange(16)<8, 'H$_2$SO$_4$', 'HCl')
print_df_2b['Dir'].replace({'up':'Asc', 'down':'Des'}, inplace=True)
print_df_2b = print_df_2b.reindex(columns=['Scan', 'Type', 'Soln', 'Rate', 'Dir', '$A_H$ (V)', '$A_{Fe}$ (V)', '$j_{corr}$ (A/mm$^2$)', '$\Delta \phi_{corr}$ (V)',
        '$\sigma^2(A_H)$', '$\sigma^2(A_{Fe})$', '$\sigma^2(j_0)$', '$\sigma^2(\Delta \phi_{corr})$', 'n'])
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
#



