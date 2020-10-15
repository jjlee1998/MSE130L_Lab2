import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lmfit import minimize, Parameters, fit_report

# Scan 0: 1018MS H2SO4 Anodic/Cathodic 1 mA ('./Data/1018MS 1M H2SO4 Cathodic Anodic')
# Scan 1: 1018MS H2SO4 LPR 1 mA ('./Data/1018MS 1M H2SO4 LPR')
# Scan 2: 1018MS H2SO4 Anodic/Cathodic 1 mA ('./Data/1018MS 1M H2SO4 Cathodic Anodic 10mA')
# Scan 3: 1018MS H2SO4 LPR 1 mA ('./Data/1018MS 1M H2SO4 LPR 10mA')
# dimensions: 3.12 mm diameter, 14.8 mm immersion

# Scan 4: 1018MS HCl Anodic/Cathodic 1 mA ('./Data/1018MS 1M HCl Cathodic Anodic 1mA')
# Scan 5: 1018MS HCl LPR 1 mA ('./Data/1018MS 1M H2SO4 LPR 1mA')
# Scan 6: 1018MS HCl Anodic/Cathodic 1 mA ('./Data/1018MS 1M HCl Cathodic Anodic 10mA')
# Scan 7: 1018MS HCl LPR 1 mA ('./Data/1018MS 1M HCl LPR 10mA')
# dimensions: 3.12 mm diameter, 15.6 mm immersion

# Scan 8: 304SS 1M H2SO4 Cathodic ('./Data/304SS 1M H2SO4 Cathodic')
# Scan 9: 304SS 1M H2SO4 Anodic ('./Data/304SS 1M H2SO4 Anodic')
# dimensions: 3.12 mm diameter, 16.1 mm immersion

# Scan 10: 304SS 1M HCl Cathodic ('./Data/304SS 1M HCl Cathodic')
# Scan 11: 304SS 1M HCl Anodic ('./Data/304SS 1M HCl Anodic')
# dimensions: 3.12 mm diameter, 13.6 mm immersion

# read in raw data in designated scan order:

sep = '\t'
names = ['potential_V', 'current_mA']
filenames = [
'./Data/1018MS 1M H2SO4 Cathodic Anodic',
'./Data/1018MS 1M H2SO4 LPR',
'./Data/1018MS 1M H2SO4 Cathodic Anodic 10mA',
'./Data/1018MS 1M H2SO4 LPR 10mA',
'./Data/1018MS 1M HCl Cathodic Anodic 1mA',
'./Data/1018MS 1M HCl LPR 1mA',
'./Data/1018MS 1M HCl Cathodic Anodic 10mA',
'./Data/1018MS 1M HCl LPR 10mA',
'./Data/304SS 1M H2SO4 Cathodic',
'./Data/304SS 1M H2SO4 Anodic',
'./Data/304SS 1M HCl Cathodic',
'./Data/304SS 1M HCl Anodic']

dfs_raw = [pd.read_csv(filename, sep=sep, names=names) for filename in filenames]

# parameters for data cleaning:

diameter = 3.12 #mm
immersion = [
        14.8, 14.8, 14.8, 14.8,
        15.6, 15.6, 15.6, 15.6,
        16.1, 16.1,
        13.6, 13.6] #mm

remove_idxes = [
        [192, 219, 329, 518, 519, 520, 521, 522],
        [155, 591, 615],
        [129, 197, 223, 451],
        [92, 512],
        [0, 1, 2, 3, 4, 5, 201],
        [],
        [140, 233, 295, 408, 559],
        [],
        [125, 235, 370],
        [1457, 2002, 2314, 2838, 2907, 2964],
        [146],
        [88]]

# clean data (scale to sample, remove anomalies)

dfs = []

for n in range(len(dfs_raw)):

    df_raw = dfs_raw[n]
    df = df_raw.copy()
    for idx in remove_idxes[n]:
        df['current_mA'].iloc[idx] = np.nan
    if n == 11:
        df['current_mA'].iloc[1533:1551] = 10 * df['current_mA'].iloc[1533:1551]
    area_mm = (np.pi * diameter * immersion[n]) + (np.pi * diameter**2 / 4)
    df['j_mA/mm2'] = df['current_mA'] / area_mm
    df['abs_j_mA/mm2'] = np.abs(df['j_mA/mm2'])
    df['progress'] = np.linspace(0, 1, df.shape[0])
    df.dropna(inplace=True)
    dfs.append(df)

# fitting methodology will be to fit linear regions, then interpolate between them
# interpolation uses gaussian erf sigmoids unless there's a zero-crossover, in which case the two eqns are simply adjusted
# (reasonable based on logical argument, not a rigorous model)
# (the limiting current of the end results are overfit and therefore unreliable)
# (but the in-between regions should be mostly okay)

# utility fitting function for local (+-20 mV unless specified otherwise) linear region:

def j_304SS_lin(params, phi):
    eta = phi - params['phi_corr']
    j0 = 10**params['log10_j0']
    a = params['a']
    return j0*np.exp(a*eta)

def j_304SS_lin_resid(params, phi, j):

    with np.errstate(divide='ignore'):
        j_pred = j_304SS_lin(params, phi)
        alog_j_pred = np.log10(np.abs(j_pred))
        alog_j = np.log10(np.abs(j))

    fi = np.finfo(j_pred.dtype)
    posinf = fi.maxexp
    neginf = fi.minexp

    alog_j_pred = np.nan_to_num(alog_j_pred, posinf=posinf, neginf=neginf)
    alog_j = np.nan_to_num(alog_j, posinf=posinf, neginf=neginf)
    
    resid = alog_j_pred - alog_j

    return resid

def j_304SS_fit_linear(df, idx, a0=0, wr=20e-3):

    params = Parameters()
    params.add('a', value=a0, vary=True)
    params.add('log10_j0', value=-5.5, vary=True)
    params.add('phi_corr', value=-0.5, vary=True)

    phi_mid = df['potential_V'].iloc[idx]
    df_mask1 = df['potential_V'] < phi_mid + wr
    df_mask2 = df['potential_V'] > phi_mid - wr
    df_masked = df[np.logical_and(df_mask1, df_mask2)]

    phi = df_masked['potential_V']
    j = df_masked['j_mA/mm2']
    minres = minimize(j_304SS_lin_resid, params, args=(phi, j), method='leastsq')

    return minres, df_masked

# utility fitting function to merge two linear regimes together:

### PUT CODE HERE

# HCl fitting procedure:

hcl_cat_df = dfs[10]
hcl_ano_df = dfs[11]
split_pt = np.argmax(np.diff(hcl_ano_df['potential_V'].values) < 0)
hcl_ano_df_u = hcl_ano_df.iloc[:split_pt, :].reset_index(drop=True)
hcl_ano_df_d = hcl_ano_df.iloc[split_pt:, :].reset_index(drop=True)

fig, ax = plt.subplots(nrows=1, ncols=1)

if False:
    ax.scatter(hcl_cat_df.index, np.abs(hcl_cat_df['j_mA/mm2']), s=1, c='C0', label='Scan 10')
    ax.scatter(hcl_ano_df_u.index, np.abs(hcl_ano_df_u['j_mA/mm2']), s=1, c='C1', label='Scan 11u')
    ax.scatter(hcl_ano_df_d.index, np.abs(hcl_ano_df_d['j_mA/mm2']), s=1, c='C2', label='Scan 11d')
else:
    ax.scatter(hcl_cat_df['potential_V'], np.abs(hcl_cat_df['j_mA/mm2']), s=1, c='C0', label='Scan 10')
    ax.scatter(hcl_ano_df_u['potential_V'], np.abs(hcl_ano_df_u['j_mA/mm2']), s=1, c='C1', label='Scan 11u')
    ax.scatter(hcl_ano_df_d['potential_V'], np.abs(hcl_ano_df_d['j_mA/mm2']), s=1, c='C2', label='Scan 11d')

# 2 H+ + 2 e- --> H2 (limiting current):
minres_limH, df_masked_limH = j_304SS_fit_linear(hcl_cat_df, 740, -15)
phi_limH = df_masked_limH['potential_V']
ax.plot(phi_limH, j_304SS_lin(minres_limH.params, phi_limH), c='C3')

# 2 H+ + 2 e- --> H2:
minres_redH, df_masked_redH = j_304SS_fit_linear(hcl_cat_df, 100, -15)
phi_redH = df_masked_redH['potential_V']
ax.plot(phi_redH, j_304SS_lin(minres_redH.params, phi_redH), c='C3')

# Fe --> Fe2+ + 2 e-:
minres_oxFe, df_masked_oxFe = j_304SS_fit_linear(hcl_ano_df_u, 140, 15)
phi_oxFe = df_masked_oxFe['potential_V']
ax.plot(phi_oxFe, j_304SS_lin(minres_oxFe.params, phi_oxFe), c='C3')

# Fe --> Fe2+ + 2 e- (passivation):
minres_passFe, df_masked_passFe = j_304SS_fit_linear(hcl_ano_df_u, 315, -15)
phi_passFe = df_masked_passFe['potential_V']
ax.plot(phi_passFe, j_304SS_lin(minres_passFe.params, phi_passFe), c='C3')

# Fe --> Fe2+ + 2 e- (limiting current):
minres_limFe, df_masked_limFe = j_304SS_fit_linear(hcl_ano_df_d, 200, -15)
phi_limFe = df_masked_limFe['potential_V']
ax.plot(phi_limFe, j_304SS_lin(minres_limFe.params, phi_limFe), c='C3')

# Cr --> Cr3+ + 3 e-:
minres_oxCrIII, df_masked_oxCrIII = j_304SS_fit_linear(hcl_ano_df_u, 420, 15)
phi_oxCrIII = df_masked_oxCrIII['potential_V']
ax.plot(phi_oxCrIII, j_304SS_lin(minres_oxCrIII.params, phi_oxCrIII), c='C3')

# Cr --> Cr3+ + 3 e- (passivation):
minres_passCrIII, df_masked_passCrIII = j_304SS_fit_linear(hcl_ano_df_u, 650, 0)
phi_passCrIII = df_masked_passCrIII['potential_V']
ax.plot(phi_passCrIII, j_304SS_lin(minres_passCrIII.params, phi_passCrIII), c='C3')

# Cl- pitting:
minres_pitCl, df_masked_pitCl = j_304SS_fit_linear(hcl_ano_df_u, 825, 10)
phi_pitCl = df_masked_pitCl['potential_V']
ax.plot(phi_pitCl, j_304SS_lin(minres_pitCl.params, phi_pitCl), c='C3')

# Cr --> Cr3+ + 3 e- (repassivation):
minres_repassCrIII, df_masked_repassCrIII = j_304SS_fit_linear(hcl_ano_df_d, 550, 0)
phi_repassCrIII = df_masked_repassCrIII['potential_V']
ax.plot(phi_repassCrIII, j_304SS_lin(minres_repassCrIII.params, phi_repassCrIII), c='C3')

# merging linear fits:


ax.set_yscale('log')
ax.legend()
plt.show()


'''
# fitting function for basic model:

def j_1018MS(params, phi):

    a1 = params['a1']
    a2 = params['a2']
    j0 = 10**params['log10_j0']
    eta = phi - params['phi_corr']
    
    return j0*(np.exp(a1*eta) - np.exp(-a2*eta))

def j_1018MS_resid(params, phi, j):

    # note that fitting is done in ln-space, not log10-space

    with np.errstate(divide='ignore'):
        j_pred = j_1018MS(params, phi)
        alog_j_pred = np.log(np.abs(j_pred))
        alog_j = np.log(np.abs(j))

    fi = np.finfo(j_pred.dtype)
    posinf = fi.maxexp
    neginf = fi.minexp

    alog_j_pred = np.nan_to_num(alog_j_pred, posinf=posinf, neginf=neginf)
    alog_j = np.nan_to_num(alog_j, posinf=posinf, neginf=neginf)
    
    resid = (alog_j_pred - alog_j)**2

    resid = np.where(np.abs(phi-params['phi_corr']) < 0.0025, 0, resid)

    return resid

def fit_1018MS(df):

    split_pt = np.argmax(np.diff(df['potential_V'].values) < 0)
    df1 = df.iloc[:split_pt, :]
    df2 = df.iloc[split_pt:, :]

    params = Parameters()
    params.add('a1', value=15, min=0, vary=False)
    params.add('a2', value=20, min=0, vary=False)
    params.add('log10_j0', value=-5.5, vary=False)
    params.add('phi_corr', value=-0.5, vary=True)

    phi1 = df1['potential_V'].values
    phi2 = df2['potential_V'].values
    j1 = df1['j_mA/mm2'].values
    j2 = df2['j_mA/mm2'].values

    params1 = minimize(j_1018MS_resid, params, args=(phi1, j1), method='basinhopping').params
    params2 = minimize(j_1018MS_resid, params, args=(phi2, j2), method='basinhopping').params

    for param in params1.values():
        param.set(vary=True)

    for param in params2.values():
        param.set(vary=True)

    window = 0.1 #mV
    phi1_fit = phi1[np.abs(phi1-params1['phi_corr'])<window/2]
    j1_fit = j1[np.abs(phi1-params1['phi_corr'])<window/2]
    phi2_fit = phi2[np.abs(phi2-params2['phi_corr'])<window/2]
    j2_fit = j2[np.abs(phi2-params2['phi_corr'])<window/2]

    method='leastsq'
    minres1 = minimize(j_1018MS_resid, params1, args=(phi1_fit, j1_fit), method=method)
    minres2 = minimize(j_1018MS_resid, params2, args=(phi2_fit, j2_fit), method=method)

    params1 = minres1.params
    params2 = minres2.params

    return df1, df2, minres1, minres2

# actually fit to Scan 0 through Scan 7:

column_names = ['a1', 'a2', 'j0', 'phi_corr', 'a1_var', 'a2_var', 'j0_var', 'phi_corr_var', 'ndata']
params_idx = pd.MultiIndex.from_product([range(8), ['up', 'down']], names=['scan', 'direction'])
params_df = pd.DataFrame(index=params_idx, columns=column_names)


for n in range(8):
    
    df1, df2, minres_u, minres_d = fit_1018MS(dfs[n])

    params_u = minres_u.params
    ndata = minres_u.ndata
    a1 = params_u['a1'].value
    a2 = params_u['a2'].value
    log10_j0 = params_u['log10_j0'].value
    phi_corr = params_u['phi_corr'].value
    a1_se = params_u['a1'].stderr
    a2_se = params_u['a2'].stderr
    log10_j0_se = params_u['log10_j0'].stderr
    phi_corr_se = params_u['phi_corr'].stderr
    j0 = 10**log10_j0
    a1_var = ndata * a1_se**2
    a2_var = ndata * a2_se**2
    j0_var = (np.log(10) * j0)**2 * ndata * log10_j0_se**2
    phi_corr_var = ndata * phi_corr_se**2
    params_df.loc[(n, 'up')] = [a1, a2, j0, phi_corr, a1_var, a2_var, j0_var, phi_corr_var, ndata]

    params_d = minres_d.params
    ndata = minres_u.ndata
    a1 = params_d['a1'].value
    a2 = params_d['a2'].value
    log10_j0 = params_d['log10_j0'].value
    phi_corr = params_d['phi_corr'].value
    a1_se = params_d['a1'].stderr
    a2_se = params_d['a2'].stderr
    log10_j0_se = params_d['log10_j0'].stderr
    phi_corr_se = params_d['phi_corr'].stderr
    j0 = 10**log10_j0
    a1_var = ndata * a1_se**2
    a2_var = ndata * a2_se**2
    j0_var = (np.log(10) * j0)**2 * ndata * log10_j0_se**2
    phi_corr_var = ndata * phi_corr_se**2
    params_df.loc[(n, 'down')] = [a1, a2, j0, phi_corr, a1_var, a2_var, j0_var, phi_corr_var, ndata]

    df1.to_csv(f'./Processed Data/nonlinear1018MS_scan{n}u.csv')
    df2.to_csv(f'./Processed Data/nonlinear1018MS_scan{n}d.csv')

    print(f'\t[nonlinear1018MS] Scan {n} fit complete and clean data written to file.')

    # uncomment these lines to inspect as we go:

params_df.to_csv('./Processed Data/nonlinear1018MS_fit_params.csv')
print('\t[nonlinear1018MS] Fitting parameters written to file.')
'''

