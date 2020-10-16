import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import erf
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
    df['j_A/mm2'] = df['current_mA'] / area_mm
    df['abs_j_mA/mm2'] = np.abs(df['j_A/mm2'])
    df['progress'] = np.linspace(0, 1, df.shape[0])
    df.dropna(inplace=True)
    dfs.append(df)

# create combined dataset:

cat_df = dfs[8].copy()
ano_df = dfs[9].copy()

zero_idx_cat = np.argmax(np.abs(np.diff(np.sign(cat_df['j_A/mm2']))))
zero_idx_ano = np.argmax(np.abs(np.diff(np.sign(ano_df['j_A/mm2']))))
phi0_cat = cat_df['potential_V'].iloc[zero_idx_cat]
phi0_ano = ano_df['potential_V'].iloc[zero_idx_ano]
cat_df['potential_V'] = cat_df['potential_V'] - phi0_cat + phi0_ano

split_idx = np.argmax(np.diff(ano_df['potential_V'].values) < -1)
ano_df_u = ano_df.iloc[22:split_idx, :].copy()
ano_df_d = ano_df.iloc[split_idx:, :].copy()
split_phi = ano_df['potential_V'].iloc[split_idx]

cat_df['renorm_potential_V'] = cat_df['potential_V']
ano_df_u['renorm_potential_V'] = ano_df_u['potential_V']
ano_df_d['renorm_potential_V'] = 1*split_phi - ano_df_d['potential_V']

h2so4_df = cat_df[::-1].append([ano_df_u, ano_df_d]).reset_index(drop=True)

# utility fitting function for local (+-20 mV unless specified otherwise) linear region:

def j_304SS_lin(params, phi):
    j0 = 10**params['log10_j0']
    a = params['a']
    return j0*np.exp(a*phi)

def j_304SS_lin_resid(params, phi, j):

    with np.errstate(divide='ignore'):
        j_pred = j_304SS_lin(params, phi)
        alog_j_pred = np.log10(np.abs(j_pred))
        alog_j = np.log10(np.abs(j))

    fi = np.finfo(alog_j_pred.dtype)
    posinf = fi.maxexp
    neginf = fi.minexp

    alog_j_pred = np.nan_to_num(alog_j_pred, posinf=posinf, neginf=neginf)
    alog_j = np.nan_to_num(alog_j, posinf=posinf, neginf=neginf)

    resid = alog_j_pred - alog_j

    return resid

def j_304SS_fit_linear(df, idx, asc, dec, a0=0, wr=20e-3):

    params = Parameters()
    params.add('a', value=a0, vary=True)
    params.add('log10_j0', value=-5.5, vary=True)

    rr = 1e-3 #1 mV/sec ramp rate
    wr_idx = int(wr/rr)
    window_mask = np.logical_and(df.index > idx-wr_idx, df.index <= idx+wr_idx)

    phi_split_idx = np.argmax(df['potential_V'])
    
    if asc and not dec:
        dir_mask = df.index <= phi_split_idx
    elif dec and not asc:
        dir_mask = df.index > phi_split_idx
    elif asc and dec:
        dir_mask = True

    mask = np.logical_and(window_mask, dir_mask)

    df_masked = df[mask]

    phi = df_masked['potential_V']
    j = df_masked['j_A/mm2']
    minres = minimize(j_304SS_lin_resid, params, args=(phi, j), method='leastsq')

    return minres, df_masked

# configure main fitting algorithm:

fig, ax = plt.subplots(1, 1)

fit_idxes = [400, 625, 1500, 1900, 2600, 2750, 3450, 3685]
fit_merge = [False, True, True, True, False, True, False]
fit_asc = [True, True, True, True, True, False, False, True]
fit_dec = [False, False, False, False, False, True, True, True]
fit_ano = [False, True, True, True, True, True, True, False]
fit_minreses = []
fit_masked_dfs = []
n_params = len(fit_idxes)
params_304SS = Parameters()

for n in range(n_params):
    minres_n, df_n = j_304SS_fit_linear(h2so4_df, fit_idxes[n], fit_asc[n], fit_dec[n])
    params_n = minres_n.params
    ax.plot(df_n['potential_V'], j_304SS_lin(params_n, df_n['potential_V']), lw=3)
    params_n.pretty_print()
    params_304SS.add(f'a_{n}', value=params_n['a'].value)
    params_304SS.add(f'log10_j0_{n}', value=params_n['log10_j0'].value)

for n in range(n_params - 1):
    std = 10e-3 if fit_merge[n] else 1e-9
    params_304SS.add(f'std_{n}_{n+1}', value=std)
    mu_idx = int((fit_idxes[n]+fit_idxes[n+1])/2)
    params_304SS.add(f'mu_idx_{n}_{n+1}', value=mu_idx)

params_304SS['log10_j0_7'].set(value=-7)

# main fitting algorithm

def j_304SS(params, phi):

    phi_split_idx = np.argmax(phi)
    idx = np.arange(phi.size)

    a = np.asarray([params[f'a_{n}'] for n in range(n_params)])
    j0 = np.asarray([10**params[f'log10_j0_{n}'] for n in range(n_params)])
    std = np.zeros_like(a)
    mu_idx = np.zeros_like(a)

    for n in range(n_params-1):
        std[n] = params[f'std_{n}_{n+1}']
        mu_idx[n] = params[f'mu_idx_{n}_{n+1}']

    js = np.asarray([j0[n]*np.exp(a[n]*phi) for n in range(n_params)])
    weights = np.ones_like(js)

    for n in range(n_params-1):
        if not fit_asc[n]:
            js[n, :phi_split_idx] = 0

        if not fit_dec[n]:
            js[n, phi_split_idx:] = 0

    for n in range(n_params-1):
        if fit_merge[n]: 
            erf_arg = 1e-3 * (idx-mu_idx[n]) / (std[n]*np.sqrt(2))
            weight_mod = 1/2*(1 + erf(erf_arg))
            weights[n,:] = weights[n,:] * (1 - weight_mod)
            weights[n+1,:] = weights[n+1,:] * weight_mod

    for n in range(n_params):
        if not fit_ano[n]:
            js[n, :] = -js[n, :]

    return np.sum(weights*js, axis=0)

def j_304SS_resid(params, phi, j):

    with np.errstate(divide='ignore'):
        j_pred = j_304SS(params, phi)
        alog_j_pred = np.log10(np.abs(j_pred))
        alog_j = np.log10(np.abs(j))

    fi = np.finfo(alog_j_pred.dtype)
    posinf = fi.maxexp
    neginf = fi.minexp

    alog_j_pred = np.nan_to_num(alog_j_pred, posinf=posinf, neginf=neginf)
    alog_j = np.nan_to_num(alog_j, posinf=posinf, neginf=neginf)

    resid = alog_j_pred - alog_j

    return resid

# fit to data sequentially:

phi = h2so4_df['potential_V']
j = h2so4_df['j_A/mm2']

for name, param in params_304SS.items():
    param.set(vary=False)

for n in range(n_params-1):
    params_304SS[f'std_{n}_{n+1}'].set(vary=True)
    params_304SS[f'mu_idx_{n}_{n+1}'].set(vary=True)
    params_304SS = minimize(j_304SS_resid, params_304SS, args=(phi, j), method='leastsq').params
    params_304SS[f'std_{n}_{n+1}'].set(vary=False)
    params_304SS[f'mu_idx_{n}_{n+1}'].set(vary=False)

for n in range(n_params):
    params_304SS[f'a_{n}'].set(vary=True)
    params_304SS[f'log10_j0_{n}'].set(vary=True)
    params_304SS = minimize(j_304SS_resid, params_304SS, args=(phi, j), method='leastsq').params
    params_304SS[f'a_{n}'].set(vary=False)
    params_304SS[f'log10_j0_{n}'].set(vary=False)

j_pred = j_304SS(params_304SS, phi)

ax.scatter(h2so4_df.index, np.abs(j), s=0.5, c='k')
ax.scatter(h2so4_df.index, np.abs(j_pred), c=np.linspace(0, 1, phi.size), cmap='viridis', s=1)

#ax.scatter(phi, np.abs(j), s=0.5, c='k')
#ax.scatter(phi, np.abs(j_pred), c=np.linspace(0, 1, phi.size), cmap='viridis', s=1)

ax.set_yscale('log')
plt.show()

'''

# identify indices to apply local linear fits:
hcl_fit_idxes = [50, 660, 800, 900, 1065, 1170, 1400, 1620, 1950, 2260]
hcl_lin_minreses = []
hcl_lin_masked_dfs = []

# actually apply local linear fits:

for fit_idx in hcl_fit_idxes:
    minres, masked_df = j_304SS_fit_linear(hcl_df, fit_idx, 0)
    hcl_lin_minreses.append(minres)
    hcl_lin_masked_dfs.append(masked_df)

# perform interpolation operations:

hcl_mix_minreses = []

for n in range(len(hcl_lin_minreses) - 1):
    params_a = hcl_lin_minreses[n].params
    params_b = hcl_lin_minreses[n+1].params
    mdf_a = hcl_lin_masked_dfs[n]
    mdf_b = hcl_lin_masked_dfs[n+1]
    phi_min = mdf_a['potential_V'].min()
    phi_max = mdf_a['potential_V'].max()
    mix_mask = np.logical_and(hcl_df['potential_V'] > phi_min, hcl_df['potential_V'] < phi_max)
    mix_df = hcl_df[mix_mask]
    phi = mix_df['potential_V']
    j = mix_df['j_A/mm2']
    j_pred_a = j_304SS_lin(params_a, phi)
    j_pred_b = j_304SS_lin(params_b, phi)


for i in range(len(hcl_lin_minreses)):
    minres_i = hcl_lin_minreses[i]
    mdf_i = hcl_lin_masked_dfs[i]
    phi_i = mdf_i['potential_V']
    ax.plot(phi_i, j_304SS_lin(minres_i.params, phi_i))

ax.set_yscale('log')
#ax.legend()
plt.show()
'''
