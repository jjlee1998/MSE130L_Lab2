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
    df['j_A/mm2'] = df['current_mA'] / area_mm
    df['abs_j_mA/mm2'] = np.abs(df['j_A/mm2'])
    df['progress'] = np.linspace(0, 1, df.shape[0])
    df.dropna(inplace=True)
    dfs.append(df)

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
    j = df_masked['j_A/mm2']
    minres = minimize(j_304SS_lin_resid, params, args=(phi, j), method='leastsq')

    return minres, df_masked

# create combined HCl dataset:

cat_df = dfs[10].copy()
ano_df = dfs[11].copy()

zero_idx_cat = np.argmax(np.abs(np.diff(np.sign(cat_df['j_A/mm2']))))
zero_idx_ano = np.argmax(np.abs(np.diff(np.sign(ano_df['j_A/mm2']))))
phi0_cat = cat_df['potential_V'].iloc[zero_idx_cat]
phi0_ano = ano_df['potential_V'].iloc[zero_idx_ano]
cat_df['potential_V'] = cat_df['potential_V'] - phi0_cat + phi0_ano

split_idx = np.argmax(np.diff(ano_df['potential_V'].values) < 0)
ano_df_u = ano_df.iloc[23:split_idx, :].copy()
ano_df_d = ano_df.iloc[split_idx:, :].copy()
split_phi = ano_df['potential_V'].iloc[split_idx]

cat_df['renorm_potential_V'] = cat_df['potential_V']
ano_df_u['renorm_potential_V'] = ano_df_u['potential_V']
ano_df_d['renorm_potential_V'] = 2*split_phi - ano_df_d['potential_V']

hcl_df = cat_df[::-1].append([ano_df_u, ano_df_d]).reset_index(drop=True)

# deconvolve the H+ reduction reaction:

fig, ax = plt.subplots(nrows=1, ncols=1)

fit_idx_h = 660 #hcl
minres_h, df_h = j_304SS_fit_linear(hcl_df, fit_idx_h, 0)
j_pred_h = - j_304SS_lin(minres_h.params, hcl_df['potential_V'])
ax.scatter(hcl_df['potential_V'], np.abs(hcl_df['j_A/mm2']), s=0.5)
ax.scatter(hcl_df['potential_V'], np.abs(j_pred_h), s=0.5)
ax.scatter(hcl_df['potential_V'], np.abs(hcl_df['j_A/mm2'] - j_pred_h), s=0.5)

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
