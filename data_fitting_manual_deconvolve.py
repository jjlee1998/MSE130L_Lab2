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
names = ['potential_V', 'current_A']
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
        df['current_A'].iloc[idx] = np.nan
    if n == 11:
        df['current_A'].iloc[1533:1551] = 10 * df['current_A'].iloc[1533:1551]
    area_mm = (np.pi * diameter * immersion[n]) + (np.pi * diameter**2 / 4)
    df['j_A/mm2'] = df['current_A'] / area_mm
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
cat_df = cat_df[10:]

split_idx = np.argmax(ano_df['potential_V'].values)
ano_df_u = ano_df.iloc[12:split_idx, :].copy()
ano_df_d = ano_df.iloc[split_idx:, :].copy()
split_phi = ano_df['potential_V'].iloc[split_idx]
cat_df['renorm_potential_V'] = cat_df['potential_V']
ano_df_u['renorm_potential_V'] = ano_df_u['potential_V']
ano_df_d['renorm_potential_V'] = 1*split_phi - ano_df_d['potential_V']

h2so4_df = cat_df[::-1].append([ano_df_u, ano_df_d]).reset_index(drop=True)

# fit everything mostly manually with assistance from lmfit:

def j_304SS(params, phi, idx):

    phi_split_idx = np.argmax(phi)

    j0_h_red = 10**params['log10_j0_h_red']
    a_h_red = params['a_h_red']
    j_h_red = j0_h_red * np.exp(a_h_red*phi)

    j0_fe_ox = 10**params['log10_j0_fe_ox']
    a_fe_ox = params['a_fe_ox']
    j_fe_ox = j0_fe_ox * np.exp(a_fe_ox*phi)

    j0_pass = 10**params['log10_j0_pass']
    phi_pass = params['phi_pass']
    a_pass = params['a_pass']
    cdf_pass = np.where(phi>phi_pass, 1 - np.exp(-a_pass * (phi-phi_pass)**3), 0)

    j0_crvi_ox = 10**params['log10_j0_crvi_ox']
    a_crvi_ox = params['a_crvi_ox']
    j_crvi_ox = j0_crvi_ox * np.exp(a_crvi_ox*phi)

    j0_h2o_ox = 10**params['log10_j0_h2o_ox']
    a_h2o_ox = params['a_h2o_ox']
    j_h2o_ox = j0_h2o_ox * np.exp(a_h2o_ox*phi)

    phi_h2obr = params['phi_h2obr']
    sig_h2obr = params['sig_h2obr']
    cdf_h2obr = 1/2*(1+erf((phi-phi_h2obr)/(sig_h2obr*np.sqrt(2))))

    j0_crfe_red = 10**params['log10_j0_crfe_red']
    a_crfe_red = params['a_crfe_red']
    j_crfe_red = j0_crfe_red * np.exp(a_crfe_red*(phi-0.761))

    result = -j_h_red
    result += (1-cdf_pass)*j_fe_ox
    result += (cdf_pass)*j0_pass
    result += (1-cdf_h2obr)*j_crvi_ox
    result += (cdf_h2obr)*j_h2o_ox
    result += (cdf_pass)*-j_crfe_red

    return result

def j_304SS_resid(params, phi, idx, j):

    phi_split_idx = np.argmax(phi)
    mask = np.ones_like(phi, dtype=bool)
    mask[phi<-0.65] = 0
    #mask[np.logical_and(phi<0.10, phi>-0.20)] = 0
    mask[np.logical_and(phi>0.70, phi<1)] = 0
    mask[phi>phi_split_idx] = 0

    phi_masked = phi[mask]
    idx_masked = idx[mask]
    j_masked = j[mask]

    with np.errstate(divide='ignore'):
        j_pred = j_304SS(params, phi_masked, idx_masked)
        alog_j_pred = np.log10(np.abs(j_pred))
        alog_j = np.log10(np.abs(j_masked))

    fi = np.finfo(alog_j_pred.dtype)
    posinf = fi.maxexp
    neginf = fi.minexp

    alog_j_pred = np.nan_to_num(alog_j_pred, posinf=posinf, neginf=neginf)
    alog_j = np.nan_to_num(alog_j, posinf=posinf, neginf=neginf)

    resid = alog_j_pred - alog_j

    return resid

params = Parameters()

params.add('log10_j0_h_red', value=-8, vary=False)
params.add('a_h_red', value=-20, vary=False)

params.add('log10_j0_fe_ox', value=1.724, vary=False)
params.add('a_fe_ox', value=35.48, vary=False)

params.add('log10_j0_pass', value=-4.3, vary=True)

params.add('phi_pass', value=-0.4916)
params.add('a_pass', value=213.8)

params.add('log10_j0_crvi_ox', value=-19, vary=False)
params.add('a_crvi_ox', value=35, vary=False)

params.add('log10_j0_h2o_ox', value=-9.5, vary=False)
params.add('a_h2o_ox', value=11, vary=False)

params.add('phi_h2obr', 1.03, min=0.75, vary=False)
params.add('sig_h2obr', 0.1, max=0.15, vary=False)

params.add('log10_j0_crfe_red', value=-5, vary=True)
params.add('a_crfe_red', value=-2.079, vary=True)

phi = h2so4_df['potential_V'].values
idx = h2so4_df.index.values
j = h2so4_df['current_A']
phi_fit = phi[np.logical_and(phi>-0.35, phi<0.5)]
idx_fit = idx[np.logical_and(phi>-0.35, phi<0.5)]
j_fit = j[np.logical_and(phi>-0.35, phi<0.5)]

params = minimize(j_304SS_resid, params, args=(phi_fit, idx_fit, j_fit), method='leastsq').params
j_pred = j_304SS(params, phi, idx)

params.pretty_print()

fig, ax = plt.subplots(1, 1)
ax.scatter(phi, np.abs(j), s=0.5, c='k')
ax.scatter(phi, np.abs(j_pred), cmap='viridis', s=1)
ax.set_yscale('log')
plt.show()




'''





#ax.scatter(phi, np.abs(j), s=0.5, c='k')
#ax.scatter(phi, np.abs(j_pred), c=np.linspace(0, 1, phi.size), cmap='viridis', s=1)

ax.set_yscale('log')
plt.show()

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
