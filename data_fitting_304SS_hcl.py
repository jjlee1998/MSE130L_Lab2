import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import erf, lambertw
from lmfit import minimize, Parameters, fit_report
from deconvolve import deconvolve, predict_j

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

# create combined HCl dataset:

cat_df = dfs[10].copy()
ano_df = dfs[11].copy()

zero_idx_cat = np.argmax(np.abs(np.diff(np.sign(cat_df['j_A/mm2']))))
zero_idx_ano = np.argmax(np.abs(np.diff(np.sign(ano_df['j_A/mm2']))))
phi0_cat = cat_df['potential_V'].iloc[zero_idx_cat]
phi0_ano = ano_df['potential_V'].iloc[zero_idx_ano]
cat_df['potential_V'] = cat_df['potential_V'] - phi0_cat + phi0_ano
cat_df = cat_df[10:]

split_idx = np.argmax(ano_df['potential_V'].values)
ano_df_u = ano_df.iloc[23:split_idx, :].copy()
ano_df_d = ano_df.iloc[split_idx:, :].copy()
split_phi = ano_df['potential_V'].iloc[split_idx]
cat_df['renorm_potential_V'] = cat_df['potential_V']
ano_df_u['renorm_potential_V'] = ano_df_u['potential_V']
ano_df_d['renorm_potential_V'] = 1*split_phi - ano_df_d['potential_V']

hcl_df = cat_df[::-1].append([ano_df_u, ano_df_d]).reset_index(drop=True)

# set up diagnostic graph:

fig, axes = plt.subplots(nrows=1, ncols=2)
ax = axes.ravel()
ax[0].set_yscale('log')
ax[1].set_yscale('log')

phi_hcl = hcl_df['potential_V'].values
idx_hcl = hcl_df.index.values
j_hcl = hcl_df['j_A/mm2'].values

ax[0].scatter(phi_hcl, np.abs(j_hcl), s=0.5, c='k')
ax[1].scatter(idx_hcl, np.abs(j_hcl), s=0.5, c='k')

# fit everything mostly manually with assistance from lmfit using the deconvolve function:

sl_h = np.s_[:700]
params_h = deconvolve(phi_hcl[sl_h], j_hcl[sl_h], -0.4, -1e-9, -1e2, 5e2, fit=True)
j_h = predict_j(params_h, phi_hcl[sl_h])
j_h_dcv = predict_j(params_h, phi_hcl)
j_hcl = j_hcl - j_h_dcv
ax[0].scatter(phi_hcl[sl_h], np.abs(j_h), s=1)
ax[1].scatter(idx_hcl[sl_h], np.abs(j_h), s=1)

sl_pass = np.r_[770:1025, 1275:1450]
sl_pass_2 = np.r_[780:1450]
params_pass = deconvolve(phi_hcl[sl_pass], j_hcl[sl_pass], -0.4, 1e-8, 1e2, 1e3, 
        phi_pass=-0.20, alpha_pass=1000, rho_pass=7e3, fit=True)
j_pass = predict_j(params_pass, phi_hcl[sl_pass_2])
j_pass_dcv = predict_j(params_pass, phi_hcl)
j_hcl = j_hcl - j_pass_dcv
ax[0].scatter(phi_hcl[sl_pass_2], np.abs(j_pass), s=1)
ax[1].scatter(idx_hcl[sl_pass_2], np.abs(j_pass), s=1)

sl_cl = np.r_[1500:1650]
params_cl = deconvolve(phi_hcl[sl_cl], j_hcl[sl_cl], 0.4, 1e-4, 1e1, 1e1, fit=True)
j_cl = predict_j(params_cl, phi_hcl[sl_cl])
j_cl_dcv = predict_j(params_cl, phi_hcl)
j_hcl = j_hcl - j_cl_dcv
ax[0].scatter(phi_hcl[sl_cl], np.abs(j_cl), s=1)
ax[1].scatter(idx_hcl[sl_cl], np.abs(j_cl), s=1)

params_fe = params_pass
params_fe['log10_rho_pass'].set(value=np.finfo(np.float).minexp)
j_fe_dcv = predict_j(params_fe, phi_hcl)

# show diagnostic plots:

j_dcv = j_h_dcv + j_pass_dcv + j_cl_dcv
j_dcv_pit = j_h_dcv + j_fe_dcv

#ax[0].scatter(phi_hcl, np.abs(j_dcv), s=0.5, c=(j_dcv>0), cmap='viridis')
#ax[1].scatter(idx_hcl, np.abs(j_dcv), s=0.5, c=(j_dcv>0), cmap='viridis')

ax[0].plot(phi_hcl, np.abs(j_dcv))
ax[1].plot(idx_hcl, np.abs(j_dcv))

ax[0].plot(phi_hcl, np.abs(j_dcv_pit))
ax[1].plot(idx_hcl, np.abs(j_dcv_pit))

plt.show()

