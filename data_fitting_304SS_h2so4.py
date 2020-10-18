import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import erf, lambertw
from lmfit import minimize, Parameters, fit_report
from deconvolve import deconvolve, predict_j, basic_j, params_to_dfs

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
    df['abs_j_A/mm2'] = np.abs(df['j_A/mm2'])
    df['progress'] = np.linspace(0, 1, df.shape[0])
    df.dropna(inplace=True)
    dfs.append(df)

# create combined H2SO4 dataset:

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

so4_df = cat_df[::-1].append([ano_df_u, ano_df_d]).reset_index(drop=True)

# set up diagnostic graph:

fig, axes = plt.subplots(nrows=1, ncols=2)
ax = axes.ravel()
ax[0].set_yscale('log')
ax[1].set_yscale('log')

phi_so4 = so4_df['potential_V'].values
idx_so4 = so4_df.index.values
j_so4_raw = so4_df['j_A/mm2'].values
j_so4_asc = so4_df['j_A/mm2'].values
j_so4_dec = so4_df['j_A/mm2'].values

ax[0].scatter(phi_so4, np.abs(j_so4_raw), s=0.5, c='k')
ax[1].scatter(idx_so4, np.abs(j_so4_raw), s=0.5, c='k')

# fit everything mostly manually with assistance from lmfit using the deconvolve function:

sl_h = np.s_[0:450]
params_h, minres_h = deconvolve(phi_so4[sl_h], j_so4_asc[sl_h], -0.4, -1e-9, -5e1, 5e2, fit=True)
j_h = predict_j(params_h, phi_so4[sl_h])
j_h_dcv = predict_j(params_h, phi_so4)
j_so4_asc = j_so4_asc - j_h_dcv
j_so4_dec = j_so4_dec - j_h_dcv
ax[0].scatter(phi_so4[sl_h], np.abs(j_h), s=1)
ax[1].scatter(idx_so4[sl_h], np.abs(j_h), s=1)
print('\t[304SS_h2so4] Hydrogen reduction deconvolved.')

sl_h2o_asc = np.r_[2500:2679]
params_h2o_asc, minres_h2o_asc = deconvolve(phi_so4[sl_h2o_asc], j_so4_asc[sl_h2o_asc], 1.6, 1e-4, 1e1, 1e-5, fit=True)
j_h2o_asc = basic_j(params_h2o_asc, phi_so4[sl_h2o_asc])
j_h2o_asc_dcv = basic_j(params_h2o_asc, phi_so4)
j_so4_asc = j_so4_asc - j_h2o_asc_dcv
ax[0].scatter(phi_so4[sl_h2o_asc], np.abs(j_h2o_asc), s=1)
ax[1].scatter(idx_so4[sl_h2o_asc], np.abs(j_h2o_asc), s=1)
print('\t[304SS_h2so4] Water breakdown (ascending) deconvolved.')

sl_pass = np.r_[560:700, 1050:1600]
sl_pass_2 = np.r_[560:1600]
params_pass, minres_pass = deconvolve(phi_so4[sl_pass], j_so4_asc[sl_pass], -0.4, 1e-6, 1e1, 1e3, 
        phi_pass=-0.20, alpha_pass=100, rho_pass=6e6, fit=True)
j_pass = predict_j(params_pass, phi_so4[sl_pass_2])
j_pass_dcv = predict_j(params_pass, phi_so4, pr=False)
j_so4_asc = j_so4_asc - j_pass_dcv
ax[0].scatter(phi_so4[sl_pass_2], np.abs(j_pass), s=1)
ax[1].scatter(idx_so4[sl_pass_2], np.abs(j_pass), s=1)
print('\t[304SS_h2so4] Passivation potential deconvolved.')

sl_cr_asc = np.r_[1875:2450]
params_cr_asc, minres_cr_asc = deconvolve(phi_so4[sl_cr_asc], j_so4_asc[sl_cr_asc], 1, 1e-7, 5e1, 1e3,
        phi_pass=1.3, alpha_pass=1000, rho_pass=3e3, fit=True)
j_cr_asc = predict_j(params_cr_asc, phi_so4[sl_cr_asc])
j_cr_asc_dcv = predict_j(params_cr_asc, phi_so4)
j_so4_asc = j_so4_asc - j_cr_asc_dcv
ax[0].scatter(phi_so4[sl_cr_asc], np.abs(j_cr_asc), s=1)
ax[1].scatter(idx_so4[sl_cr_asc], np.abs(j_cr_asc), s=1)
print('\t[304SS_h2so4] Chromium breakdown (ascending) deconvolved.')

sl_h2o_dec = np.r_[2700:2825]
params_h2o_dec, minres_h2o_dec = deconvolve(phi_so4[sl_h2o_dec], j_so4_dec[sl_h2o_dec], 1.6, 1e-4, 1e1, 1e-5, fit=True)
j_h2o_dec = basic_j(params_h2o_dec, phi_so4[sl_h2o_dec])
j_h2o_dec_dcv = basic_j(params_h2o_dec, phi_so4)
j_so4_dec = j_so4_dec - j_h2o_dec_dcv
ax[0].scatter(phi_so4[sl_h2o_dec], np.abs(j_h2o_dec), s=1)
ax[1].scatter(idx_so4[sl_h2o_dec], np.abs(j_h2o_dec), s=1)
print('\t[304SS_h2so4] Water breakdown (descending) deconvolved.')

sl_cr_sol = np.r_[3620:3698]
params_cr_sol, minres_cr_sol = deconvolve(phi_so4[sl_cr_sol], j_so4_dec[sl_cr_sol], 1, -1e-6, -1e3, 1e7, fit=True)
j_cr_sol = predict_j(params_cr_sol, phi_so4[sl_cr_sol])
j_cr_sol_dcv = predict_j(params_cr_sol, phi_so4)
j_so4_dec = j_so4_dec - j_cr_sol_dcv
ax[0].scatter(phi_so4[sl_cr_sol], np.abs(j_cr_sol), s=1)
ax[1].scatter(idx_so4[sl_cr_sol], np.abs(j_cr_sol), s=1)
print('\t[304SS_h2so4] Chromium VI -> III deposition deconvolved.')

sl_cr_dec = np.r_[2950:3520]
params_cr_dec, minres_cr_dec = deconvolve(phi_so4[sl_cr_dec], j_so4_dec[sl_cr_dec], 1, 1e-7, 5e1, 1e3,
        phi_pass=1.3, alpha_pass=1000, rho_pass=3e3, fit=True)
j_cr_dec = predict_j(params_cr_dec, phi_so4[sl_cr_dec])
j_cr_dec_dcv = predict_j(params_cr_dec, phi_so4)
j_so4_dec = j_so4_dec - j_cr_dec_dcv
ax[0].scatter(phi_so4[sl_cr_dec], np.abs(j_cr_dec), s=1)
ax[1].scatter(idx_so4[sl_cr_dec], np.abs(j_cr_dec), s=1)
print('\t[304SS_h2so4] Chromium breakdown (descending) deconvolved.')

sl_sng = np.r_[700:1000]
params_sng, minres_sng = deconvolve(phi_so4[sl_sng], j_so4_raw[sl_sng], 0.15, -5e-11, -7e1, 2e5,
        phi_pass=-0.03, alpha_pass=10, rho_pass=1e7, rev_pass=True, fit=False)
j_sng = predict_j(params_sng, phi_so4[sl_sng], rev_pass=True)
j_sng_dcv = predict_j(params_sng, phi_so4, rev_pass=True)
#j_so4_asc = j_so4_asc - j_sng_dcv
ax[0].scatter(phi_so4[sl_sng], np.abs(j_sng), s=3)
ax[1].scatter(idx_so4[sl_sng], np.abs(j_sng), s=3)
print('\t[304SS_h2so4] Singularity reduction source deconvolved.')

# show diagnostic plots:

j_dcv_asc = j_h_dcv + j_pass_dcv + j_cr_asc_dcv + j_h2o_asc_dcv + j_sng_dcv
j_dcv_dec = j_h_dcv + j_cr_sol_dcv + j_cr_dec_dcv + j_h2o_dec_dcv

#ax[0].scatter(phi_so4, np.abs(j_dcv_asc), s=0.5, c=(j_dcv_asc>0), cmap='viridis')
#ax[1].scatter(idx_so4, np.abs(j_dcv_asc), s=0.5, c=(j_dcv_asc>0), cmap='viridis')

#ax[0].scatter(phi_so4, np.abs(j_so4_dec), s=0.5, c=(j_so4_dec>0), cmap='viridis')
#ax[1].scatter(idx_so4, np.abs(j_so4_dec), s=0.5, c=(j_so4_dec>0), cmap='viridis')

ax[0].plot(phi_so4, np.abs(j_dcv_asc))
ax[1].plot(idx_so4, np.abs(j_dcv_asc))

ax[0].plot(phi_so4, np.abs(j_dcv_dec))
ax[1].plot(idx_so4, np.abs(j_dcv_dec))

#plt.show()

params_dict = {
        'hydrogen_reduction': params_h,
        'iron_ox_passivation': params_pass,
        'cr_breakdown_asc': params_cr_asc,
        'h2o_breakdown_asc': params_h2o_asc,
        'singularity_asc': params_sng,
        'cr_deposition': params_cr_sol,
        'cr_breakdown_dec': params_cr_dec,
        'h2o_breakdown_dec': params_h2o_dec
        }

params_val_df = params_to_dfs(params_dict)

so4_df.to_csv('./Processed Data/304SS_h2so4_merged_scan.csv')
print('\t[304SS_h2so4] Merged dataset written to file.')
params_val_df.to_csv('./Processed Data/304SS_h2so4_fit_params.csv')
print('\t[304SS_h2so4] Fitting parameters written to file.')
