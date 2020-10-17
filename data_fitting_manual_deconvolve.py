import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import erf, lambertw
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

h2so4_df = cat_df[::-1].append([ano_df_u, ano_df_d]).reset_index(drop=True)

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

# fit everything mostly manually with assistance from lmfit:

def j_304SS_hcl(params, phi, idx):

    phi_split_idx = np.argmax(phi)
    result = np.zeros_like(phi)

    phi_corr_fe_h = params['phi_corr_fe_h']
    eta_fe_h = phi - phi_corr_fe_h

    try:
        j0_h_red = 10**params['log10_j0_h_red']
        a_h_red = params['a_h_red']
        rho_h_red = 10**params['log10_rho_h_red']
        x_h_red = -a_h_red*j0_h_red*rho_h_red*np.exp(a_h_red*eta_fe_h)
        w_h_red = np.real(lambertw(x_h_red))
        j_h_red = - w_h_red/(a_h_red*rho_h_red)
        result += j_h_red
    except Exception as e:
        pass
        #print(e)

    try:
        j0_fe_ox = 10**params['log10_j0_fe_ox']
        a_fe_ox = params['a_fe_ox']
        rho_fe_ox = 10**params['log10_rho_fe_ox']
        try:
            a_pass = params['a_pass']
            eta0_pass = params['phi0_pass'] - phi_corr_fe_h
            rho_fe_pass = 10**params['log10_rho0_pass']
            jmak = rho_fe_pass*(1-np.exp(-a_pass*(eta_fe_h-eta0_pass)**3))
            rho_fe_ox = rho_fe_ox + np.where(eta_fe_h>eta0_pass, jmak, 0)
        except Exception as e:
            pass
        x_fe_ox = a_fe_ox*j0_fe_ox*rho_fe_ox*np.exp(a_fe_ox*eta_fe_h)
        w_fe_ox = np.real(lambertw(x_fe_ox))
        j_fe_ox = - w_fe_ox/(a_fe_ox*rho_fe_ox)

        '''
        if np.isscalar(rho_fe_ox):
            j_fe_ox = - w_fe_ox/(a_fe_ox*rho_fe_ox)
        else:
            with np.errstate(divide='ignore'):
                j_fe_ox_tmp = j0_fe_ox*np.exp(a_fe_ox*eta_fe_h)
                j_fe_ox_pass = -w_fe_ox/(a_fe_ox*rho_fe_ox)
            j_fe_ox = np.where(rho_fe_ox > 0, - j_fe_ox_pass, j_fe_ox_tmp)
            print(j_fe_ox)
            '''
        result += j_fe_ox
    except Exception as e:
        pass

    phi_pit = params['phi_pit']
    eta_pit = phi - phi_pit

    try:
        j0_cl_pit = 10**params['log10_j0_cl_pit']
        a_cl_pit = params['a_cl_pit']
        rho_cl_pit = 10**params['log10_rho_cl_pit']
        x_cl_pit = a_cl_pit*j0_cl_pit*rho_cl_pit*np.exp(a_cl_pit*eta_pit)
        w_cl_pit = np.real(lambertw(x_cl_pit))
        j_cl_pit = - w_cl_pit/(a_cl_pit*rho_cl_pit)
        result += j_cl_pit
    except Exception as e:
        pass
        #print(e)


    return result

def j_304SS_hcl_resid(params, phi, idx, j):

    phi_split_idx = np.argmax(phi)

    with np.errstate(divide='ignore'):
        j_pred = j_304SS_hcl(params, phi, idx)
        alog_j_pred = np.log10(np.abs(j_pred))
        alog_j = np.log10(np.abs(j))

    fi = np.finfo(alog_j_pred.dtype)
    posinf = fi.maxexp
    neginf = fi.minexp
    
    alog_j_pred = np.nan_to_num(alog_j_pred, posinf=posinf, neginf=neginf)
    alog_j = np.nan_to_num(alog_j, posinf=posinf, neginf=neginf)

    resid = alog_j_pred - alog_j

    return resid**2

phi_hcl = hcl_df['potential_V'].values
idx_hcl = hcl_df.index.values
j_hcl = hcl_df['j_A/mm2'].values
mask_hcl = np.zeros_like(idx_hcl, dtype=bool)

def lock_params(params):
    for name, param in params.items():
        param.set(vary=False)

params_hcl = Parameters()

params_hcl.add('phi_corr_fe_h', value=-0.5)
params_hcl.add('phi_pit', value=-0.50)

params_hcl.add('log10_j0_h_red', value=-5)
params_hcl.add('log10_rho_h_red', value=2)
params_hcl.add('a_h_red', value=-15)
mask_hcl[:700] = True
#params_hcl=minimize(j_304SS_hcl_resid, params_hcl, args=(phi_hcl[mask_hcl], idx_hcl[mask_hcl], j_hcl[mask_hcl])).params
lock_params(params_hcl)

params_hcl.add('log10_j0_fe_ox', value=0)
params_hcl.add('log10_rho_fe_ox', value=3)
params_hcl.add('a_fe_ox', value=30)
mask_hcl[700:1000] = True
mask_hcl[1800:2100] = True
#params_hcl=minimize(j_304SS_hcl_resid, params_hcl, args=(phi_hcl[mask_hcl], idx_hcl[mask_hcl], j_hcl[mask_hcl])).params
lock_params(params_hcl)

params_hcl_dec = params_hcl

mask_hcl[:] = False
mask_hcl[775:995] = True
mask_hcl[1350:1450] = True
params_hcl.add('a_pass', value=1e2)
params_hcl.add('log10_rho0_pass', value=3.9)
params_hcl.add('phi0_pass', value=-0.25)
#params_hcl=minimize(j_304SS_hcl_resid, params_hcl, args=(phi_hcl[mask_hcl], idx_hcl[mask_hcl], j_hcl[mask_hcl])).params
lock_params(params_hcl)


params_hcl.add('log10_j0_cl_pit', value=-8)
params_hcl.add('log10_rho_cl_pit', value=2)
params_hcl.add('a_cl_pit', value=20)
mask_hcl[1550:1650] = True
#params_hcl=minimize(j_304SS_hcl_resid, params_hcl, args=(phi_hcl[mask_hcl], idx_hcl[mask_hcl], j_hcl[mask_hcl])).params
lock_params(params_hcl)

j_pred_hcl = j_304SS_hcl(params_hcl, phi_hcl, idx_hcl)
params_hcl.pretty_print()

fig, axes = plt.subplots(nrows=1, ncols=2)
ax = axes.ravel()
ax[0].scatter(phi_hcl, np.abs(j_hcl), s=0.5, c='k')
ax[0].scatter(phi_hcl, np.abs(j_pred_hcl), c=(j_pred_hcl>0), cmap='viridis', s=1)
ax[1].scatter(idx_hcl, np.abs(j_hcl), s=0.5, c='k')
ax[1].scatter(idx_hcl, np.abs(j_pred_hcl), c=(j_pred_hcl>0), cmap='viridis', s=1)
ax[0].set_yscale('log')
ax[1].set_yscale('log')
plt.show()

