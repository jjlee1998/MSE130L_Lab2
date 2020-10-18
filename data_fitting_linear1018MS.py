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

# fitting function for 1018MS (Scan 0 through Scan 7):

T = 298.15 #K
R = 8.314 #J/mol*K
n_e = 2 # electrons exchanged in reaction
F = 96485 #C/mol
C = (n_e * F) / (R * T)

def j_1018MS_lin(params, phi):
    eta = phi - params['phi_corr']
    j0 = params['j0']
    return C * j0 * eta

def j_1018MS_lin_resid(params, phi, j):
    j_pred = j_1018MS_lin(params, phi)
    resid = j_pred - j
    return resid

def fit_1018MS(df):

    split_pt = np.argmax(np.diff(df['potential_V'].values) < 0)
    df1 = df.iloc[:split_pt, :]
    df2 = df.iloc[split_pt:, :]

    zerocross1 = np.nonzero(np.diff(np.sign(df1['j_A/mm2'].values)))[0]
    zerocross2 = np.nonzero(np.diff(np.sign(df2['j_A/mm2'].values)))[0]
    phi_corr_est1 = df1['potential_V'].iloc[zerocross1].values[0]
    phi_corr_est2 = df2['potential_V'].iloc[zerocross2].values[0]
    df1 = df1[np.abs(df1['potential_V'] - phi_corr_est1) < 20e-3]
    df2 = df2[np.abs(df2['potential_V'] - phi_corr_est2) < 20e-3]
    
    params1 = Parameters()
    params1.add('j0', value=1e-6, vary=True)
    params1.add('phi_corr', value=phi_corr_est1, vary=False)

    params2 = Parameters()
    params2.add('j0', value=1e-6, vary=True)
    params2.add('phi_corr', value=phi_corr_est2, vary=False)

    phi1 = df1['potential_V'].values
    phi2 = df2['potential_V'].values
    j1 = df1['j_A/mm2'].values
    j2 = df2['j_A/mm2'].values

    method = 'leastsq'
    minres1 = minimize(j_1018MS_lin_resid, params1, args=(phi1, j1), method=method)
    minres2 = minimize(j_1018MS_lin_resid, params2, args=(phi2, j2), method=method)
    params1 = minres1.params
    params2 = minres2.params
    
    return df1, df2, minres1, minres2

# actually fit to Scan 0 through Scan 7:

column_names = ['j0', 'phi_corr', 'j0_var', 'phi_corr_var', 'C', 'ndata']
params_idx = pd.MultiIndex.from_product([range(8), ['up', 'down']], names=['scan', 'direction'])
params_df = pd.DataFrame(index=params_idx, columns=column_names)

for n in range(8):
    
    df1, df2, minres_u, minres_d = fit_1018MS(dfs[n])

    params_u = minres_u.params
    ndata = minres_u.ndata
    j0 = params_u['j0'].value
    phi_corr = params_u['phi_corr'].value
    j0_se = params_u['j0'].stderr
    phi_corr_se = params_u['phi_corr'].stderr
    j0_var = ndata * j0_se**2
    phi_corr_var = ndata * phi_corr_se**2
    params_df.loc[(n, 'up')] = [j0, phi_corr, j0_var, phi_corr_var, C, ndata]

    params_d = minres_d.params
    ndata = minres_d.ndata
    j0 = params_d['j0'].value
    phi_corr = params_d['phi_corr'].value
    j0_se = params_d['j0'].stderr
    phi_corr_se = params_d['phi_corr'].stderr
    j0_var = ndata * j0_se**2
    phi_corr_var = ndata * phi_corr_se**2
    params_df.loc[(n, 'down')] = [j0, phi_corr, j0_var, phi_corr_var, C, ndata]

    df1.to_csv(f'./Processed Data/linear1018MS_scan{n}u.csv')
    df2.to_csv(f'./Processed Data/linear1018MS_scan{n}d.csv')

    print(f'\t[linear1018MS] Scan {n} fit complete and clean data written to file.')

    # uncomment these lines to inspect as we go:
    '''
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(df1['potential_V'], j_1018MS_lin(params_u, df1['potential_V']), label=f'Scan {n}u')
    ax.scatter(df1['potential_V'], df1['j_A/mm2'], s=1)
    ax.plot(df2['potential_V'], j_1018MS_lin(params_d, df2['potential_V']), label=f'Scan {n}d')
    ax.scatter(df2['potential_V'], df2['j_A/mm2'], s=1)
    ax.legend()
    plt.show()
    '''

params_df.to_csv('./Processed Data/linear1018MS_fit_params.csv')
print('\t[linear1018MS] Fitting parameters written to file.')
print(params_df)
