import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lmfit import minimize, Parameters
from copy import deepcopy

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

# fitting function for 1018MS (Scan 0 through Scan 7):

def j_1018MS(params, phi):

    a1 = params['a1']
    a2 = params['a2']
    j0_1 = 10**params['log10_j0_1']
    j0_2 = 10**params['log10_j0_2']
    eta = phi - params['phi_corr']

    return j0_1*np.exp(a1*eta) - j0_2*np.exp(-a2*eta)

def j_1018MS_resid(params, phi, j, sigma=None):

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

    if sigma is not None:
        weights = np.exp(-(phi-params['phi_corr'])**2 / (2*sigma**2))
        resid = weights*resid

    return resid

def fit_1018MS(df):

    split_pt = np.argmax(np.diff(df['potential_V'].values) < 0)
    df1 = df.iloc[:split_pt, :]
    df2 = df.iloc[split_pt:, :]

    params = Parameters()
    params.add('a1', value=15, min=0, vary=False)
    params.add('a2', value=20, min=0, vary=False)
    params.add('log10_j0_1', value=-5.5, vary=False)
    params.add('log10_j0_2', value=-5.5, vary=False)
    params.add('phi_corr', value=-0.5, vary=True)

    phi1 = df1['potential_V'].values
    phi2 = df2['potential_V'].values
    j1 = df1['j_mA/mm2'].values
    j2 = df2['j_mA/mm2'].values

    params1 = minimize(j_1018MS_resid, params, args=(phi1, j1), method='basinhopping').params
    params2 = minimize(j_1018MS_resid, params, args=(phi2, j2), method='basinhopping').params

    params1.pretty_print()
    params2.pretty_print()

    for param in params1.values():
        param.set(vary=True)
    params1['phi_corr'].set(vary=False)

    for param in params2.values():
        param.set(vary=True)
    params2['phi_corr'].set(vary=False)

    sigma = 0.1
    method='least_squares'
    params1 = minimize(j_1018MS_resid, params1, args=(phi1, j1, sigma), method=method).params
    params2 = minimize(j_1018MS_resid, params2, args=(phi2, j2, sigma), method=method).params

    params1.pretty_print()
    params2.pretty_print()

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.scatter(phi1, np.abs(j1), c='C0', s=1)
    ax.plot(phi1, np.abs(j_1018MS(params1, phi1)), c='C0')
    ax.scatter(phi2, np.abs(j2), c='C3', s=1)
    ax.plot(phi2, np.abs(j_1018MS(params2, phi2)), c='C3')

    ax.set_yscale('log')

    #df1.plot.scatter(x='potential_V', y='abs_j_mA/mm2', ax=ax, c='C0', logy=True, s=1)
    #df2.plot.scatter(x='potential_V', y='abs_j_mA/mm2', ax=ax, c='C1', logy=True, s=1)
    plt.show()

# actually fit to Scan 0 through Scan 7:

for n in range(8):
    fit_1018MS(dfs[n])

'''
for filename in filenames:

    fig, axes = plt.subplots(nrows=2, ncols=1)
    ax = axes.ravel()
    df = pd.read_csv(filename, sep=sep, names=names)
    df['abs_current_mA'] = np.abs(df['current_mA'])
    df['progress'] = np.linspace(0, 1, df.shape[0])
    #df.plot(x='potential_V', y='current_mA', ax=ax[0])
    df.plot.scatter(x='potential_V', y='current_mA', c='progress',
            colormap=plt.get_cmap('viridis'), colorbar=True, ax=ax[0], s=1)
    #df.plot(x='potential_V', y='abs_current_mA', ax=ax[1], logy=True)
    df.plot.scatter(x='potential_V', y='abs_current_mA', c='progress',
            colormap=plt.get_cmap('viridis'), colorbar=True, ax=ax[1],
            logy=True, s=1)

    figname = filename[7:]
    ax[0].set_title(f'{figname} - Raw Data')
    ax[1].set_title(f'{figname} - Absolute Log')
    fig.set_size_inches(12, 16)
    plt.savefig(fname=f'Graphs/Initial {figname}', format='png', dpi=100)
    #plt.show()
'''
