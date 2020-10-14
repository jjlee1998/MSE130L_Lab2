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

### add fitting functions and other operations here:
