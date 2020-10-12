import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sep = '\t'
names = ['potential_V', 'current_mA']
filenames = [
'./Data/304SS 1M H2SO4 Anodic',
'./Data/304SS 1M H2SO4 Cathodic',
'./Data/304SS 1M HCl Anodic',
'./Data/304SS 1M HCl Cathodic',
'./Data/1018MS 1M H2SO4 Cathodic Anodic',
'./Data/1018MS 1M H2SO4 Cathodic Anodic 10mA',
'./Data/1018MS 1M H2SO4 LPR',
'./Data/1018MS 1M H2SO4 LPR 10mA',
'./Data/1018MS 1M HCl Cathodic Anodic 1mA',
'./Data/1018MS 1M HCl Cathodic Anodic 10mA',
'./Data/1018MS 1M HCl LPR 1mA',
'./Data/1018MS 1M HCl LPR 10mA']

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
