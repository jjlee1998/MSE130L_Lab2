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

    fig, ax = plt.subplots(nrows=1, ncols=1)
    df = pd.read_csv(filename, sep=sep, names=names)
    df.plot(x='potential_V', y='current_mA', ax=ax)

    figname = filename[7:]
    ax.set_title(figname)
    plt.savefig(fname=f'Graphs/Initial {figname}', format='png')
    #plt.show()
