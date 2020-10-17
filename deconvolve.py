import numpy as np
import pandas as pd
from scipy.special import lambertw
from lmfit import Parameters, minimize, fit_report

def basic_j(params, phi):
    
    eta = phi - params['phi0']
    j0 = params['sign_j0'] * 10**params['log10_j0']
    a0 = params['a0']
    return j0*np.exp(a0*eta)

def predict_j(params, phi, pr=False, rev_pass=False):

    eta = phi - params['phi0']
    j0 = params['sign_j0'] * 10**params['log10_j0']
    a0 = params['a0']
    rho_lim = 10**params['log10_rho_lim']

    eta_pass = phi-params['phi_pass'] if not rev_pass else params['phi_pass']-phi
    alpha_pass = params['alpha_pass']
    rho_pass = 10**params['log10_rho_pass']

    with np.errstate(over='ignore', invalid='ignore'):

        rho_jmak_raw = rho_pass*(1-np.exp(-alpha_pass*eta_pass**3))
        rho_jmak = np.where(eta_pass>0, rho_jmak_raw, 0)

        rho = rho_lim + rho_jmak

        x = a0*j0*rho*np.exp(a0*eta)
        w = np.real(lambertw(x))

        j_pred_w = w/a0/rho
        j_pred_p = eta/(rho_pass+rho_lim)
        j_pred = np.where(np.isinf(j_pred_w), j_pred_p, j_pred_w)

        if pr:
            print(j_pred)
            print(eta_pass)
            print(rho_jmak)
            print(rho)
            print(j_pred_w)
            print(j_pred_p)

    return j_pred

def j_resid(params, phi, j_data, rev_pass=False):

    with np.errstate(divide='ignore'):
        j_pred = predict_j(params, phi)
        alog_j_pred = np.log10(np.abs(j_pred))
        alog_j_data = np.log10(np.abs(j_data))
        fi = np.finfo(alog_j_pred.dtype)
        posinf = fi.maxexp
        neginf = fi.minexp
        alog_j_pred = np.nan_to_num(alog_j_pred, posinf=posinf, neginf=neginf)
        alog_j_data = np.nan_to_num(alog_j_data, posinf=posinf, neginf=neginf)
    
    resid = alog_j_pred - alog_j_data

    return resid

def deconvolve(phi, j_data, phi0, j0, a0, rho_lim, phi_pass=None, alpha_pass=None, rho_pass=None, fit=True, lock=False, rev_pass=False):

    vary = not lock
    params = Parameters()
    params.add('phi0', value=phi0, vary=vary)
    params.add('log10_j0', value=np.log10(np.abs(j0)), vary=vary)
    params.add('sign_j0', value=np.sign(j0), vary=vary)
    params.add('a0', value=a0, vary=vary)
    params.add('log10_rho_lim', value=np.log10(np.abs(rho_lim)), vary=vary)

    if (phi_pass is not None) and (alpha_pass is not None) and (rho_pass is not None):
        params.add('phi_pass', value=phi_pass)
        params.add('alpha_pass', value=alpha_pass)
        params.add('log10_rho_pass', value=np.log10(np.abs(rho_pass)))
    else:
        params.add('phi_pass', value=phi0, vary=False)
        params.add('alpha_pass', value=0, vary=False)
        params.add('log10_rho_pass', value=np.finfo(phi.dtype).minexp)

    minres=None

    if fit:
        minres = minimize(j_resid, params, args=(phi, j_data, rev_pass), method='leastsq')
        params = minres.params

    return params, minres

def params_to_dfs(params_dict):

    column_names = ['reaction', 'phi0', 'j0', 'a0', 'rho_lim', 'phi_pass', 'alpha_pass', 'rho_pass']
    params_val_df = pd.DataFrame(columns=column_names)
    params_val_df.set_index('reaction', inplace=True, drop=True)

    params_var_df = pd.DataFrame(columns=column_names)
    params_var_df.set_index('reaction', inplace=True, drop=True)

    for rxn, params in params_dict.items():

        #params = minres.params
        #ndata = minres.ndata
        #params.pretty_print()

        phi0 = params['phi0'].value
        j0 = params['sign_j0'].value * 10**params['log10_j0'].value
        a0 = params['a0'].value
        rho_lim = 10**params['log10_rho_lim'].value
        phi_pass = params['phi_pass'].value
        alpha_pass = params['alpha_pass'].value
        rho_pass = 10**params['log10_rho_pass'].value
        params_val_df.loc[rxn, :] = [phi0, j0, a0, rho_lim, phi_pass, alpha_pass, rho_pass]

        '''
        phi0_se = params['phi0'].stderr
        j0_se = params['sign_j0'].stderr * 10**params['log10_j0'].stderr
        a0_se = params['a0'].stderr
        rho_lim_se = 10**params['log10_rho_lim'].stderr
        phi_pass_se = params['phi_pass'].stderr
        alpha_pass_se = params['alpha_pass'].stderr
        rho_pass_se = 10**params['log10_rho_pass'].stderr
        se_array = np.asarray([phi0_se, j0_se, a0_se, rho_lim_se, phi_pass_se, alpha_pass_se, rho_pass_se])
        var_array = se_array**2 * n_data
        params_var_df.loc[rxn, :] = var_array
        '''

    return params_val_df
