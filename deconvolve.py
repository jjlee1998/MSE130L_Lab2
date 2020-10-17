import numpy as np
import pandas as pd
from scipy.special import lambertw
from lmfit import Parameters, minimize, fit_report

def predict_j(params, phi):

    eta = phi - params['phi0']
    j0 = params['sign_j0'] * 10**params['log10_j0']
    a0 = params['a0']
    rho_lim = 10**params['log10_rho_lim']

    eta_pass = phi - params['phi_pass']
    alpha_pass = params['alpha_pass']
    rho_pass = 10**params['log10_rho_pass']
    rho_jmak_raw = rho_pass*(1-np.exp(-alpha_pass*eta_pass**3))
    rho_jmak = np.where(eta_pass>0, rho_jmak_raw, 0)

    rho = rho_lim + rho_jmak

    x = a0*j0*rho*np.exp(a0*eta)
    w = np.real(lambertw(x))
    j_pred = w/a0/rho

    return j_pred

def j_resid(params, phi, j_data):

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

def deconvolve(phi, j_data, phi0, j0, a0, rho_lim, phi_pass=None, alpha_pass=None, rho_pass=None, fit=True):

    params = Parameters()
    params.add('phi0', value=phi0, vary=True)
    params.add('log10_j0', value=np.log10(np.abs(j0)))
    params.add('sign_j0', value=np.sign(j0))
    params.add('a0', value=a0)
    params.add('log10_rho_lim', value=np.log10(np.abs(rho_lim)))

    if (phi_pass is not None) and (alpha_pass is not None) and (rho_pass is not None):
        params.add('phi_pass', value=phi_pass)
        params.add('alpha_pass', value=alpha_pass)
        params.add('log10_rho_pass', value=np.log10(np.abs(rho_pass)))
    else:
        params.add('phi_pass', value=phi0, vary=False)
        params.add('alpha_pass', value=0, vary=False)
        params.add('log10_rho_pass', value=np.finfo(phi.dtype).minexp)

    if fit:
        params = minimize(j_resid, params, args=(phi, j_data), method='leastsq').params

    return params
