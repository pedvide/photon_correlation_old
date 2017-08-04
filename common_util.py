# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 18:52:13 2017

@author: villanueva
"""

import warnings

import numpy as np
from scipy.optimize import curve_fit


def exp_decay(t, A, tau, *others):
    return np.piecewise(t, [t>=0, t<0], [lambda t: 1, lambda t: A*np.exp(t/tau) + 1])

def exp_decay_sigma(cov, t, A, tau, *others):
    left_side = lambda t: np.exp(t/tau)*np.sqrt(cov[0,0] + (cov[0,1]+ cov[1,0])*A*t + cov[1,1]*A**2*t**2)
    return np.piecewise(t, [t>=0, t<0], [lambda t: 0, left_side])

def symmetric_decay(t, A, tau, t0, *others):
    return 1 + A*np.exp(-np.abs(t-t0)/tau)**2

def symmetric_decay_sigma(cov, t, A, tau, t0, *others):
    common_part = (cov[0,0] +
                   cov[1,1]*A**2*np.abs(t-t0)**2/tau**4 +
                   (cov[1,2]+cov[2,1])*A**2*np.abs(t-t0)/tau**3 +
                   cov[2,2]*A**2/tau**2 +
                   (cov[1,0]+cov[0,1])*A*np.abs(t-t0)/tau**2 +
                   (cov[2,0]+cov[0,2])*A/tau)
    abs_exp = np.exp(-np.abs(t-t0)/tau)**2
    return np.sqrt(np.abs(common_part*abs_exp))


def noise_amp_fun(t, A):
    return A*t**(-0.5)

def get_signal_and_noise_from_hist(norm_hist, hist_time, fit_function_name='symmetric',
                                   expected_tau=None):
    '''Fits the folded histogram to a exponential decay.
    Returns the decay parameters and std, the background mean, and noise.'''
    fit_function = globals()[fit_function_name + '_decay']

    bin_width = hist_time[1] - hist_time[0]
    hist_len = len(norm_hist)
    signal_data = norm_hist
    signal_time = hist_time
    background_data = np.concatenate((signal_data[:hist_len//4], signal_data[-hist_len//4:]))
    background_mean = np.mean(background_data)
    noise = np.std(background_data)

    if fit_function_name is 'symmetric':
        # fit histogram
        amp_p0 = np.max(signal_data[hist_len//2-20:hist_len//2+20])
        t0_0 = signal_time[np.argmax(signal_data)]
        t0_range = (signal_time[hist_len*9//20], signal_time[hist_len*11//20])
        t0_0 = t0_0 if t0_range[0] < t0_0 < t0_range[-1] else 0
        if expected_tau is not None:
            init_params = (amp_p0, expected_tau, t0_0)
            bounds=([0, expected_tau*0.5, t0_range[0]], [2*np.max(signal_data), expected_tau*1.5, t0_range[-1]])
        else:
            init_params = (amp_p0, 50*bin_width, t0_0)
            bounds=([0, 2*bin_width, t0_range[0]], [2*np.max(signal_data), np.inf, t0_range[-1]])
    elif fit_function_name is 'exp':
        amp_p0 = np.max(signal_data[hist_len//2-20:hist_len//2+20])
        init_params = (amp_p0*2, 20*bin_width)
        bounds=(0, np.inf)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            decay_params, decay_cov = curve_fit(fit_function, signal_time, signal_data,
                                                p0=init_params, bounds=bounds)
        except RuntimeError:
            num_params = len(init_params)
            decay_params = np.zeros((num_params, 0))
            decay_cov = np.zeros((num_params, num_params))
            background_mean = 0
            noise = 0

    return decay_params, decay_cov, background_mean, noise

def get_noise_fit(noise_amplitudes, experimental_time):
    (A, ), p_cov = curve_fit(noise_amp_fun, experimental_time[noise_amplitudes>0], noise_amplitudes[noise_amplitudes>0],
                             p0=(noise_amplitudes[0]), bounds=(0, np.inf))
    A_err = np.sqrt(np.diag(p_cov))[0]

    return A, A_err