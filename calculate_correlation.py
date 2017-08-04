# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 18:46:10 2017

@author: villanueva
"""
import time
import os
import logging

import numpy as np

from multiprocessing import Process, Queue

import ptu_parser
from plotter import Plotter
from common_util import get_signal_and_noise_from_hist, get_noise_fit

# fast
import pyximport; pyximport.install()
from calculate_correlation_continuous_cython import calculate_correlation
# slow
#from calculate_correlation_continuous import calculate_correlation

#### CHANGE THIS ####
bin_width = 1e-6  # in seconds
expected_tau = 20e-6
filename = "1Pr_NaLaF4_1h_20m_300i_400s_splitter.ptu"
#filename = "CdSe/CdSe_core_shell_3d_14h_49m_700i_900s_exc230nm.ptu"
# for continuous measurements use a rate of at least 100 s,
# so the total number of updates is not too high (depends on RAM)
PLOT_UPDATE_RATE = 100 # s, of the measurement
hist_len = 400  # bins
fit_function_name = 'symmetric' # 'symmetric' or 'exp'
#####################

plotter = Plotter(hist_len, bin_width, fit_function_name=fit_function_name)

# everything inside the if runs only in the main process, not in the plot process
if __name__=='__main__':

    basename = os.path.splitext(os.path.basename(filename))[0]

    logger = logging.getLogger(__name__)
    for handler in logger.handlers:
        logger.removeHandler(handler)
    logger.setLevel(logging.INFO)
    # create console handler and set level to INFO
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create file handler which logs even debug messages
    fh = logging.FileHandler('correlation_{}.log'.format(basename), mode='w')
    fh.setLevel(logging.INFO)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # add formatter to ch and fh
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    # add ch and fh to logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    # starting of script
    start_time = time.time()
    logger.info("Script started.")
    logger.info("Filename: {}.".format(filename))

    # get header with all the information about the file
    header = ptu_parser.parse_header(filename)
    num_events = header['TTResult_NumberOfRecords']
    resolution = header['MeasDesc_GlobalResolution']

    logger.info("Number of events (0 if running): {}.".format(num_events))
    logger.info("Resolution: {} s.".format(resolution))
    if num_events != 0:
        exp_time = header['TTResult_StopAfter']*1e-3  # in s
        logger.info("Measurement time: {:.2g} s.".format(exp_time))
    logger.info("Bin width: {:.1g} s.".format(bin_width))

    # Start plot as a new process
    plot_queue = Queue()
    process = Process(target=plotter, args=(plot_queue,), daemon=True, name='Plot')
    process.start()
    # prepare reading data file
    with open(filename, "rb") as data_file:
        # calculate correlation
        (norm_hist, total_time,
         synccnt, inputcnt, amplitudes) = calculate_correlation(data_file, header,
                                                                bin_width, hist_len, plot_queue,
                                                                plot_update_rate=PLOT_UPDATE_RATE,
                                                                fit_function_name=fit_function_name,
                                                                expected_tau=expected_tau)

    hist_time = (np.linspace(-hist_len//2+1, hist_len//2, num=hist_len)-1)*bin_width # time in s
    full_hist = np.array(np.column_stack((hist_time, norm_hist)))

    # Get final signal amplitude and standard deviation, and noise amplitude
    decay_params, decay_cov, background_mean, noise_amp = get_signal_and_noise_from_hist(norm_hist, hist_time,
                                                                                         fit_function_name=fit_function_name,
                                                                                         expected_tau=expected_tau)
    decay_error = np.sqrt(np.diag(decay_cov))
    signal_amplitude = decay_params[0]
    signal_amplitude_std = decay_error[0]
    signal_decay_tau = decay_params[1]
    signal_decay_std = decay_error[1]
    if len(decay_params) > 2:
        signal_time_offset = decay_params[2]
        signal_time_offset_std = decay_error[2]

    signal_noise_amplitudes = np.array(amplitudes)
    experimental_time = signal_noise_amplitudes[:, 0]
    noise_amplitudes = signal_noise_amplitudes[:, 4]

    noise_A, noise_A_err = get_noise_fit(noise_amplitudes, experimental_time)
    if noise_amplitudes[-1] < signal_amplitude:
        crossing_time = (noise_A/signal_amplitude)**2
        crossing_time_std = crossing_time*np.sqrt((signal_amplitude_std/signal_amplitude)**2 + (noise_A_err/noise_A)**2)
    else:
        crossing_time = crossing_time_std = 0.0

    # experimental time in the header, it should be the same as total_time
    exp_time = header['TTResult_StopAfter']*1e-3  # in s

    # log some info to file
    logger.info("Counts on SYNC: {}.".format(synccnt))
    logger.info("Counts on INPUT: {}.".format(inputcnt))
    logger.info("Average counts on SYNC: {:.0f} cps.".format(synccnt/total_time))
    logger.info("Average counts on INPUT: {:.0f} cps.".format(inputcnt/total_time))
    logger.info("Ratio SYNC/INPUT: {:.1f}.".format(synccnt/inputcnt))
    logger.info("Signal amplitude: {:.3g}\u00B1{:.2g}.".format(signal_amplitude, signal_amplitude_std))
    logger.info("Signal decay lifetime: {:.3g}\u00B1{:.1g} s.".format(signal_decay_tau, signal_decay_std))
    logger.info("Background mean residual\u00B1noise: {:.2g}\u00B1{:.2g}.".format(background_mean-1, noise_amp))
    logger.info("Noise fit parameter: {:.3g}\u00B1{:.2g}.".format(noise_A, noise_A_err))
    if len(decay_params) > 2:
        logger.info("Signal fit time offset: {:.3g}\u00B1{:.3g} s.".format(signal_time_offset, signal_time_offset_std))
    logger.info("Crossing time: {:.3g}\u00B1{:.2g} s.".format(crossing_time, crossing_time_std))
    logger.info('Experiment time (header) {:.2g} s ({:.2g} s).'.format(total_time, exp_time))

    # write results to file
    with open('histogram_{}.txt'.format(basename), 'wt') as output_file:
        output_file.write('# Filename: {}\n'.format(filename))
        output_file.write('# delay (seconds), normalized histogram\n')
        for bin_time, hist_count in full_hist:
            output_file.write("{:e}\t{:f}\n".format(bin_time, hist_count))
    # write results to file
    with open('signal_and_noise_{}.txt'.format(basename), 'wt') as output_file:
        output_file.write('# Filename: {}\n'.format(filename))
        output_file.write('# Signal time (s), Signal amplitude, Signal STD, Background level, Noise Amplitude\n')
        for signal_time, signal_amp, signal_std, noise_mean, noise_std in signal_noise_amplitudes:
            output_file.write("{:e}\t{:f}\t{:f}\t{:f}\t{:f}\n".format(signal_time, signal_amp, signal_std, noise_mean, noise_std))

    # finish script
    logger.info("Analysis time: {:.1f} s.".format(time.time() - start_time))
    logging.shutdown()
    for handler in logger.handlers:
        logger.removeHandler(handler)

    input('Press any key to continue.')
    plot_queue.put(None)  # put None to close the plot window
    process.join()

