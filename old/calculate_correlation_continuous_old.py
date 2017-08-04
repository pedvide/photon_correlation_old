# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 22:19:25 2017

@author: Villanueva
"""

import struct
import time
import os
import logging

import tqdm
import numpy as np

#import matplotlib.animation as animation
import matplotlib.pyplot as plt

from plotter import setup_plotting
from common_util import get_signal_and_noise_from_hist
import ptu_parser


def read_continuous_data(file):
    while True:
        data = file.read(4)
        if data is None or data == b'':  # end of the file
            # check the header section TTResult_StopReason, if it's -1 the user stoped the measurement
            old_pos = file.tell()
            file.seek(0, os.SEEK_SET)
            for tag_id, tag_value, _ in ptu_parser.iterate_header(file):
                if tag_id == 'TTResult_StopReason':
                    if tag_value != -1:
                        return None
                    else:
                        break
            file.seek(old_pos, os.SEEK_SET)
            time.sleep(10)
            continue
        return data


def calculate_correlation(data_file, header, bin_width, hist_len, plot_update_rate=60, fit_function_name=None):
    res = header['MeasDesc_GlobalResolution']

    # skip header
    data_file.seek(header['end_header_pos'], os.SEEK_SET)

    logger = logging.getLogger(__name__)
    event_num = 0

    overflow = 0
    synccnt = 0  # counts on SYNC
    inputcnt = 0  # counts in INPUT
    currentlist = [[0,2],[0,2]]
    hist = np.zeros((hist_len, )) #[0]*hist_len
    hist_time = (np.linspace(-hist_len//2+1, hist_len//2, num=hist_len)-1)*bin_width # time in s
    current_time = 0

    EVENT_TIME_CHANGE = 63
    EVENT_PHOTON_SYNC = 0
    EVENT_PHOTON_INPUT = 0

    last_update_time = 0

    signal_error_amplitudes = []

    update_plots = setup_plotting(hist_len, bin_width, fit_function_name=fit_function_name)

    max_delta_t = bin_width*hist_len/2  # in s, maximum time difference to calculate correlation

    with tqdm.tqdm(desc='Calculating correlation', unit='events') as pbar:
        while True:
            k = 0
            while currentlist[k][0] - currentlist[0][0] < max_delta_t:
                while len(currentlist) <= k + 1:
                    # process event
                    new_data = read_continuous_data(data_file)
                    if new_data is None:  # user stopped the measurement
                        norm_factor = synccnt * inputcnt / current_time * bin_width
                        norm_hist =  hist/norm_factor
                        decay_params, decay_cov, noise_mean, noise_std = get_signal_and_noise_from_hist(norm_hist, hist_time,
                                                                                                        fit_function_name)
                        decay_error = np.sqrt(np.diag(decay_cov))
                        signal_error_amplitudes.append((current_time, decay_params[0], decay_error[0],
                                                        noise_mean, noise_std))
                        update_plots(norm_hist, decay_params, decay_cov, current_time)
                        plt.ioff()
                        return (norm_hist, current_time, synccnt, inputcnt, signal_error_amplitudes)
                    pbar.update(1)
                    event_num += 1
                    event_data = format(struct.unpack('I', new_data)[0], '032b')

                    # process event
                    event_time = int(event_data[-25:], 2)  # in ps
                    event_channel = int(event_data[1:7], 2)
                    event_special = int(event_data[0], 2)
                    current_time = res*(overflow + event_time) # seconds
                    if event_special == EVENT_PHOTON_INPUT:  # INPUT photon
                        inputcnt += 1
                        currentlist.append([current_time, 1])
                    else:
                        if event_channel == EVENT_TIME_CHANGE: # overflow
                            if event_time == 0:  # never happens in the new format
                                overflow += 2**25
                            else:
                                overflow += event_time*(2**25)
                        elif event_channel == EVENT_PHOTON_SYNC: # SYNC photon
                            synccnt += 1
                            currentlist.append([current_time, 0])
                        elif 1 <= event_channel <= 15:
                            logger.debug('Marker event')
                        else:
                            logger.debug('Unknown event')

                    if len(currentlist) > k + 1:
                        break

#                print(currentlist)

                # update histogram
                if currentlist[0][1] == 0 and currentlist[k][1] == 1:
                    dif = (currentlist[k][0] - currentlist[0][0])/bin_width
                    locdif = int(dif) + hist_len//2
                    hist[locdif] += 1
                if currentlist[0][1] == 1 and currentlist[k][1] == 0:
                    dif = (currentlist[0][0] - currentlist[k][0])/bin_width
                    locdif = int(dif) + hist_len//2-1
                    hist[locdif] += 1
                k += 1

                # plot current histogram and exponential decay fit
                if current_time - last_update_time > plot_update_rate and synccnt+inputcnt != 0:
                    last_update_time = current_time

                    norm_factor = synccnt * inputcnt / current_time * bin_width
                    temp_norm_hist =  hist/norm_factor
                    decay_params, decay_cov, noise_mean, noise_std = get_signal_and_noise_from_hist(temp_norm_hist, hist_time,
                                                                                                    fit_function_name)
                    if np.allclose(noise_std, 0.0):
                        continue
                    decay_error = np.sqrt(np.diag(decay_cov))
                    signal_error_amplitudes.append((current_time, decay_params[0], decay_error[0],
                                                    noise_mean, noise_std))
                    update_plots(temp_norm_hist, decay_params, decay_cov, current_time)

            del currentlist[0]


    norm_factor = synccnt * inputcnt / current_time * bin_width
    norm_hist = hist/norm_factor
    decay_params, decay_cov, noise_mean, noise_std = get_signal_and_noise_from_hist(norm_hist, hist_time, fit_function_name)
    decay_error = np.sqrt(np.diag(decay_cov))
    signal_error_amplitudes.append((current_time, decay_params[0], decay_error[0], noise_mean, noise_std))
    update_plots(norm_hist, decay_params, decay_cov, current_time)
    plt.ioff()

    return (norm_hist, current_time, synccnt, inputcnt, signal_error_amplitudes)
