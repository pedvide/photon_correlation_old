# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 22:19:25 2017

@author: Villanueva
"""

import time
import os
import logging
from collections import deque

import tqdm
import numpy as np
import struct as struct_mod

from common_util import get_signal_and_noise_from_hist
import ptu_parser

#@profile
def read_continuous_data(file):
    '''Yield data from the file. Starts at the beginning unless start is not None.
        Continues until the EOF or end, if given.'''
    cdef bytes data
    while True:
        data = file.read(4)
        if data == b'':  # end of the file
            # check the header section TTResult_StopReason,
            # if it's -1 the user stoped the measurement
            old_pos = file.tell()
            file.seek(0, os.SEEK_SET)
            stop_reason = ptu_parser.get_from_header(file, 'TTResult_StopReason')
            if stop_reason != -1:
                return
            # go to the last position and sleep for a while, then continue
            file.seek(old_pos, os.SEEK_SET)
            time.sleep(10)
            continue
        yield data

#@profile
def produce_photons(data_file, header):
#    EVENT_TIME_CHANGE = '1111111'
#    EVENT_PHOTON_SYNC = '1000000'
#    EVENT_PHOTON_INPUT = '0000000'
    cdef int EVENT_TIME_CHANGE_B = 254
    cdef int EVENT_PHOTON_SYNC_B = 128
    cdef int EVENT_PHOTON_INPUT_B = 0

    # skip header
    cdef double res
    res = header['MeasDesc_GlobalResolution']
    data_file.seek(header['end_header_pos'], os.SEEK_SET)

    cdef unsigned long long overflow = 0
    cdef unsigned long long synccnt = 0  # counts on SYNC
    cdef unsigned long long inputcnt = 0  # counts in INPUT
    cdef long double current_time = 0

    cdef unsigned long long event_time
    cdef int event_type

    for new_event in read_continuous_data(data_file):
        # other way: slower
#        event_data = format(struct_mod.unpack('I', new_event)[0], '032b')
#        event_time = int(event_data[-25:], 2)
#        event_type = event_data[0:7]

        # new_event are 4 bytes in little-endian format
        # new_event[-1] contains the 7 bits of the event type
        # the rest contains the event time
        # in ps
        event_time = int.from_bytes(new_event[:-1], 'little') + ((new_event[-1] & 0x01) << 24)
#        print(event_time, overflow, event_time+overflow)
        event_type = new_event[-1] & 0xFE

        current_time = res*(overflow + event_time) # seconds
        if event_type == EVENT_TIME_CHANGE_B:
            overflow += event_time*(2**25)
        elif event_type == EVENT_PHOTON_SYNC_B:
            synccnt += 1
            yield (current_time, 0, synccnt, inputcnt)
        elif event_type == EVENT_PHOTON_INPUT_B:
            inputcnt += 1
            yield (current_time, 1, synccnt, inputcnt)
#        else:
#            print('Unknown event')

    yield (current_time, 2, synccnt, inputcnt)  # last time
    return

#@profile
def process_photons(produce_photons, bin_width, hist_len, plot_update_rate):
    temp_norm_hist = hist = np.zeros((hist_len, ))
    cdef double max_delta_t = bin_width*hist_len/2  # in s, maximum time difference to calculate correlation

    cdef long double last_update_time = plot_update_rate
    cdef long double current_time = 0

    # double ended-queue: fast access to the first and last elements
    photon_list = deque()
    for photon in produce_photons:
        current_time, detector, synccnt, inputcnt = photon
        photon_list.append((current_time, detector))

        if photon_list[-1][0] - photon_list[0][0] >= max_delta_t: # list is full
            # update histogram
            first_elem = photon_list.popleft()
            for elem in photon_list:
                if first_elem[1] == 0 and elem[1] == 1:
                    dif = (elem[0] - first_elem[0])/bin_width
                    locdif = int(dif) + hist_len//2
                    if 0 <= locdif < hist_len:
                        hist[locdif] += 1
                if first_elem[1] == 1 and elem[1] == 0:
                    dif = (first_elem[0] - elem[0])/bin_width
                    locdif = int(dif) + hist_len//2-1
                    if 0 <= locdif < hist_len:
                        hist[locdif] += 1

            # plot current histogram and exponential decay fit
            if current_time - last_update_time >= plot_update_rate and np.all(hist>0):
                last_update_time = current_time
                norm_factor = synccnt * inputcnt / current_time * bin_width
                temp_norm_hist =  hist/norm_factor
                yield temp_norm_hist, current_time, synccnt, inputcnt

    norm_factor = synccnt * inputcnt / current_time * bin_width
    temp_norm_hist =  hist/norm_factor
    yield temp_norm_hist, current_time, synccnt, inputcnt


#@profile
def calculate_correlation(data_file, header, bin_width, hist_len, plot_queue,
                          plot_update_rate=100, fit_function_name=None,
                          expected_tau=None):
    hist_time = (np.linspace(-hist_len//2+1, hist_len//2, num=hist_len)-1)*bin_width # time in s
    signal_error_amplitudes = []

    get_photons = produce_photons(data_file, header)

    with tqdm.tqdm(desc='Calculating correlation', unit='exp. s') as pbar:
        for updated_hist in process_photons(get_photons, bin_width, hist_len, plot_update_rate):
            temp_norm_hist, current_time, synccnt, inputcnt = updated_hist
            pbar.update(int(current_time)-pbar.n)

            (decay_params, decay_cov,
             noise_mean, noise_std) = get_signal_and_noise_from_hist(temp_norm_hist, hist_time,
                                                                     fit_function_name,
                                                                     expected_tau)
            decay_error = np.sqrt(np.diag(decay_cov))
            signal_error_amplitudes.append((current_time, decay_params[0],
                                            decay_error[0], noise_mean, noise_std))

            # send new data to the plot
            plot_queue.put((temp_norm_hist, decay_params, decay_cov, current_time, synccnt, inputcnt))

    return (temp_norm_hist, current_time, synccnt, inputcnt, signal_error_amplitudes)
