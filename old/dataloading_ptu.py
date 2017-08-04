# import packages
import numpy as np
import struct
import time
import os
import warnings
import logging

import tqdm
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
plt.ion()

from scipy.optimize import curve_fit

import ptu_parser


def setup_plot_histogram(ax, hist_len, bin_width):
    # setup the plots
    hist_time = (np.linspace(-hist_len//2+1, hist_len//2, num=hist_len)-1)*bin_width # time in s
    signal_time = hist_time[hist_len//2:]

    ax.set_xlim(left=1.01*hist_time[0]/bin_width, right=1.01*hist_time[-1]/bin_width)
    x_label_exp = '{' + str(int(np.log10(bin_width))) + '}'
    ax.set_xlabel(r'Delay $\tau$ (10$^{}$ s)'.format(x_label_exp))
    ax.set_ylabel('Correlation g$^{(2)}$')
    ax.set_title('Correlation histogram')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(-3,3), useMathText=True, useOffset=False)
    text_rate_str = 'SYNC count rate: {:d} cps.\nINPUT count rate: {:d} cps.'
    text_rates = ax.annotate(s=text_rate_str.format(0, 0), textcoords='axes fraction', xy=(0,1), xytext=(0.5, 0.9))

    plot_data, = ax.plot(hist_time/bin_width, np.ones_like(hist_time), '.')
    plot_fit, = ax.plot(signal_time/bin_width, exp_decay_fun(signal_time, 0, 1/(10*bin_width))+1, 'r-')
    plot_ci = ax.axhspan(ymin=1-0, ymax=1+0, alpha=0.25)
    plot_errbar = ax.errorbar(0, 1, yerr=0, fmt='o')

    def update_plot_histogram(temp_norm_hist, popt, perr, current_time, sync_counts, input_counts):
        nonlocal plot_errbar
        signal_data = (temp_norm_hist[hist_len//2:] + temp_norm_hist[:hist_len//2][::-1])/2-1
        noise_mean = np.mean(signal_data[-hist_len//4:])+1
        noise_std = np.std(signal_data[-hist_len//4:])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plot_data.set_ydata(temp_norm_hist)
            plot_fit.set_ydata(exp_decay_fun(signal_time, *popt)+1)
            plot_ci.set_xy([[0, noise_mean-noise_std], [0, noise_mean+noise_std],
                            [1, noise_mean+noise_std], [1, noise_mean-noise_std]])
            plot_errbar.remove()
            plot_errbar = ax.errorbar(0, popt[0]+1, yerr=perr[0], fmt='ro', ecolor='r', capsize=5)
            text_rates.set_text(s=text_rate_str.format(int(sync_counts//current_time), int(input_counts//current_time)))

            ax.autoscale_view(scalex=False, scaley=True)
            ax.relim()
            plt.pause(0.0001)
    return update_plot_histogram

def setup_plot_signal_noise(ax, hist_len):
    '''Plots the signal and the noise as a function of the experimental time.
        Returns the crossing time, ie, the time at which the signal equals the noise.'''
    ax.set_ylabel('Amplitude')
    ax.set_xlabel('Measurement time (s)')
    ax.set_title('Correlation signal and noise')
    text_crossing = ax.annotate('Estimated crossing time:', textcoords='axes fraction',
                                xy=(0,0), xytext=(0.5, 0.9))
    ax.autoscale_view(scalex=False, scaley=True)

    noise_amp_fun = lambda t, A: A*t**(-0.5)

    signal_std_arr = np.array([])

    plot_signal_amp, = ax.plot([], [], 'bo', markersize=2, label='Signal amplitude')
    plot_signal_err = ax.fill_between([], y1=[], y2=[],
                    alpha=0.5, color='b', label=r'2$\sigma$ confidence bands')
    plot_signal_line = ax.axhline(y=1, xmin=-hist_len//2, xmax=hist_len//2,
               alpha=0.5, label='Final signal amplitude')
    plot_noise_amp, = ax.plot([], [], 'ro', markersize=2, label='Noise amplitude')
    plot_noise_fit, = ax.plot([], [], 'r-',
                             alpha=0.5, label=r'Fit to noise amplitude t$^{-\frac{1}{2}}$')

    # sort the legend so the 3 signal labels are together
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(2, handles[-1])
    labels.insert(2, labels[-1])
    del handles[-1]
    del labels[-1]
    ax.legend(handles, labels)

    def update_plot_signal_noise(temp_norm_hist, popt, perr, current_time):
        nonlocal plot_signal_err, signal_std_arr
        signal_data = (temp_norm_hist[hist_len//2:] + temp_norm_hist[:hist_len//2][::-1])/2-1
        noise_std = np.std(signal_data[-hist_len//4:])
        signal_amp = popt[0]
        signal_std = perr[0]
        signal_std_arr = np.append(signal_std_arr, signal_std)
        # time
        time_arr = np.append(plot_signal_amp.get_xdata(), current_time)
        # signal amplitude
        plot_signal_amp.set_ydata(np.append(plot_signal_amp.get_ydata(), signal_amp))
        plot_signal_amp.set_xdata(time_arr)
        # confidence bands
        plot_signal_err.remove()
        cb_max = plot_signal_amp.get_ydata()+2*signal_std_arr
        cb_min = (plot_signal_amp.get_ydata()-2*signal_std_arr).clip(min=0)
        plot_signal_err = ax.fill_between(plot_signal_amp.get_xdata(),
                                          y1=cb_max, y2=cb_min,
                                          alpha=0.5, color='b', label=r'2$\sigma$ confidence bands')
        # last signal amplitude
        plot_signal_line.set_ydata(signal_amp)
        # noise amplitude
        plot_noise_amp.set_ydata(np.append(plot_noise_amp.get_ydata(), noise_std))
        plot_noise_amp.set_xdata(time_arr)
        # noise fit to t^(-1/2)
        if len(time_arr) > 3:
            (A, ), _ = curve_fit(noise_amp_fun, time_arr, plot_noise_amp.get_ydata(),
                                 p0=(plot_noise_amp.get_ydata()[0]))
            # extrapolate from the time at which the noise amplitude is the maximum signal error (t_1=(A/max_std)**2)
            extrapolated_meas_time = np.linspace((A/np.max(cb_max))**2, time_arr[-1], len(time_arr)*100)
            fit_noise_amplitude = noise_amp_fun(extrapolated_meas_time, A)
            # find crossing between noise fit and final signal amplitude
            idx = len(fit_noise_amplitude)-np.searchsorted(fit_noise_amplitude[::-1], signal_amp)
            if idx < len(extrapolated_meas_time):
                crossing_time = extrapolated_meas_time[idx]
                text_crossing.set_text('Estimated crossing time: {:.1f} s'.format(crossing_time))
            # update plot
            plot_noise_fit.set_ydata(fit_noise_amplitude)
            plot_noise_fit.set_xdata(extrapolated_meas_time)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ax.autoscale_view(scalex=True, scaley=True)
            ax.relim()
            plt.pause(0.0001)

    return update_plot_signal_noise


def exp_decay_fun(t, A, k):
    return A*np.exp(-k*t)

def read_even(thefile):
    thefile.seek(0, os.SEEK_SET) # Go to the end of the file
    while True:
        line = thefile.readline()
        if not line:
            time.sleep(0.1) # Sleep briefly
            continue
        yield line

def calculate_correlation(data_file, num_events, bin_width, hist_len, res):
    j = 0
    overflow = 0
    synccnt = 0  # counts on SYNC
    inputcnt = 0  # counts in INPUT
    currentlist = [[0,2],[0,2]]
    hist = [0]*hist_len
    current_time = 0

    EVENT_TIME_CHANGE = 63
    EVENT_PHOTON_SYNC = 0
    EVENT_PHOTON_INPUT = 0

    signal_error_amplitudes = []
    update_freq = num_events//100
    update_error = update_freq

    # setup the plots
#    fig, (ax_histo, ax_signal_noise) = plt.subplots(nrows=2, ncols=1)
#    mng = plt.get_current_fig_manager()
#    mng.window.showMaximized()
#    mng.window.activateWindow()
#    mng.window.raise_()
#    xpos, ypos, width, height = fig.canvas.manager.window.geometry().getRect()
#    fig.canvas.manager.window.move(xpos-1.1*width//2, ypos)
#    update_plot_histogram = setup_plot_histogram(ax_histo, hist_len, bin_width)
#    update_plot_signal_noise = setup_plot_signal_noise(ax_signal_noise, hist_len)
#    plt.tight_layout()

    max_delta_t = bin_width*hist_len/2  # in s, maximum time difference to calculate correlation

    with tqdm.tqdm(total=num_events, desc='Calculating correlation', unit='events') as pbar:
        while j < num_events:
            k = 0
            while currentlist[k][0] - currentlist[0][0] < max_delta_t:
                while len(currentlist) <= k + 1:
                    pbar.update(1)
                    j += 1
                    if j == num_events:
                        break

                    # process event
                    event_data = format(struct.unpack('I', data_file.read(4))[0], '032b')
                    event_time = int(event_data[-25:], 2)  # in ps
                    event_channel = int(event_data[1:7], 2)
                    event_special = int(event_data[0], 2)
                    current_time = res*(overflow + event_time)
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

                # plot current histogram and exponetial decay fit
                if current_time != 0:
                    norm_factor = synccnt * inputcnt / current_time * bin_width
                    if norm_factor != 0 and j > update_error:
                        update_error += update_freq

                        temp_norm_hist =  np.array(hist)/norm_factor
                        signal_data = (temp_norm_hist[hist_len//2:] +
                                       temp_norm_hist[:hist_len//2][::-1])/2-1
                        signal_time = hist_time[hist_len//2:]
                        noise_mean = np.mean(signal_data[-50:])+1
                        noise_std = np.std(signal_data[-50:])
                        # fit signal to exponential decay
                        # ignore exp function overflow
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            popt, pcov = curve_fit(exp_decay_fun, signal_time, signal_data,
                                                   p0=(signal_data[0], 1/(10*bin_width)))
                            perr = np.sqrt(np.diag(pcov))

                        signal_error_amplitudes.append((current_time, popt[0], perr[0], noise_mean, noise_std))
#
#                        update_plot_histogram(temp_norm_hist, popt, perr, current_time, synccnt, inputcnt)
#                        update_plot_signal_noise(temp_norm_hist, popt, perr, current_time)

                if j == num_events:
                    break

            del currentlist[0]

    return (hist, current_time, synccnt, inputcnt, signal_error_amplitudes)


logger = logging.getLogger('correlation')
logger.setLevel(logging.DEBUG)
# create console handler and set level to INFO
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create file handler which logs even debug messages
fh = logging.FileHandler('correlation.log', mode='w')
fh.setLevel(logging.DEBUG)
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
logger.info("Script started")

filename = "CdSe_3d_16h_26m_150i_600s.ptu"

logger.info("Filename: {}".format(filename))

hist_len = 400  # bins
bin_width = 1e-9  # in seconds
hist_time = (np.linspace(-hist_len//2+1, hist_len//2, num=hist_len)-1)*bin_width # time in s

# get header with all the information about the file
header = ptu_parser.parse_header(filename)
num_events = header['TTResult_NumberOfRecords']
resolution = header['MeasDesc_GlobalResolution']
exp_time = header['TTResult_StopAfter']*1e-3  # in s

logger.info("Number of events: {}".format(num_events))

# if the header is wrong, fix it
if num_events == 0:
    logger.info('Fxing number of events in header')
    file_size = os.path.getsize(filename)
    data_size = file_size - header['end_header_pos']
    num_events = int(data_size/header['TTResultFormat_BitsPerRecord'])
    header['TTResult_NumberOfRecords'] = num_events
    new_items = {'TTResult_NumberOfRecords': num_events}
    ptu_parser.fix_header(filename, new_items)
    logger.info("New number of events: {}".format(num_events))

logger.info("Resolution: {} s".format(resolution))
logger.info("Measurement time: {} s".format(exp_time))

# prepare reading data file
with open(filename, "rb") as data_file:
    # skip header
    data_file.seek(header['end_header_pos'], os.SEEK_SET)
    # calculate correlation
    (hist, total_time,
     synccnt, inputcnt, amplitudes) = calculate_correlation(data_file, num_events,
                                                            bin_width, hist_len, resolution)
    logger.info("Finished calculating correlation.")


# normalize histogram
totalcnt = synccnt + inputcnt
norm_factor = synccnt * inputcnt / total_time * bin_width
norm_hist = np.array(hist)/norm_factor
full_hist = np.array(np.column_stack((hist_time, norm_hist)))

# Get final signal amplitude and standard deviation, and noise amplitude
signal_data = (norm_hist[hist_len//2:] + norm_hist[:hist_len//2][::-1])/2-1
popt, pcov = curve_fit(exp_decay_fun,
                       hist_time[hist_len//2:], signal_data,
                       p0=(np.mean(signal_data[:10]), 1/(10*bin_width)))
perr = np.sqrt(np.diag(pcov))
signal_amplitude = popt[0]
signal_amplitude_std = perr[0]
signal_decay_rate = popt[1]
signal_decay_std = perr[1]
background_mean = np.mean(signal_data[-hist_len//4:])
noise_amp = np.std(signal_data[-hist_len//4:])

# Plot signal and noise amplitudes as a function of time
signal_noise_amplitudes = np.array(amplitudes)
#crossing_time = plot_signal_noise(signal_noise_amplitudes)

# log some info to file
logger.info("Counts on SYNC: {}.".format(synccnt))
logger.info("Counts on INPUT: {}.".format(inputcnt))
logger.info("Average counts on SYNC: {:.0f} cps.".format(synccnt/exp_time))
logger.info("Average counts on INPUT: {:.0f} cps.".format(inputcnt/exp_time))
logger.info("Ratio INPUT/SYNC: {:.1f}.".format(inputcnt/synccnt))
logger.info("Signal amplitude\u00B1std: {:.3f}\u00B1{:.2f}.".format(signal_amplitude, signal_amplitude_std))
logger.info("Signal decay lifetime\u00B1std: {:.3g}\u00B1{:.1g} s.".format(1/signal_decay_rate, signal_decay_std/signal_decay_rate**2))
logger.info("Background mean\u00B1noise: {:.3f}\u00B1{:.3f}.".format(background_mean, noise_amp))
#logger.info("Crossing time: {:.2f} s.".format(crossing_time))
logger.info('Experimental time {:.1f} s.'.format(total_time))

# write results to file
with open('correlation_histogram.txt', 'wt') as output_file:
    output_file.write('# Filename: {}\n'.format(filename))
    output_file.write('# time (seconds), normalized histogram\n')
    for bin_time, hist_count in full_hist:
        output_file.write("{:e}\t{:f}\n".format(bin_time, hist_count))
# write results to file
with open('correlation_signal_and_noise_amplitude.txt', 'wt') as output_file:
    output_file.write('# Filename: {}\n'.format(filename))
    output_file.write('# Signal time (s), Signal amplitude, Signal STD, Background level, Noise Amplitude\n')
    for signal_time, signal_amp, signal_std, background_mean, noise_amp in signal_noise_amplitudes:
        output_file.write("{:e}\t{:f}\t{:f}\t{:f}\n".format(signal_time, signal_amp, signal_std, background_mean, noise_amp))

# finish script
logger.info("Total time: {:.1f} s.".format(time.time() - start_time))
logging.shutdown()

#fig, ax_histo = plt.subplots(nrows=1, ncols=1)
#mng = plt.get_current_fig_manager()
#mng.window.showMaximized()
#mng.window.activateWindow()
#mng.window.raise_()
#xpos, ypos, width, height = fig.canvas.manager.window.geometry().getRect()
#fig.canvas.manager.window.move(xpos-1.1*width//2, ypos)
#update_plot_histogram = setup_plot_histogram(ax_histo, hist_len, bin_width)
##update_plot_signal_noise = setup_plot_signal_noise(ax_signal_noise, hist_len)
#plt.tight_layout()
#update_plot_histogram(norm_hist, popt, perr, total_time, synccnt, inputcnt)
##update_plot_signal_noise(norm_hist, popt, perr, total_time)

def plot_histogram(histogram, figure=None):
    '''Plot the histogram.'''
    # plot histogram
    fig = figure or plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(histogram[:, 0], histogram[:, 1], '.')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Correlation')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)
    plt.show()

plot_histogram(full_hist)