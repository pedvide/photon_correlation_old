# import packages
import numpy as np
import struct
import time
#import sys

import tqdm
import matplotlib.pyplot as plt


# functions
def log(message):
    msg = time.strftime("%X",time.localtime()) + ': ' + str(message)
    print(msg)
    log_file.write(msg + "\n")


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


def calculate_correlation(data_file, bin_width, hist_len):
    j = 0
    overflow = 0
    synccnt = 0  # counts on SYNC
    inputcnt = 0  # counts in INPUT
    currentlist = [[0,2],[0,2]]
    hist = [0]*hist_len

    max_delta_t = 200e-6  # in s, maximum time difference to calculate correlation

    pbar = tqdm.tqdm(total=num_events, desc='Calculating correlation', unit='events')
    while True:
        k = 0
        while currentlist[k][0] - currentlist[0][0] < max_delta_t:
            while len(currentlist) <= k + 1:
                pbar.update(1)
                data_file.seek(684 + 4*j)  # advance to next event
                j += 1
                if j == num_events:
                    break

                event_data = format(struct.unpack('I', data_file.read(4))[0],'032b')
                event_type = event_data[0:7]
                event_time = int(event_data[-25:], 2)  # in ps
                if event_type == '1111111':  # time change
                    overflow += 1
                elif event_type == '1000000':  # photon event at SYNC
                    synccnt += 1
                    # append time in seconds
                    currentlist.append([1e-12*res*(overflow*2**25 + event_time), 0])
                elif event_type == '0000000':  # photon event at INPUT
                    inputcnt += 1
                    # append time in seconds
                    currentlist.append([1e-12*res*(overflow*2**25 + event_time), 1])

            if currentlist[0][1] == 0 and currentlist[k][1] == 1:
                dif = (currentlist[k][0] - currentlist[0][0])/bin_width
                locdif = int(dif) + hist_len//2
                hist[locdif] += 1
            if currentlist[0][1] == 1 and currentlist[k][1] == 0:
                dif = (currentlist[0][0] - currentlist[k][0])/bin_width
                locdif = int(dif) + hist_len//2-1
                hist[locdif] += 1
            k += 1
            if j == num_events:
                break

        del currentlist[0]
        if j >= num_events*0.1:
            break

    pbar.close()  # close the progress bar

    return (hist, overflow, synccnt, inputcnt)


# starting of script
start_time = time.time()
log_file = open("dataloading.log","w",1)
log("Script started")

# prepare reading data file
with open("Pr3+_twodetectors_25000s_sync250cps_input150cps_003.tt2", "rb") as data_file:
    data_file.seek(352)
    res = struct.unpack('d', data_file.read(8))[0]  # ??
    data_file.seek(676)
    num_events = struct.unpack('i', data_file.read(4))[0]  # number of events (including photons and time changes)
    log("Number of events: {}".format(num_events))
    hist_len = 400  # bins
    bin_width = 1e-6  # in seconds

    hist_time = (np.linspace(-hist_len//2+1, hist_len//2, num=hist_len)-1)*bin_width # time in s

    # calculate correlation
    (hist, overflow, synccnt, inputcnt) = calculate_correlation(data_file, bin_width, hist_len)
    log("Finished calculating correlation.")


# normalize histogram
totalcnt = synccnt + inputcnt
total_time = 1e-12*res*overflow*2**25
norm_factor = synccnt * inputcnt / total_time * bin_width
norm_hist = np.array(hist)/norm_factor
full_hist = np.array(np.column_stack((hist_time, norm_hist)))

# log some info to file
log("Counts on SYNC: {}.".format(synccnt))
log("Counts on INPUT: {}.".format(inputcnt))
log('Experimental time {:.1f} s.'.format(total_time))

# write results to file
with open('correlationnew.txt', 'wt') as output_file:
    output_file.write('# time in seconds, counts normalized\n')
    for bin_time, bin_count in full_hist:
        output_file.write("{:e}\t{:f}\n".format(bin_time, bin_count))

# finish script
log("Total time: {:.1f} s.".format(time.time() - start_time))
log_file.close()

# plot histogram
plot_histogram(full_hist)
