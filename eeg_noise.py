import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import csv
from scipy.signal import periodogram
import numpy as np

data_dir = 'E:/brains/eeg_meditation/'

subj_1 = 'EC.csv'
subj_2 = 'EO.csv'

subj_1_reader = csv.reader(open(data_dir + subj_1))
subj_2_reader = csv.reader(open(data_dir + subj_2))

time_series_1 = np.asarray(list(subj_1_reader)[1:], dtype='float32')[:, :]
time_series_2 = np.asarray(list(subj_2_reader)[1:], dtype='float32')[:119984, :]

print(time_series_1.shape, time_series_2.shape)

n_leads = time_series_1.shape[-1]

average_power_diff = np.zeros((time_series_1.shape[0] // 2 + 1))

for i in range(n_leads):
    freq_1, pxx_1 = periodogram(time_series_1[:, i], fs=1000)
    freq_2, pxx_2 = periodogram(time_series_2[:, i], fs=1000)

    power_diff = pxx_1 / pxx_2

    plt.loglog(power_diff)

    average_power_diff += power_diff
    # plt.loglog(freq_1, pxx_1)
    # plt.loglog(freq_2, pxx_2)

# plt.loglog(average_power_diff, color='blue')

plt.xlabel('frequency', fontsize=20)
plt.ylabel('noise amplitude', fontsize=20)

frame1 = plt.gca()
for xlabel_i in frame1.axes.get_xticklabels():
    xlabel_i.set_visible(False)
    xlabel_i.set_fontsize(0.0)

plt.tight_layout()
plt.savefig('E:/brains/eeg_meditation/channel_difference.png', dpi=500)