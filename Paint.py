import numpy as np
import joblib
from Utils import get_color
import matplotlib.pyplot as plt
from DimensionReduction import get_normalize

# 获取一位受试者的训练数据
train_data = np.array(joblib.load('data/event_data_by_S_single.pkl')[0])
train_event = np.array(joblib.load('data/event_labels_by_S_single.pkl')[0])

non_p300_wave_avg = np.zeros(200)
p300_wave_avg = np.zeros(200)

for i in range(60):
    wave = np.array(get_normalize(train_data[i])).T[0]
    if train_event[i] == 0:
        non_p300_wave_avg = non_p300_wave_avg + wave
    else:
        p300_wave_avg = p300_wave_avg + wave

non_p300_wave_avg = non_p300_wave_avg / 50
p300_wave_avg = p300_wave_avg / 10

print(train_data.shape)

print(train_data.shape, train_event.shape)

default_colors = ['r', 'b']
colors = get_color(range(8))

x = range(0, 800, 4)

plt.plot(x, non_p300_wave_avg, c='r', label='non p300')
plt.plot(x, p300_wave_avg, c='b', label='p300')
plt.xlabel('time(ms)')
my_x_ticks = np.arange(0, 200, 50)
plt.grid()
plt.legend(loc='best')
plt.show()





