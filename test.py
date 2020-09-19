import numpy as np
import joblib
from Utils import get_color
import matplotlib.pyplot as plt
from Filter import cb_filter

# 获取一位受试者的训练数据
train_data = np.array(joblib.load('data/event_data_by_S.pkl')[0])

print(train_data.shape)

x = range(0, 800, 4)
y = np.array(train_data[0]).T[0]

plt.plot(x, y, linestyle='-', label='P300')
y = np.array(train_data[6]).T[0]
plt.plot(x, y, linestyle='--', label='non P300')
my_x_ticks = np.arange(0, 840, 40)
plt.xticks(my_x_ticks)
plt.legend(loc='best')
plt.grid()
plt.show()
"""
"""