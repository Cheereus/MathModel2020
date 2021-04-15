import numpy as np
import joblib
from Utils import get_color
import matplotlib.pyplot as plt
from Filter import cb_filter

# 获取一位受试者的训练数据
train_data = np.array(joblib.load('data/train_data_by_S.pkl')[0])

print(np.array(train_data[0]).T[0])

x = range(0, 3125)
y = cb_filter(np.array(train_data[0]).T[0])
plt.plot(x, y, linestyle='-', label='after', c='k')
# y = np.array(train_data[6]).T[0]
# y = np.array(train_data[0]).T[0]
# plt.plot(x, y, linestyle='-', label='before', c='k')
my_x_ticks = np.arange(0, 3500, 100)
plt.xticks(my_x_ticks)
# plt.legend(loc='best')
plt.grid()
plt.show()
"""
"""