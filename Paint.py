from Utils import get_color, draw_scatter
import numpy as np
import joblib
from Utils import get_color
import matplotlib.pyplot as plt

# 获取一位受试者的训练数据
train_data = np.array(joblib.load('data/train_data_by_S.pkl')[0][0])
train_event = np.array(joblib.load('data/train_event_by_S.pkl')[0][0])

print(train_data.shape, train_event.shape)

default_colors = ['r', 'b']
colors = get_color(range(8))

x = range(3125)

plt.plot(x, train_data.T[8])

plt.show()





