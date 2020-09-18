import numpy as np
import joblib
from Utils import get_color
import matplotlib.pyplot as plt

# 获取一位受试者的训练数据
train_data = np.array(joblib.load('data/event_data_by_S.pkl')[0])
train_event = np.array(joblib.load('data/event_labels_by_S.pkl')[0])

y = train_data[1].T[0]

print(y.shape)

x = range(len(y))

plt.plot(x, y)
my_x_ticks = np.arange(0, 100, 10)
plt.xticks(my_x_ticks)
plt.grid()
plt.show()
