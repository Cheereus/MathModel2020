import numpy as np
import joblib
from Utils import get_color
import matplotlib.pyplot as plt
from Filter import cb_filter

x = range(1, 11)

# 获取一位受试者的训练数据
# train_data = np.array(joblib.load('data/event_data_by_S.pkl')[0])
y5 = [0.5831, 0.636363636, 0.6, 0.636363636, 0.538461538461538, 0.545454545454545, 0.473684210526315, 0.4375, 0.416666666666666, 0.411764705882352]
plt.plot(x, y5, 'r*-')

my_x_ticks = np.arange(1, 11)
plt.xlabel('Channel Removed')
plt.ylabel('CS')
plt.xticks(my_x_ticks)
plt.legend(loc='best')
plt.grid(axis='x', linestyle='--')
plt.show()
"""
"""