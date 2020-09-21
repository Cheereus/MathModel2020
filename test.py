import numpy as np
import joblib
from Utils import get_color
import matplotlib.pyplot as plt
from Filter import cb_filter

x = np.arange(0.1, 1.1, 0.1)

# 获取一位受试者的训练数据
# train_data = np.array(joblib.load('data/event_data_by_S.pkl')[0])
y1 = [0.416666666666666, 0.4375, 0.538461538461538, 0.538461538461538, 0.6777777, 0.5831, 0.5831, 0.6777777, 0.6777777, 0.6777777]
plt.plot(x, y1, 'c*-', label='10% Labeled data')
y2 = [0.333, 0.545454545454545, 0.538461538461538, 0.538461538461538, 0.6777777, 0.5831, 0.6777777, 0.636363636, 0.735, 0.636363636]
plt.plot(x, y2, 'k*-', label='30% Labeled data')
y3 = [0.333, 0.4375, 0.538461538461538, 0.538461538461538, 0.6777777, 0.5831, 0.6777777, 0.735, 0.8, 0.8]
plt.plot(x, y3, 'g*-', label='90% Labeled data')
y4 = [0.473684210526315, 0.4375, 0.545454545454545, 0.538461538461538, 0.5831, 0.5831, 0.6777777, 0.735, 0.735, 0.735]
plt.plot(x, y4, 'b*-', label='70% Labeled data')
y5 = [0.411764705882352, 0.416666666666666, 0.473684210526315, 0.4375, 0.545454545454545, 0.538461538461538, 0.5831, 0.636363636, 0.6777777, 0.6777777]
plt.plot(x, y5, 'r*-', label='50% Labeled data')

my_x_ticks = np.arange(0.1, 1.1, 0.1)
plt.xlabel('Unlabel data used')
plt.ylabel('Accuracy')
plt.xticks(my_x_ticks)
plt.legend(loc='best')
plt.grid(axis='x', linestyle='--')
plt.show()
"""
"""