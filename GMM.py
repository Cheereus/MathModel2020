import xlrd
import numpy as np
from DimensionReduction import get_normalize
from Clustering import knn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from Metrics import accuracy
import matplotlib.pyplot as plt


def read_from_xlsx(filePath):
    f = xlrd.open_workbook(filePath)
    sheets = f.sheets()
    sheet_names = f.sheet_names()
    return sheets, sheet_names


def read_data_by_sheets(filePath):
    data = []
    labels = []
    data_sheets, sheet_names = read_from_xlsx(filePath)
    i = 0
    for sheet in data_sheets:
        row = sheet.nrows  # 行数
        col = sheet.ncols  # 列数
        for j in range(1, row):
            labels.append(sheet_names[i])
            data.append(sheet.row_values(j)[1:5])
        i += 1

    data = np.array(data)
    return data, [i[-2] for i in labels]


sleep_data, sleep_labels = read_data_by_sheets('data/sleep.xlsx')
# 归一化
sleep_data = get_normalize(sleep_data)

default_colors = [[0, 0.8, 1], [0, 0.5, 0.5], [0.2, 0.8, 0.8], [0.2, 0.4, 1], [0.6, 0.8, 1], [1, 0.6, 0.8],
                  [0.8, 0.6, 1], [1, 0.8, 0.6], [1, 0, 0], [0, 1, 0]]

for i in range(5, 15):

    acc_arr = []
    for j in np.arange(0.1, 1, 0.05):

        # 数据集切分
        train_data, test_data, train_label, test_label = train_test_split(sleep_data, sleep_labels, test_size=j, shuffle=True)
        # knn 模型
        knn_model = KNeighborsClassifier(n_neighbors=10)
        knn_model.fit(train_data, train_label)
        predict_label = knn_model.predict(test_data)
        acc = accuracy(predict_label, test_label)
        acc_arr.append(acc)

    x = np.arange(0.1, 1, 0.05)
    plt.plot(x, acc_arr, '*-', color=default_colors[i-5], label='k='+str(i))

my_x_ticks = np.arange(0.1, 1, 0.05)
plt.xticks(my_x_ticks)
plt.legend(loc='best')
plt.grid(axis='x', linestyle='--')
plt.xlabel('test percent')
plt.ylabel('accuracy')
plt.show()
