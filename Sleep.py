import xlrd
import numpy as np
from Utils import get_color, draw_scatter
from DimensionReduction import get_pca
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from Metrics import accuracy, ARI
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

# PCA
dim_data, ratio, result = get_pca(sleep_data, c=2, with_normalize=False)
# print(ratio)

# 绘图13
x = [i[0] for i in dim_data]
y = [i[1] for i in dim_data]
default_colors = ['r', 'b', 'g', 'c', 'm']
colors = get_color(sleep_labels, default_colors)
print('Drawing...')
draw_scatter(x, y, sleep_labels, colors)

default_colors = [[0, 0.8, 1], [0, 0.5, 0.5], [0.2, 0.8, 0.8], [0.2, 0.4, 1], [0.6, 0.8, 1], [1, 0.6, 0.8],
                  [0.8, 0.6, 1], [1, 0.8, 0.6], [1, 0, 0], [0, 1, 0]]
for j in range(1, 10):

    acc_arr = []

    # 数据集切分
    train_data, test_data, train_label, test_label = train_test_split(sleep_data, sleep_labels, test_size=(j / 10), shuffle=True)

    # knn 模型遍历训练
    for i in range(1, 51):
        knn_model = KNeighborsClassifier(n_neighbors=i)
        knn_model.fit(train_data, train_label)
        predict_label = knn_model.predict(test_data)
        acc = accuracy(predict_label, test_label)
        ari = ARI(test_label, predict_label)
        print(ari)
        acc_arr.append(acc)

    x = range(1, 51)
    plt.plot(x, acc_arr, '*-', color=default_colors[j - 1], label='test percent:'+str(j / 10))

# 图14
my_x_ticks = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
plt.xticks(my_x_ticks)
plt.legend(loc='best')
plt.grid(axis='x', linestyle='--')
plt.xlabel('k')
plt.ylabel('accuracy')
plt.show()
