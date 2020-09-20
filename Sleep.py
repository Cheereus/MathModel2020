import xlrd
import numpy as np
import joblib
from Utils import get_color, draw_scatter, draw_scatter3d
from DimensionReduction import t_SNE, get_pca, get_normalize, Isometric
from Clustering import knn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from Metrics import accuracy, F1, ARI
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.manifold import locally_linear_embedding, MDS


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
# sleep_data = get_normalize(sleep_data)

# 降维 PCA 和 t-SNE
# PCA
dim_data, ratio, result = get_pca(sleep_data, c=2, with_normalize=False)
# print(ratio)
# t-SNE
# dim_data = t_SNE(sleep_data, perp=50, with_normalize=False)
# Iso map
# dim_data = Isometric(sleep_data, n_neighbors=30, n_components=3)
# LLE
# dim_data = locally_linear_embedding(sleep_data, n_neighbors=10, n_components=2)[0]
# MDS
# dim_data = MDS(n_components=2).fit_transform(sleep_data)
# print(dim_data)

# 绘图
# get coordinates
x = [i[0] for i in dim_data]
y = [i[1] for i in dim_data]
# z = [i[2] for i in dim_data]
# get color list based on labels
default_colors = ['r', 'b', 'g', 'c', 'm']
colors = get_color(sleep_labels, default_colors)

print('Drawing...')
# draw_scatter(x, y, sleep_labels, colors)
draw_scatter(x, y, sleep_labels, colors)
"""
"""
default_colors = [[0, 0.8, 1], [0, 0.5, 0.5], [0.2, 0.8, 0.8], [0.2, 0.4, 1], [0.6, 0.8, 1], [1, 0.6, 0.8],
                  [0.8, 0.6, 1], [1, 0.8, 0.6], [1, 0, 0],[0, 1, 0]]

for j in range(1, 10):

    # 数据集切分
    train_data, test_data, train_label, test_label = train_test_split(sleep_data, sleep_labels, test_size=(j / 10), shuffle=True)
    # knn 模型
    acc_arr = []

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

my_x_ticks = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
plt.xticks(my_x_ticks)
plt.legend(loc='best')
plt.grid(axis='x', linestyle='--')
plt.xlabel('k')
plt.ylabel('accuracy')
plt.show()
# 交叉验证
# scores = cross_val_score(knn_model, dim_data, sleep_labels, cv=10)
# print(scores)

"""
C2 = confusion_matrix(test_label, predict_label)
sns.heatmap(C2, annot=True, cmap='YlGnBu', fmt='.20g')
labels_name = [6, 5, 4, 3, 2]
num_local = np.array(range(len(labels_name) + 1))
plt.xticks(num_local, labels_name, rotation=90)    # 将标签印在x轴坐标上
plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.imshow(C2, interpolation='nearest')
plt.show()
"""
