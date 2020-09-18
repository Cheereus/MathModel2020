import xlrd
import numpy as np
import joblib
from Utils import get_color, draw_scatter, draw_scatter3d
from DimensionReduction import t_SNE, get_pca, get_normalize, Isometric
from Clustering import knn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from Metrics import accuracy


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

# 降维 PCA 和 t-SNE
# PCA
# dim_data, ratio, result = get_pca(sleep_data, c=3, with_normalize=True)
# print(ratio)
# t-SNE
# dim_data = t_SNE(sleep_data, perp=50, with_normalize=True)
# Iso map
dim_data = Isometric(sleep_data, n_neighbors=30, n_components=3)

# 数据集切分
train_data, test_data, train_label, test_label = train_test_split(dim_data, sleep_labels, test_size=0.3, shuffle=True)
# knn 模型
knn_model = KNeighborsClassifier(n_neighbors=8)
knn_model.fit(train_data, train_label)
predict_label = knn_model.predict(test_data)
acc = accuracy(predict_label, test_label)
print(acc)

# 交叉验证
# scores = cross_val_score(knn_model, dim_data, sleep_labels, cv=10)
# print(scores)

# 绘图
# get coordinates
x = [i[0] for i in dim_data]
y = [i[1] for i in dim_data]
z = [i[2] for i in dim_data]
# get color list based on labels
default_colors = ['r', 'b', 'g', 'c', 'm']
colors = get_color(sleep_labels, default_colors)

print('Drawing...')
draw_scatter3d(x, y, z, sleep_labels, colors)

