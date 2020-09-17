import xlrd
import numpy as np
import joblib
from Utils import get_color, draw_scatter, draw_scatter3d
from DimensionReduction import t_SNE, get_pca, get_normalize


def read_from_xlsx(filePath):
    f = xlrd.open_workbook(filePath)
    sheets = f.sheets()
    sheet_names = f.sheet_names()
    return sheets, sheet_names


def read_data_by_sheets(filePath, data_type='data'):
    data = []
    labels = []
    data_sheets, sheet_names = read_from_xlsx(filePath)
    i = 0
    for sheet in data_sheets:
        row = sheet.nrows  # 行数
        col = sheet.ncols  # 列数
        for x in range(1, row):
            labels.append(sheet_names[i])
            data.append(sheet.row_values(x)[1:5])
        i += 1

    if data_type != 'event':
        data = np.array(data)
    else:
        data = np.array(data, dtype='int64')
        print(data)
    print(data.shape, data.dtype, sheet_names)
    return data, labels


sleep_data, sleep_labels = read_data_by_sheets('data/sleep.xlsx', 'data')

sleep_labels = [i[-2] for i in sleep_labels]

print(sleep_data[0], len(sleep_labels))

# dimension reduction
# PCA
# dim_data, ratio, result = get_pca(sleep_data, c=3, with_normalize=True)
# print(ratio)
# t-SNE
dim_data = t_SNE(sleep_data, perp=50, with_normalize=True)
# get two coordinates
x = [i[0] for i in dim_data]
y = [i[1] for i in dim_data]
z = [i[2] for i in dim_data]
# get color list based on labels
default_colors = ['r', 'b', 'g', 'c', 'm']
colors = get_color(sleep_labels, default_colors)

print('Drawing...')
draw_scatter3d(x, y, z, sleep_labels, colors)

