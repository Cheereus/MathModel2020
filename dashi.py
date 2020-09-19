import xlrd
import numpy as np
import joblib
from DimensionReduction import t_SNE, get_pca, get_normalize
from Utils import get_color, draw_scatter, draw_scatter3d


def read_from_xlsx(filePath):
    f = xlrd.open_workbook(filePath)
    sheets = f.sheets()
    sheet_names = f.sheet_names()
    return sheets, sheet_names


def read_data_by_sheets(filePath):
    data = []
    data_sheets, sheet_names = read_from_xlsx(filePath)
    for sheet in data_sheets:
        row = sheet.nrows  # 行数
        col = sheet.ncols  # 列数
        sheet_data = []
        for x in range(row):
            sheet_data.append(sheet.row_values(x))
        data.append(sheet_data)

    data = np.array(data)
    print(data.shape, data.dtype, sheet_names)
    return data


data1 = read_data_by_sheets('data/2222222.xlsx')[0][1:, 1:].T

print(data1.shape)

# dimension reduction
# t-SNE
# dim_data, ratio, result = get_pca(data1, c=20, with_normalize=False)
# print(sum(ratio))
dim_data = t_SNE(data1, perp=5, with_normalize=True)
print(len(dim_data))
# get two coordinates
x = [i[0] for i in dim_data]
y = [i[1] for i in dim_data]
z = [i[2] for i in dim_data]
# get color list based on labels
default_colors = ['r', 'b']
colors = get_color([1] * len(dim_data), default_colors)
draw_scatter3d(x, y, z, [1] * len(dim_data), colors)
