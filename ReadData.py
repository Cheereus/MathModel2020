import xlrd
import numpy as np
import joblib


def read_from_xlsx(filePath):
    f = xlrd.open_workbook(filePath)
    sheets = f.sheets()
    sheet_names = f.sheet_names()
    return sheets, sheet_names


def read_data_by_sheets(filePath, data_type='data'):
    data = []
    data_sheets, sheet_names = read_from_xlsx(filePath)
    for sheet in data_sheets:
        row = sheet.nrows  # 行数
        col = sheet.ncols  # 列数
        sheet_data = []
        for x in range(row):
            sheet_data.append(sheet.row_values(x))
        data.append(sheet_data)

    if data_type != 'event':
        data = np.array(data)
    else:
        data = np.array(data, dtype='int64')
        print(data)
    print(data.shape, data.dtype, sheet_names)
    return data


train_data_by_S = []
train_event_by_S = []
test_data_by_S = []
test_event_by_S = []

# 遍历读取数据文件
for i in range(1, 6):
    train_data_path = 'data/S' + str(i) + '/S' + str(i) + '_train_data.xlsx'
    train_event_path = 'data/S' + str(i) + '/S' + str(i) + '_train_event.xlsx'
    test_data_path = 'data/S' + str(i) + '/S' + str(i) + '_test_data.xlsx'
    test_event_path = 'data/S' + str(i) + '/S' + str(i) + '_test_event.xlsx'

    train_data_by_S.append(read_data_by_sheets(train_data_path, 'data'))
    train_event_by_S.append(read_data_by_sheets(train_event_path, 'event'))
    test_data_by_S.append(read_data_by_sheets(test_data_path, 'data'))
    test_event_by_S.append(read_data_by_sheets(test_event_path, 'event'))


joblib.dump(train_data_by_S, 'data/train_data_by_S.pkl')
joblib.dump(train_event_by_S, 'data/train_event_by_S.pkl')
joblib.dump(test_data_by_S, 'data/test_data_by_S.pkl')
joblib.dump(test_event_by_S, 'data/test_event_by_S.pkl')
