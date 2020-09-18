import numpy as np
import joblib
from DimensionReduction import get_normalize
from Distance import cosine_matrix

# 获取一位受试者的测试数据
test_data = np.array(joblib.load('data/test_event_data_by_S.pkl')[0])
# test_data = test_data.reshape((540, 4000))
reshaped_data = []

# 降采样并拉平，计算相似度矩阵
for i in range(len(test_data)):
    item = []
    for column in test_data[i].T:
        item.append([column[i] for i in range(len(column)) if i % 4 == 0])

    reshaped_data.append(np.array(item).T.reshape(380,))

reshaped_data = np.array(reshaped_data)
cosine_distance = cosine_matrix(reshaped_data)
model = joblib.load('model/svm.pkl')

predict_labels = model.predict(reshaped_data)

print(predict_labels[48:60])
