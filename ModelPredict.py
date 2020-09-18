import numpy as np
import joblib
from DimensionReduction import get_normalize

# 获取一位受试者的测试数据
test_data = np.array(joblib.load('data/test_event_data_by_S.pkl')[0])
# test_data = test_data.reshape((540, 4000))
reshaped_data = []

for i in range(len(test_data)):
    reshaped_data.append(test_data[i].reshape(2000,))

reshaped_data = np.array(reshaped_data)
model = joblib.load('model/svm.pkl')

predict_labels = model.predict(reshaped_data)

print(predict_labels)
