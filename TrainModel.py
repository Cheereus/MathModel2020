import numpy as np
import joblib
from DimensionReduction import t_SNE, get_pca, get_normalize
from Utils import get_color, draw_scatter, draw_scatter3d
from Clustering import k_means, knn
from sklearn.model_selection import cross_val_score, LeaveOneOut
from SupportVectorMachine import svm_cross_validation
from sklearn import svm
from Distance import cosine_matrix

char_names = ['B', 'D', 'G', 'L', 'O', 'Q', 'S', 'V', 'Z', '4', '7', '9']

char_labels = []

for i in range(12):
    for j in range(60):
        char_labels.append(char_names[i])

char_labels = np.array(char_labels)

# 获取一位受试者的训练数据
train_data = np.array(joblib.load('data/event_data_by_S.pkl')[0])
train_event = np.array(joblib.load('data/event_labels_by_S.pkl')[0])


reshaped_data = []

# 降采样并拉平，计算相似度矩阵
for i in range(len(train_data)):
    item = []
    for column in train_data[i].T:
        item.append([column[i] for i in range(len(column)) if i % 8 == 0])
    reshaped_data.append(np.array(item).T.reshape(500,))

reshaped_data = cosine_matrix(np.array(reshaped_data))

print(reshaped_data.shape)
"""

# dimension reduction
# t-SNE
# dim_data, ratio, result = get_pca(reshaped_data, c=2, with_normalize=True)
# print(ratio)
dim_data = t_SNE(reshaped_data, perp=5, with_normalize=True)
# get two coordinates
x = [i[0] for i in dim_data]
y = [i[1] for i in dim_data]
z = [i[2] for i in dim_data]
# get color list based on labels
default_colors = ['r', 'b']
colors = get_color(train_event, default_colors)
draw_scatter3d(x, y, z, train_event, colors)


# PCA
# dim_data, ratio, result = get_pca(reshaped_data, c=2, with_normalize=True)
# print(ratio)



loo = LeaveOneOut()
correct = 0
for train, test in loo.split(train_data):
    model = knn(train_data[train], train_event[train], 3)
    labels_predict = model.predict(train_data[test])
    if labels_predict == train_event[test]:
        correct += 1
print(correct / len(train_data))
loo = LeaveOneOut()
correct = 0
for train, test in loo.split(train_data):
    model = svm.SVC(C=1, gamma=0.001, degree=3, kernel='poly', decision_function_shape='ovo')
    model.fit(train_data[train], train_event[train])
    labels_predict = model.predict(train_data[test])
    if labels_predict == train_event[test]:
        correct += 1
    print(correct, '/', test[0] + 1)
print(correct / len(train_data))
"""

params = {
    'C': 1,
    'gamma': 0.001,
    'degree': 3,
    'kernel': 'rbf',
    'decision_function_shape': 'ovo',
    'class_weight': 'balanced'
}

# normalized_data = get_normalize(train_data)

print(svm_cross_validation(reshaped_data, train_event, s=10, params=params))

model = svm.SVC(C=1, gamma=0.001, degree=3, kernel='linear')
model.fit(reshaped_data, train_event)

joblib.dump(model, 'model/svm.pkl')
