import numpy as np
import joblib
from Clustering import knn
from sklearn.model_selection import LeaveOneOut
from SupportVectorMachine import svm_cross_validation
from sklearn import svm

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
reshaped_label = []

DATA_SIZE = 800

# 降采样并拉平，计算相似度矩阵
for i in range(len(train_data)):
    item0, item1, item2, item3, item4 = [], [], [], [], []
    for column in train_data[i].T:
        item0.append([column[i] for i in range(len(column)) if i % 5 == 0])
        item1.append([column[i] for i in range(len(column)) if i % 5 == 1])
        item2.append([column[i] for i in range(len(column)) if i % 5 == 2])
        item3.append([column[i] for i in range(len(column)) if i % 5 == 3])
        item4.append([column[i] for i in range(len(column)) if i % 5 == 4])
    reshaped_data.append(np.array(item0).T.reshape(DATA_SIZE,))
    reshaped_label.append(train_event[i])

    if train_event[i] == 1:
        reshaped_data.append(np.array(item1).T.reshape(DATA_SIZE, ))
        reshaped_data.append(np.array(item2).T.reshape(DATA_SIZE, ))
        reshaped_data.append(np.array(item3).T.reshape(DATA_SIZE, ))
        reshaped_data.append(np.array(item4).T.reshape(DATA_SIZE, ))
        reshaped_label.append(train_event[i])
        reshaped_label.append(train_event[i])
        reshaped_label.append(train_event[i])
        reshaped_label.append(train_event[i])

reshaped_data = np.array(reshaped_data)
reshaped_label = np.array(reshaped_label)

# 打乱数据
index = [i for i in range(len(reshaped_data))]
np.random.shuffle(index)
reshaped_data = reshaped_data[index]
reshaped_label = reshaped_label[index]
print(reshaped_data.shape)

# cosine_distance = cosine_matrix(np.array(reshaped_data))


"""

# dimension reduction
# t-SNE
dim_data, ratio, result = get_pca(reshaped_data, c=20, with_normalize=False)
print(sum(ratio))
# dim_data = t_SNE(reshaped_data, perp=5, with_normalize=False)
# get two coordinates
x = [i[0] for i in dim_data]
y = [i[1] for i in dim_data]
z = [i[2] for i in dim_data]
# get color list based on labels
default_colors = ['r', 'b']
colors = get_color(train_event, default_colors)
draw_scatter3d(x, y, z, train_event, colors)
print(reshaped_data.shape)


# PCA
# dim_data, ratio, result = get_pca(reshaped_data, c=2, with_normalize=True)
# print(ratio)



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
    model = knn(reshaped_data[train], reshaped_label[train], 5)
    labels_predict = model.predict(reshaped_data[test])
"""

params = {
    'C': 1,
    'gamma': 0.001,
    'degree': 3,
    'kernel': 'linear',
    'decision_function_shape': 'ovo',
    'class_weight': 'balanced'
}

# reshaped_data = dim_data

loo = LeaveOneOut()
correct = 0
for train, test in loo.split(reshaped_data):
    model = svm.SVC(C=1, gamma=0.001, degree=3, kernel='linear')
    model.fit(reshaped_data[train], reshaped_label[train])
    labels_predict = model.predict(reshaped_data[test])
    if labels_predict == reshaped_label[test]:
        correct += 1
print(correct / len(reshaped_data))

print(svm_cross_validation(reshaped_data, reshaped_label, s=10, params=params))

model = svm.SVC(C=1, gamma=0.001, degree=3, kernel='linear')
model.fit(reshaped_data, reshaped_label)
joblib.dump(model, 'model/svm4.pkl')

model = knn(reshaped_data, reshaped_label, 5)
joblib.dump(model, 'model/knn.pkl')
