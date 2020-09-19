import numpy as np
import joblib
from sklearn import svm
from scipy.spatial.distance import pdist


def cos_sim(vector_a, vector_b):

    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    d = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / d
    sim = 0.5 + 0.5 * cos
    return sim


# 获取第 idx 位受试者的训练数据
def get_train_data(idx, channel):

    train_data = np.array(joblib.load('data/event_data_by_S.pkl')[idx])
    train_event = np.array(joblib.load('data/event_labels_by_S.pkl')[idx])

    reshaped_data = []
    reshaped_label = []

    # 降采样并拉平，并对 channel 进行取舍
    for i in range(len(train_data)):
        item0, item1, item2, item3, item4 = [], [], [], [], []
        for column in train_data[i].T:
            item0.append([column[i] for i in range(len(column)) if i % 5 == 0])
            item1.append([column[i] for i in range(len(column)) if i % 5 == 1])
            item2.append([column[i] for i in range(len(column)) if i % 5 == 2])
            item3.append([column[i] for i in range(len(column)) if i % 5 == 3])
            item4.append([column[i] for i in range(len(column)) if i % 5 == 4])
        DATA_SIZE = np.array(item0)[channel].shape[0] * np.array(item0)[channel].shape[1]
        reshaped_data.append(np.array(item0)[channel].T.reshape(DATA_SIZE,))
        reshaped_label.append(train_event[i])

        if train_event[i] == 1:
            reshaped_data.append(np.array(item1)[channel].T.reshape(DATA_SIZE, ))
            reshaped_data.append(np.array(item2)[channel].T.reshape(DATA_SIZE, ))
            reshaped_data.append(np.array(item3)[channel].T.reshape(DATA_SIZE, ))
            reshaped_data.append(np.array(item4)[channel].T.reshape(DATA_SIZE, ))
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

    return reshaped_data, reshaped_label


# 获取第 idx 位受试者的测试数据
def get_test_data(idx, channel):

    test_data = np.array(joblib.load('data/test_event_data_by_S.pkl')[idx])
    reshaped_data = []

    # 降采样并拉平，并对 channel 进行取舍
    for i in range(len(test_data)):
        item = []
        for column in test_data[i].T:
            item.append([column[i] for i in range(len(column)) if i % 5 == 0])
        DATA_SIZE = np.array(item)[channel].shape[0] * np.array(item)[channel].shape[1]
        reshaped_data.append(np.array(item)[channel].T.reshape(DATA_SIZE,))

    reshaped_data = np.array(reshaped_data)
    return reshaped_data


def calc_loss(predict_vec):
    loss = 0

    # M, F, 5, 2, I 对应的向量
    characters = [
        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    ]
    for col in range(5):
        for row in range(12):
            if characters[col][row] != predict_vec[col][row]:
                # loss += 1
                X = np.vstack([characters[col], predict_vec[col]])
                d2 = np.corrcoef(X)[0][1]
                loss += d2
        # loss += cos_sim(characters[col], predict_vec[col])

    return loss


channel_now = [_ for _ in range(20)]
# del channel_now[18]

for c in channel_now:
    selected_channel = [_ for _ in range(20)]
    del selected_channel[c]
    train_data_by_S, train_label_by_S = get_train_data(0, selected_channel)
    test_data_by_S = get_test_data(0, selected_channel)
    model = svm.SVC(C=1, gamma=0.001, degree=3, kernel='linear')
    model.fit(train_data_by_S, train_label_by_S)
    predict_labels = model.predict(test_data_by_S)

    labels_5 = []

    # 只考虑前五个字符
    for char_idx in range(5):
        labels_5.append(predict_labels[char_idx * 12:(char_idx + 1) * 12])

    current_loss = calc_loss(labels_5)
    print('Remove channel:', c+1, 'loss:', current_loss)
