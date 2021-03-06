import numpy as np
import joblib
from DimensionReduction import get_normalize
from Filter import cb_filter

# 针对 event 中的每条标签 从数据中提取出持续时间 800ms 的单次试验数据段
# 即单次试验从刺激开始算起, 刺激开始后的 800ms 结束
# 采样频率 250Hz 即每个采样点间隔 4 ms 每条标签需要取 200 条采样数据
# 针对上述每条样本, 根据行列和字符的对应关系, 标注为 0 和 1

labels = np.array([
    ['A', 'B', 'C', 'D', 'E', 'F'],
    ['G', 'H', 'I', 'J', 'K', 'L'],
    ['M', 'N', 'O', 'P', 'Q', 'R'],
    ['S', 'T', 'U', 'V', 'W', 'X'],
    ['Y', 'Z', '1', '2', '3', '4'],
    ['5', '6', '7', '8', '9', '0'],
])

char_names = ['B', 'D', 'G', 'L', 'O', 'Q', 'S', 'V', 'Z', '4', '7', '9']

train_data_by_S = joblib.load('data/train_data_by_S.pkl')
train_event_by_S = joblib.load('data/train_event_by_S.pkl')


# 根据字符和行列标签, 标注数据为 0 或 1
def get_true_label(name, label):
    label = int(label)
    positionX, positionY = np.where(labels == name)
    x = positionX[0] + 1
    y = positionY[0] + 7
    if label == x or label == y:
        return 1
    else:
        return 0


event_labels_by_S = []
event_data_by_S = []

for i in range(5):
    train_data = train_data_by_S[i]
    train_event = train_event_by_S[i]

    event_labels = []
    event_data_s = []

    for j in range(12):
        char_name = char_names[j]
        event = train_event[j]
        event_data = np.zeros((200, 20))

        for k in range(1, 13):
            event_label = get_true_label(char_name, k)
            event_labels.append(event_label)
            event_data = np.zeros((200, 20))

            for idx in range(1, len(event)):  # 0 - 66
                # 剔除实验开始结束时的标记 event 求五个轮次中同一行列的均值
                if event[idx][0] == k:
                    # 开始和结束时间在数据表中的索引
                    event_start = event[idx][1] - 1
                    event_end = event_start + 200
                    # 获取对应时间段的采样数据
                    event_data = event_data + get_normalize(np.array(cb_filter(train_data[j])[event_start:event_end]))

            event_data = event_data / 5

            event_data_s.append(event_data)
            print(len(event_data_s))

    event_data_by_S.append(event_data_s)
    event_labels_by_S.append(event_labels)

joblib.dump(event_data_by_S, 'data/event_data_by_S.pkl')
joblib.dump(event_labels_by_S, 'data/event_labels_by_S.pkl')

print(len(event_data_by_S[0]))
print(event_labels_by_S[0])
