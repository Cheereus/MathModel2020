import numpy as np
import joblib
import scipy.signal as signal


def cb_filter(data):

    fs = 250.0  # Sample frequency (Hz)
    f0 = 50.0  # Frequency to be removed from signal (Hz)
    Q = 30.0  # Quality factor
    w0 = f0/(fs/2)  # Normalized Frequency
    # Design notch filter
    b, a = signal.cheby1(8, 1, 0.08, btype='low')

    return np.array(signal.filtfilt(b, a, np.array(data).T)).T


def custom_filter(data):
    row_data = np.array(data)
    new_row_data = []

    for row in row_data:
        row_mean = np.sum(row) / len(row)
        # print(row_mean)
        new_row_data.append(row_mean)

    return new_row_data


def iir_notch_filter(data):

    fs = 250.0  # Sample frequency (Hz)
    f0 = 50.0  # Frequency to be removed from signal (Hz)
    Q = 30.0  # Quality factor
    w0 = f0/(fs/2)  # Normalized Frequency
    # Design notch filter
    b, a = signal.iirnotch(w0, Q)

    return np.array(signal.filtfilt(b, a, np.array(data).T)).T


"""
    for row in row_data:
        col = len(row)
        new_row = [row[i] for i in range(col) if i % 8 == 0]
        p_80 = np.sort(new_row)[int(len(new_row) * 0.8)]
        p_max = np.sort(new_row)[-1]
        # print(p_90)
        for i in range(len(new_row)):
            if new_row[i] < p_80:
                new_row[i] = 0
            else:
                new_row[i] = (new_row[i] - p_80) / (p_max - p_80)
        new_row_data.append(new_row)
"""
