import numpy as np
from scipy.stats import zscore


def process_signal(signal, VR, VM):
    signal = signal.copy()
    window_size = 5
    num_windows = len(signal) // window_size

    for w in range(num_windows):
        start_idx = w * window_size
        end_idx = start_idx + window_size
        local_std = np.std(signal[start_idx:end_idx])
        threshold = 1 * local_std
        window_mean = np.mean(signal[start_idx:end_idx])

        for i in range(start_idx + 1, end_idx):
            if abs(signal[i] - signal[i - 1]) > threshold:
                signal[i] = window_mean

    avg_window_amplitude = np.zeros(num_windows)
    for w in range(num_windows):
        start_idx = w * window_size
        end_idx = start_idx + window_size
        avg_window_amplitude[w] = np.mean(signal[start_idx:end_idx])

    for w in range(num_windows):
        start_idx = w * window_size
        end_idx = start_idx + window_size
        signal[start_idx:end_idx] = avg_window_amplitude[w]

    stR = np.std(signal[VR] - np.mean(signal[VR]))
    stM = np.std(signal[VM] - np.mean(signal[VM]))
    MR = np.mean(signal[VR])
    MM = np.mean(signal[VM])

    num_sections = len(VR) // 5
    for i in range(num_sections):
        idx_vr = VR[i * 5:(i + 1) * 5]
        idx_vm = VM[i * 5:(i + 1) * 5]
        if abs(signal[idx_vr[0]] - MR) > 1.5 * stR:
            signal[idx_vr] = MR
        if abs(signal[idx_vm[0]] - MM) > 1.5 * stM:
            signal[idx_vm] = MM
    return signal


def fill_outliers(segment):
    z_scores = np.abs(zscore(segment))
    threshold = 3
    outlier_indices = np.where(z_scores > threshold)[0]
    if len(outlier_indices) > 0:
        segment_no_outliers = segment.copy()
        segment_no_outliers[outlier_indices] = np.nan
        nans = np.isnan(segment_no_outliers)
        not_nans = ~nans
        if np.sum(not_nans) >= 2:
            segment_no_outliers[nans] = np.interp(
                nans.nonzero()[0], not_nans.nonzero()[0], segment_no_outliers[not_nans]
            )
        else:
            segment_no_outliers[nans] = np.nanmean(segment_no_outliers)
        return segment_no_outliers
    return segment
