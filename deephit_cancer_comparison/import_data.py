import numpy as np
import pandas as pd

import deephit_cancer_comparison.constants as const


def f_get_Normalization(X, norm_mode):
    """Normalize the input data matrix X according to the selected normalization mode.

    norm_mode: str, either 'standard' (zero mean, unit variance) or 'normal' (min-max normalization)
    """
    num_Patient, num_Feature = X.shape

    if norm_mode == "standard":  # Zero mean unit variance
        for j in range(num_Feature):
            if np.std(X[:, j]) != 0:
                X[:, j] = (X[:, j] - np.mean(X[:, j])) / np.std(X[:, j])
            else:
                X[:, j] = X[:, j] - np.mean(X[:, j])
    elif norm_mode == "normal":  # Min-max normalization
        for j in range(num_Feature):
            X[:, j] = (X[:, j] - np.min(X[:, j])) / (np.max(X[:, j]) - np.min(X[:, j]))
    else:
        raise ValueError("Invalid normalization mode selected!")

    return X


def f_get_fc_mask2(time, label, num_Event, num_Category):
    mask = np.zeros([time.shape[0], num_Event, num_Category])

    for i in range(time.shape[0]):
        if label[i, 0] != 0:
            mask[i, int(label[i, 0] - 1), int(time[i, 0])] = 1
        else:
            mask[i, :, int(time[i, 0] + 1) :] = 1

    return mask


def f_get_fc_mask3(time, meas_time, num_Category):
    """Mask3 is required to calculate the ranking loss (for pair-wise comparison)

    mask5 size is [N, num_Category].
    - For longitudinal measurements:
         1's from the last measurement to the event time (exclusive and inclusive, respectively)
    - For single measurement:
         1's from start to the event time (inclusive)
    """
    mask = np.zeros([np.shape(time)[0], num_Category])

    if isinstance(meas_time, np.ndarray) and np.shape(meas_time)[0] > 0:
        for i in range(np.shape(time)[0]):
            t1 = int(meas_time[i, 0])
            t2 = int(time[i, 0])
            mask[i, (t1 + 1) : (t2 + 1)] = 1

    else:
        for i in range(np.shape(time)[0]):
            t = int(time[i, 0])
            mask[i, : (t + 1)] = 1

    return mask


def import_cohort_data(cancer_type: str = None):
    if not cancer_type:
        raise ValueError("Must provide cancer_type")
    X_datafile = f"X_{cancer_type}.csv"
    y_datafile = f"y_{cancer_type}.csv"

    CANCER_SPECIFIC_DATA_PATH = const.DATA_PATH / "cancer_specific_data" / cancer_type

    X = pd.read_csv(CANCER_SPECIFIC_DATA_PATH / X_datafile)
    y = pd.read_csv(CANCER_SPECIFIC_DATA_PATH / y_datafile)

    label = np.asarray(y[["outcome"]])
    time = np.asarray(y[["time"]])
    data = np.asarray(X)
    data = f_get_Normalization(data, norm_mode="standard")

    num_Category = int(np.max(time) * 1.2)
    num_Event = int(len(np.unique(label)) - 1)

    x_dim = data.shape[1]

    mask1 = f_get_fc_mask2(time, label, num_Event, num_Category)
    mask2 = f_get_fc_mask3(time, -1, num_Category)

    DIM = x_dim
    DATA = (data, time, label)
    MASK = (mask1, mask2)

    return DIM, DATA, MASK
