import numpy as np
import pandas as pd

import deephit_cancer_comparison.constants as const


def f_get_Normalization(X, norm_mode):
    """Normalize the input data matrix X according to the selected normalization mode.

    norm_mode: str, either 'standard' (zero mean, unit variance) or 'normal' (min-max normalization)
    """
    num_Patient, num_Feature = X.shape

    if norm_mode == "standard":  # Zero mean unit variance
        # Per-column standardisation; constant columns are only re-centred to
        # avoid division by zero.
        for j in range(num_Feature):
            if np.std(X[:, j]) != 0:
                X[:, j] = (X[:, j] - np.mean(X[:, j])) / np.std(X[:, j])
            else:
                X[:, j] = X[:, j] - np.mean(X[:, j])
    elif norm_mode == "normal":  # Min-max normalization
        # Per-column scaling into [0, 1].
        for j in range(num_Feature):
            X[:, j] = (X[:, j] - np.min(X[:, j])) / (np.max(X[:, j]) - np.min(X[:, j]))
    else:
        raise ValueError("Invalid normalization mode selected!")

    return X


def f_get_fc_mask2(time, label, num_Event, num_Category):
    # mask1 (called "mask2" historically): used by the log-likelihood term.
    # Shape (N, num_Event, num_Category).
    #   - Uncensored patient (label != 0): a single 1 at (observed event, time).
    #   - Censored patient:                 1s for all event types at all
    #     times strictly after the censoring time (i.e. patient was alive
    #     past those bins).
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
        # Longitudinal case: mark time bins between last measurement and event.
        for i in range(np.shape(time)[0]):
            t1 = int(meas_time[i, 0])
            t2 = int(time[i, 0])
            mask[i, (t1 + 1) : (t2 + 1)] = 1

    else:
        # Single (baseline) measurement: mark every bin up to and including
        # the event/censoring time.
        for i in range(np.shape(time)[0]):
            t = int(time[i, 0])
            mask[i, : (t + 1)] = 1

    return mask


def import_cohort_data(cancer_type: str = None, split: str = "train"):
    # Load one cancer cohort's (features, labels) split from disk and return
    # everything DeepHit needs: feature dimension, (data/time/label) tuple,
    # and the two masks.
    if not cancer_type or not split:
        raise ValueError("Must provide cancer_type and split")
    X_datafile = f"X_{cancer_type}.csv"
    y_datafile = f"y_{cancer_type}.csv"

    CANCER_SPECIFIC_DATA_PATH = const.DATA_PATH / "cancer_specific_data" / cancer_type / split

    X = pd.read_csv(CANCER_SPECIFIC_DATA_PATH / X_datafile)
    y = pd.read_csv(CANCER_SPECIFIC_DATA_PATH / y_datafile)

    # Split y into the discrete-time outcome label and survival/censoring time.
    label = np.asarray(y[["outcome"]])
    time = np.asarray(y[["time"]])
    data = np.asarray(X)
    data = f_get_Normalization(data, norm_mode="standard")

    # num_Category is fixed across cohorts (driven by T_MAX in constants);
    # num_Event is inferred from the label column (0 = censored).
    num_Category = int(const.T_MAX * 1.2)
    num_Event = int(len(np.unique(label)) - 1)

    x_dim = data.shape[1]

    mask1 = f_get_fc_mask2(time, label, num_Event, num_Category)
    mask2 = f_get_fc_mask3(time, -1, num_Category)

    DIM = x_dim
    DATA = (data, time, label)
    MASK = (mask1, mask2)

    return DIM, DATA, MASK
