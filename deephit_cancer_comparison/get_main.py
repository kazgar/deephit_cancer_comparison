import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from termcolor import colored

import deephit_cancer_comparison.constants as const
from deephit_cancer_comparison.class_deephit import DeepHit
from deephit_cancer_comparison.utils_eval import weighted_brier_score, weighted_c_index


def log(x):
    # Numerically stable log: avoids log(0) by adding a small epsilon before taking the log
    return torch.log(x + const.EPSILON)


def div(x, y):
    # Safe division: prevents division by zero by adding epsilon to the denominator
    return x / (y + const.EPSILON)


def f_get_minibatch(mb_size, x, label, time, mask1, mask2, device=const.DEVICE):
    idx = np.random.choice(np.arange(np.shape(x)[0]), mb_size, replace=False)

    # Cast to float32 before converting to tensors; the raw arrays may be float64
    x_mb = x[idx, :].astype(np.float32)
    k_mb = label[idx, :].astype(np.float32)
    t_mb = time[idx, :].astype(np.float32)
    m1_mb = mask1[idx, :, :].astype(np.float32)
    m2_mb = mask2[idx, :].astype(np.float32)
    return (
        torch.tensor(x_mb).to(device),
        torch.tensor(k_mb).to(device),
        torch.tensor(t_mb).to(device),
        torch.tensor(m1_mb).to(device),
        torch.tensor(m2_mb).to(device),
    )


def get_valid_performance(
    DATA, MASK, in_parser, out_itr, eval_time, MAX_VALUE=-99, seed=const.SEED
):
    if eval_time is None:
        raise ValueError("ERROR: eval_time is None!")

    (data, time, label) = DATA
    (mask1, mask2) = MASK

    x_dim = np.shape(data)[1]
    _, num_Event, num_Category = np.shape(mask1)

    ACTIVATION_FN = {"relu": F.relu, "elu": F.elu, "tanh": torch.tanh}

    mb_size = in_parser["mb_size"]
    iteration = in_parser["iteration"]
    lr_train = in_parser["lr_train"]

    alpha = in_parser["alpha"]
    beta = in_parser["beta"]
    gamma = in_parser["gamma"]

    # Compact string tag encoding the loss weights, e.g. "a05b10c02", used for checkpoint naming
    parameter_name = (
        "a"
        + str("%02.0f" % (10 * alpha))
        + "b"
        + str("%02.0f" % (10 * beta))
        + "c"
        + str("%02.0f" % (10 * gamma))
    )

    initial_W = torch.nn.init.xavier_uniform_

    input_dims = {"x_dim": x_dim, "num_Event": num_Event, "num_Category": num_Category}

    network_settings = {
        "h_dim_shared": in_parser["h_dim_shared"],
        "num_layers_shared": in_parser["num_layers_shared"],
        "h_dim_CS": in_parser["h_dim_CS"],
        "num_layers_CS": in_parser["num_layers_CS"],
        "active_fn": ACTIVATION_FN[in_parser["active_fn"]],
        "initial_W": initial_W,
    }

    model = DeepHit(input_dims, network_settings).to(const.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr_train)

    (
        tr_data,
        va_data,
        tr_time,
        va_time,
        tr_label,
        va_label,
        tr_mask1,
        va_mask1,
        tr_mask2,
        va_mask2,
        # 0.1765 ≈ 15 / 85: splits 85 % train / 15 % validation from the incoming (train+val) split
    ) = train_test_split(data, time, label, mask1, mask2, test_size=0.1765, random_state=seed)

    # Only validation features need to be a tensor up front; labels are used as numpy by the scorer
    va_data = torch.tensor(va_data, dtype=torch.float32).to(const.DEVICE)

    max_valid = -99
    stop_flag = 0  # counts consecutive non-improving checkpoints; training stops after 5

    SAVE_PATH = in_parser["out_path"]
    FILE_PATH_FINAL = SAVE_PATH / f"itr_{out_itr}"
    MODEL_PATH = FILE_PATH_FINAL / "models"

    os.makedirs(MODEL_PATH, exist_ok=True)

    print("MAIN TRAINING ...")
    print("EVALUATION TIMES: " + str(eval_time))

    avg_loss = 0
    for itr in range(iteration):
        if stop_flag > 5:
            break

        x_mb, k_mb, t_mb, m1_mb, m2_mb = f_get_minibatch(
            mb_size, tr_data, tr_label, tr_time, tr_mask1, tr_mask2, device=const.DEVICE
        )

        DATA = (x_mb, k_mb, t_mb)
        MASK = (m1_mb, m2_mb)
        PARAMETERS = (alpha, beta, gamma)

        loss_curr = model.training_step(DATA, MASK, PARAMETERS, optimizer)
        avg_loss += loss_curr / 1000  # accumulate mean over the next 1000-step window

        if (itr + 1) % 1000 == 0:
            print(
                "|| ITR: "
                + str("%04d" % (itr + 1))
                + " | Loss: "
                + colored(str("%.4f" % (avg_loss)), "yellow", attrs=["bold"])
            )
            avg_loss = 0

            model.eval()
            with torch.no_grad():
                pred = model(va_data)

            va_result1 = np.zeros([num_Event, len(eval_time)])

            for t, t_time in enumerate(eval_time):
                eval_horizon = int(t_time)

                if eval_horizon >= num_Category:
                    print("ERROR: evaluation horizon is out of range")
                    va_result1[:, t] = -1
                else:
                    # Cumulative incidence up to eval_horizon: sum predicted hazard across time bins
                    risk = torch.sum(pred[:, :, : (eval_horizon + 1)], dim=2).to("cpu").numpy()
                    for k in range(num_Event):
                        # Event labels are 1-indexed: event k corresponds to label value k+1
                        va_result1[k, t] = weighted_c_index(
                            tr_time,
                            (tr_label[:, 0] == k + 1).astype(int),
                            risk[:, k],
                            va_time,
                            (va_label[:, 0] == k + 1).astype(int),
                            eval_horizon,
                        )

            # Scalar summary across all events and horizons used to select the best checkpoint
            tmp_valid = np.mean(va_result1)

            if tmp_valid > max_valid:
                stop_flag = 0
                max_valid = tmp_valid
                print(f"Updated... Average C-index = {tmp_valid:.4f}")
                # MAX_VALUE guard allows the caller to skip saving obviously bad models
                if max_valid > MAX_VALUE:
                    torch.save(model.state_dict(), MODEL_PATH / f"model_itr_{out_itr}.pth")

            else:
                stop_flag += 1

    torch.cuda.empty_cache()
    return max_valid
