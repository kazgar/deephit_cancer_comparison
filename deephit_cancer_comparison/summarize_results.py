import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import deephit_cancer_comparison.constants as const
import deephit_cancer_comparison.import_data as impt
from deephit_cancer_comparison.class_deephit import DeepHit
from deephit_cancer_comparison.utils import load_logging, set_seeds
from deephit_cancer_comparison.utils_eval import (
    weighted_brier_score,
    weighted_c_index,
)


def main():
    if len(sys.argv) < 2:
        raise ValueError("Must provide treatment_arm")

    treatment_arm = sys.argv[1]

    if const.DATA_MODE in impt.data_reading_functions.keys():
        data_func = impt.data_reading_functions[const.DATA_MODE]
        x_dim_train, DATA_train, MASK_train = data_func(treatment_arm, "train")
        _, time_train, _ = DATA_train
        x_dim_test, DATA_test, MASK_test = data_func(treatment_arm, "test")
        EVAL_TIMES = list(range(const.TIMESTEP, int(np.max(time_train) * 1.2), const.TIMESTEP))
    else:
        raise ValueError("ERROR: DATA_MODE NOT FOUND!!!")

    data_train, time_train, label_train = DATA_train
    mask1_train, mask2_train = MASK_train
    _, num_Event_train, num_Category_train = mask1_train.shape

    data_test, time_test, label_test = DATA_test
    mask1_test, mask2_test = MASK_test
    _, num_Event_test, num_Category_test = mask1_test.shape

    assert x_dim_train == x_dim_test, "Dimenions (train-test) don't match"
    assert num_Event_train == num_Event_test, "Number of events (train-test) don't match"

    num_Category = max(num_Category_train, num_Category_test)

    print(f"num_Category = {num_Category}")
    print(f"EVAL_TIMES = {EVAL_TIMES}")

    if not os.path.exists(const.RESULTS_PATH / treatment_arm):
        raise FileNotFoundError(
            f"ERROR: RESULTS FOR {const.DATA_MODE} (EXP NR: {const.EXPERIMENT_NR} NOT FOUND!!!"
        )

    FINAL1 = np.zeros([num_Event_test, len(EVAL_TIMES), const.OUT_ITERATION])
    FINAL2 = np.zeros([num_Event_test, len(EVAL_TIMES), const.OUT_ITERATION])

    for out_itr in range(const.OUT_ITERATION):
        in_parser = load_logging(
            const.RESULTS_PATH / treatment_arm / f"itr_{out_itr}" / "hyperparameters.txt"
        )
        print("Hyperparameters being used:")
        for key, value in in_parser.items():
            print(f"{key}: {value}")

        h_dim_shared = in_parser["h_dim_shared"]
        h_dim_CS = in_parser["h_dim_CS"]
        num_layers_shared = in_parser["num_layers_shared"]
        num_layers_CS = in_parser["num_layers_CS"]

        active_fn_dict = {"relu": F.relu, "elu": F.elu, "tanh": F.tanh}
        if in_parser["active_fn"] in active_fn_dict:
            active_fn = active_fn_dict[in_parser["active_fn"]]
        else:
            raise ValueError("ERROR: INVALID ACTIVATION FUNCTION!!!")

        initial_W = torch.nn.init.xavier_normal_

        alpha = in_parser["alpha"]
        beta = in_parser["beta"]
        gamma = in_parser["gamma"]
        parameter_name = f"a{10 * alpha:02.0f}b{10 * beta:02.0f}c{10 * gamma:02.0f}"

        input_dims = {
            "x_dim": x_dim_test,
            "num_Event": num_Event_test,
            "num_Category": num_Category,
        }

        network_settings = {
            "h_dim_shared": h_dim_shared,
            "h_dim_CS": h_dim_CS,
            "num_layers_shared": num_layers_shared,
            "num_layers_CS": num_layers_CS,
            "active_fn": active_fn,
            "initial_W": initial_W,
        }

        set_seeds(const.SEED)
        model = DeepHit(input_dims, network_settings).to(const.DEVICE)

        print("Model initialized with the following settings:")
        print(f"x_dim: {model.x_dim}")
        print(f"num_Event: {model.num_Event}")
        print(f"num_Category: {model.num_Category}")
        print(f"h_dim_shared: {model.h_dim_shared}")
        print(f"h_dim_CS: {model.h_dim_CS}")
        print(f"num_layers_shared: {model.num_layers_shared}")
        print(f"num_layers_CS: {model.num_layers_CS}")
        print(f"active_fn: {model.active_fn}")
        print(f"initial_W: {model.initial_W}")

        print("\nShared layers:")
        for i, layer in enumerate(model.shared_layers):
            print(f"Layer {i}: {layer}")

        print("\nCause-specific layers:")
        for i, event_layers in enumerate(model.cause_specific_layers):
            print(f"Event {i} layers:")
            for j, layer in enumerate(event_layers):
                print(f"  Layer {j}: {layer}")

        assert (
            model.x_dim == input_dims["x_dim"]
        ), f"x_dim mismatch: {model.x_dim} != {input_dims['x_dim']}"
        assert (
            model.num_Event == input_dims["num_Event"]
        ), f"num_Event mismatch: {model.num_Event} != {input_dims['num_Event']}"
        assert (
            model.num_Category == input_dims["num_Category"]
        ), f"num_Category mismatch: {model.num_Category} != {input_dims['num_Category']}"

        assert (
            model.h_dim_shared == network_settings["h_dim_shared"]
        ), f"h_dim_shared mismatch: {model.h_dim_shared} != {network_settings['h_dim_shared']}"
        assert (
            model.h_dim_CS == network_settings["h_dim_CS"]
        ), f"h_dim_CS mismatch: {model.h_dim_CS} != {network_settings['h_dim_CS']}"
        assert (
            model.num_layers_shared == network_settings["num_layers_shared"]
        ), f"num_layers_shared mismatch: {model.num_layers_shared} != {network_settings['num_layers_shared']}"
        assert (
            model.num_layers_CS == network_settings["num_layers_CS"]
        ), f"num_layers_CS mismatch: {model.num_layers_CS} != {network_settings['num_layers_CS']}"

        model.load_state_dict(
            torch.load(
                const.RESULTS_PATH
                / treatment_arm
                / f"itr_{out_itr}"
                / "models"
                / f"model_itr_{out_itr}.pth",
                map_location=const.DEVICE,
            )
        )

        data_test = torch.tensor(data_test, dtype=torch.float32).to(const.DEVICE)

        model.eval()
        with torch.no_grad():
            pred = model(data_test)

        result1, result2 = np.zeros([num_Event_test, len(EVAL_TIMES)]), np.zeros(
            [num_Event_test, len(EVAL_TIMES)]
        )

        for t, t_time in enumerate(EVAL_TIMES):
            eval_horizon = int(t_time)

            if eval_horizon >= num_Category:
                print("ERROR: evaluation horizon is out of range")
                result1[:, t] = result2[:, t] = -1
            else:
                risk = np.sum(pred[:, :, : (eval_horizon + 1)].cpu().numpy(), axis=2)

                for k in range(num_Event_test):
                    result1[k, t] = weighted_c_index(
                        time_train,
                        (label_train[:, 0] == k + 1).astype(int).reshape(-1),
                        risk[:, k],
                        time_test,
                        (label_test[:, 0] == k + 1).astype(int).reshape(-1),
                        eval_horizon,
                    )
                    result2[k, t] = weighted_brier_score(
                        time_train,
                        (label_train[:, 0] == k + 1).astype(int).reshape(-1),
                        risk[:, k],
                        time_test,
                        (label_test[:, 0] == k + 1).astype(int).reshape(-1),
                        eval_horizon,
                    )

        FINAL1[:, :, out_itr] = result1
        FINAL2[:, :, out_itr] = result2

        row_header = [f"Event_{t + 1}" for t in range(num_Event_train)]
        col_header1 = [f"{t}yr c_index" for t in EVAL_TIMES]
        col_header2 = [f"{t}yr B_score" for t in EVAL_TIMES]

        df1 = pd.DataFrame(result1, index=row_header, columns=col_header1)
        df1.to_csv(
            const.RESULTS_PATH
            / treatment_arm
            / f"itr_{out_itr}"
            / f"result_CINDEX_itr_{out_itr}.csv"
        )

        df2 = pd.DataFrame(result2, index=row_header, columns=col_header2)
        df2.to_csv(
            const.RESULTS_PATH
            / treatment_arm
            / f"itr_{out_itr}"
            / f"result_BRIER_itr_{out_itr}.csv"
        )

        print("========================================================")
        print(f"ITR: {out_itr + 1} DATA MODE: {const.DATA_MODE} (a:{alpha} b:{beta} c:{gamma})")
        print(
            f"SharedNet Parameters: h_dim_shared = {h_dim_shared}, num_layers_shared = {num_layers_shared}, Non-Linearity: {active_fn}"
        )
        print(
            f"CSNet Parameters: h_dim_CS = {h_dim_CS}, num_layers_CS = {num_layers_CS}, Non-Linearity: {active_fn}"
        )
        print("--------------------------------------------------------")
        print("- C-INDEX: ")
        print(df1)
        print("--------------------------------------------------------")
        print("- BRIER-SCORE: ")
        print(df2)
        print("========================================================")

    df1_mean = pd.DataFrame(np.mean(FINAL1, axis=2), index=row_header, columns=col_header1)
    df1_std = pd.DataFrame(np.std(FINAL1, axis=2), index=row_header, columns=col_header1)
    df1_mean.to_csv(const.RESULTS_PATH / treatment_arm / "result_CINDEX_FINAL_MEAN.csv")
    df1_std.to_csv(const.RESULTS_PATH / treatment_arm / "result_CINDEX_FINAL_STD.csv")

    df2_mean = pd.DataFrame(np.mean(FINAL2, axis=2), index=row_header, columns=col_header2)
    df2_std = pd.DataFrame(np.std(FINAL2, axis=2), index=row_header, columns=col_header2)
    df2_mean.to_csv(const.RESULTS_PATH / treatment_arm / "result_BRIER_FINAL_MEAN.csv")
    df2_std.to_csv(const.RESULTS_PATH / treatment_arm / "result_BRIER_FINAL_STD.csv")

    print("========================================================")
    print("- FINAL C-INDEX: ")
    print(df1_mean)
    print("--------------------------------------------------------")
    print("- FINAL BRIER-SCORE: ")
    print(df2_mean)
    print("========================================================")


if __name__ == "__main__":
    main()
