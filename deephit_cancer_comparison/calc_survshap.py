import argparse
from random import randint

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from survshap import ModelSurvSHAP, SurvivalModelExplainer

import deephit_cancer_comparison.constants as const
import deephit_cancer_comparison.import_data as impt
from deephit_cancer_comparison.class_deephit import DeepHit
from deephit_cancer_comparison.survshap_utils import (
    _deephit_predict_pmf,
    _surv_on_grid,
    predict_cumulative_hazard_function,
    predict_survival_function,
)
from deephit_cancer_comparison.utils import load_logging, set_seeds


def main():
    parser = argparse.ArgumentParser(
        description="Compute SurvSHAP(t) feature importances for trained DeepHit models."
    )
    parser.add_argument("cancer_type", help="Cancer cohort name (e.g., 'breast').")
    parser.add_argument(
        "--iterations",
        type=int,
        nargs="+",
        default=None,
        help="Which outer iterations to run. Defaults to random.",
    )
    parser.add_argument("--n-background", type=int, default=25)
    parser.add_argument(
        "--calculation-method",
        default="sampling",
        choices=["kernel", "sampling", "shap_kernel"],
    )
    args = parser.parse_args()

    set_seeds(const.SEED)

    cancer_dir = const.RESULTS_PATH / args.cancer_type

    if args.iterations is None:
        iteration = randint(0, 4)
    else:
        iteration = args.iterations

    itr_path = cancer_dir / f"itr_{iteration}"
    model_path = itr_path / "models" / f"model_itr_{iteration}.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"No model file found for itr: {iteration}")

    in_parser = load_logging(itr_path / "hyperparameters.txt")

    h_dim_shared = in_parser["h_dim_shared"]
    h_dim_CS = in_parser["h_dim_CS"]
    num_layers_shared = in_parser["num_layers_shared"]
    num_layers_CS = in_parser["num_layers_CS"]

    active_fn_dict = {"relu": F.relu, "elu": F.elu, "tanh": F.tanh}
    if in_parser["active_fn"] in active_fn_dict:
        active_fn = active_fn_dict[in_parser["active_fn"]]
    else:
        raise ValueError("Invalid activation function.")

    initial_W = torch.nn.init.xavier_normal_

    in_parser["alpha"]
    in_parser["beta"]
    in_parser["gamma"]

    data_func = impt.import_cohort_data
    x_dim_test, DATA_test, MASK_test = data_func(args.cancer_type, "test")
    data_test, time_test, label_test = DATA_test

    event_flat = label_test.flatten().astype(int)
    time_flat = time_test.flatten().astype(int)

    mask1_test, mask2_test = MASK_test
    _, num_Event_test, num_Category_test = mask1_test.shape
    list(range(const.TIMESTEP, int(np.max(const.T_MAX) * 1.2), const.TIMESTEP))

    _time_bins = np.arange(num_Category_test, dtype=float)

    input_dims = {
        "x_dim": x_dim_test,
        "num_Event": num_Event_test,
        "num_Category": num_Category_test,
    }

    network_settings = {
        "h_dim_shared": h_dim_shared,
        "h_dim_CS": h_dim_CS,
        "num_layers_shared": num_layers_shared,
        "num_layers_CS": num_layers_CS,
        "active_fn": active_fn,
        "initial_W": initial_W,
    }

    model = DeepHit(input_dims, network_settings).to(const.DEVICE)

    model.load_state_dict(torch.load(model_path, map_location=const.DEVICE))

    data_test = torch.tensor(data_test, dtype=torch.float32).to(const.DEVICE)

    data_test_np = (
        data_test.detach().cpu().numpy() if torch.is_tensor(data_test) else np.asarray(data_test)
    )
    feature_names = [f"x{i}" for i in range(data_test_np.shape[1])]
    data_test_df = pd.DataFrame(data_test_np, columns=feature_names)

    event_bool = event_flat == const.PRIMARY_EVENT_LABEL
    time_f64 = time_flat.astype(np.float64)

    test_y = np.array(
        list(zip(event_bool, time_f64)),
        dtype=[("event", "?"), ("time", "<f8")],
    )

    REF_N = 100
    rng = np.random.default_rng(const.SEED)
    all_idx = np.arange(len(data_test_df))

    ref_idx = rng.choice(all_idx, size=REF_N, replace=False)
    data_ref_df = data_test_df.iloc[ref_idx].reset_index(drop=True)
    test_y_ref = test_y[ref_idx]

    EXPLAIN_N = 100
    remaining = np.setdiff1d(all_idx, ref_idx)
    explain_idx = rng.choice(remaining, size=EXPLAIN_N, replace=False)
    new_observations = data_test_df.iloc[explain_idx].reset_index(drop=True)

    deephit_exp = SurvivalModelExplainer(
        model=model,
        data=data_ref_df,
        y=test_y_ref,
        predict_survival_function=predict_survival_function,
        predict_cumulative_hazard_function=predict_cumulative_hazard_function,
    )

    deephit_survshap = ModelSurvSHAP(
        calculation_method="sampling",
        random_state=const.SEED,
    )

    deephit_survshap.fit(
        deephit_exp,
        new_observations=new_observations,
    )

    print("\n=== ModelSurvSHAP inspection ===")
    print("attrs:", [a for a in dir(deephit_survshap) if not a.startswith("_")])
    print("n individual explanations:", len(deephit_survshap.individual_explanations))
    first = deephit_survshap.individual_explanations[0]
    print("first expl attrs:", [a for a in dir(first) if not a.startswith("_")])
    print("first.result.shape:", first.result.shape)
    print("first.result.columns[:10]:", list(first.result.columns[:10]))
    print("first.result.head():")
    print(first.result.head())

    print(first.result.groupby("variable_name").size().head())
    print(first.result["B"].nunique(), first.result["B"].describe())

    print("simplified_result.shape:", first.simplified_result.shape)
    print(first.simplified_result.head())


if __name__ == "__main__":
    main()
