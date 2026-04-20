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
        default="kernel",
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

    test_y = pd.DataFrame(
        {
            "event": (event_flat == const.PRIMARY_EVENT_LABEL).astype(bool),
            "time": time_flat,
        }
    )

    print(test_y)

    mask1_test, mask2_test = MASK_test
    _, num_Event_test, num_Category_test = mask1_test.shape
    list(range(const.TIMESTEP, int(np.max(time_test) * 1.2), const.TIMESTEP))

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

    deephit_exp = SurvivalModelExplainer(model, data_test, test_y)

    deephit_survshap = ModelSurvSHAP(random_state=const.SEED)

    print(deephit_exp)
    print(type(deephit_exp))

    deephit_survshap.fit(deephit_exp)


if __name__ == "__main__":
    main()
