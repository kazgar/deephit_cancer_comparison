import random
import sys

import pandas as pd

import deephit_cancer_comparison.constants as const
import deephit_cancer_comparison.get_main as get_main
import deephit_cancer_comparison.import_data as impt
from deephit_cancer_comparison.utils import (
    get_random_hyperparameters,
    np,
    os,
    save_logging,
    set_seeds,
)


def main():
    if len(sys.argv) < 2:
        raise ValueError("Must provide cancer_type")

    cancer_type = sys.argv[1]

    data_func = impt.import_cohort_data
    x_dim, DATA, MASK = data_func(cancer_type)
    _, time, _ = DATA
    # Extend eval horizon 20 % beyond the observed max time to cover late events
    EVAL_TIMES = list(range(const.TIMESTEP, int(np.max(time) * 1.2), const.TIMESTEP))

    data, time, label = DATA
    mask1, mask2 = MASK

    scores = []

    for out_itr in range(const.OUT_ITERATION):
        # Each outer iteration gets its own random seed so folds are independent
        out_seed = random.getrandbits(32)
        set_seeds(out_seed)
        itr_dir = const.RESULTS_PATH / cancer_type / f"itr_{out_itr}"

        if not os.path.exists(itr_dir):
            os.makedirs(itr_dir)

        max_valid = 0.0  # best validation C-index seen so far in this outer iteration

        log_name = itr_dir / "hyperparameters.txt"

        for r_itr in range(const.RS_ITERATION):
            print(f"OUTER_ITERATION: {out_itr}")
            print(f"Random search... iteration: {r_itr}")

            # Re-seed before each random search trial for reproducible hyperparameter sampling
            inner_seed = random.getrandbits(32)
            set_seeds(inner_seed)

            new_parser = get_random_hyperparameters(const.RESULTS_PATH / cancer_type)

            # Pass current best as MAX_VALUE so get_valid_performance skips saving
            # checkpoints that cannot beat the running best
            tmp_max = get_main.get_valid_performance(
                DATA, MASK, new_parser, out_itr, EVAL_TIMES, MAX_VALUE=max_valid
            )

            if tmp_max > max_valid:
                max_valid = tmp_max
                max_parser = new_parser
                # Overwrite the log only when a strictly better configuration is found
                save_logging(max_parser, log_name)

            print(f"Current best: {max_valid}")

        scores.append(max_valid)

    # One row per outer iteration; used later to report cross-iteration variance
    results_df = pd.DataFrame({"itr": [i for i in range(const.OUT_ITERATION)], "scores": scores})

    results_df.to_csv(
        const.RESULTS_PATH / cancer_type / f"{cancer_type}_results_by_iteration.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
