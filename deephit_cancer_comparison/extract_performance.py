import os

import pandas as pd

from deephit_cancer_comparison.constants import RESULTS_PATH

cancer_col_name_map = {
    "breast": "Breast",
    "corpus": "Corpus",
    "kidney_parenchyma": "Kidney Parenchyma",
    "melanoma_of_the_skin": "Melanoma",
    "lung_and_bronchus": "Lung & Bronchus",
    "pancreas": "Pancreas",
    "prostate": "Prostate",
    "thyroid": "Thyroid",
    "urinary_bladder": "Urinary Bladder",
    "colon_and_rectum": "Colorectal",
}


def extract_cindex_brier_score(dir_path):
    cindex_df = pd.read_csv(dir_path / "result_CINDEX_FINAL_MEAN.csv")
    cindex_all_columns = cindex_df.columns.tolist()
    cindex_all_columns.remove("Unnamed: 0")
    cindex_columns = [col for col in cindex_all_columns if int(col.split("yr")[0]) <= 120]
    primary_cindex_mean = cindex_df[cindex_columns].loc[0].mean()
    primary_cindex_std = cindex_df[cindex_columns].loc[0].std()
    sae_cindex_mean = cindex_df[cindex_columns].loc[1].mean()
    sae_cindex_std = cindex_df[cindex_columns].loc[1].std()

    brier_df = pd.read_csv(dir_path / "result_BRIER_FINAL_MEAN.csv")
    brier_all_columns = brier_df.columns.tolist()
    brier_all_columns.remove("Unnamed: 0")
    brier_columns = [col for col in brier_all_columns if int(col.split("yr")[0]) <= 120]
    primary_brier_mean = brier_df[brier_columns].loc[0].mean()
    primary_brier_std = brier_df[brier_columns].loc[0].std()
    sae_brier_mean = brier_df[brier_columns].loc[1].mean()
    sae_brier_std = brier_df[brier_columns].loc[1].std()

    return (
        primary_cindex_mean,
        primary_cindex_std,
        sae_cindex_mean,
        sae_cindex_std,
        primary_brier_mean,
        primary_brier_std,
        sae_brier_mean,
        sae_brier_std,
    )


def main():
    cancer_dir_paths = [
        RESULTS_PATH / cancer_dir
        for cancer_dir in os.listdir(RESULTS_PATH)
        if os.path.isdir(RESULTS_PATH / cancer_dir)
    ]

    performance_dict = {
        "cancer": [],
        "primary_cindex_mean": [],
        "primary_cindex_std": [],
        "sae_cindex_mean": [],
        "sae_cindex_std": [],
        "primary_brier_mean": [],
        "primary_brier_std": [],
        "sae_brier_mean": [],
        "sae_brier_std": [],
    }

    for cancer_dir in cancer_dir_paths:
        cancer_name = cancer_col_name_map[cancer_dir.name]
        (
            primary_cindex_mean,
            primary_cindex_std,
            sae_cindex_mean,
            sae_cindex_std,
            primary_brier_mean,
            primary_brier_std,
            sae_brier_mean,
            sae_brier_std,
        ) = extract_cindex_brier_score(cancer_dir)

        performance_dict["cancer"].append(cancer_name)
        performance_dict["primary_cindex_mean"].append(primary_cindex_mean)
        performance_dict["primary_cindex_std"].append(primary_cindex_std)
        performance_dict["sae_cindex_mean"].append(sae_cindex_mean)
        performance_dict["sae_cindex_std"].append(sae_cindex_std)
        performance_dict["primary_brier_mean"].append(primary_brier_mean)
        performance_dict["primary_brier_std"].append(primary_brier_std)
        performance_dict["sae_brier_mean"].append(sae_brier_mean)
        performance_dict["sae_brier_std"].append(sae_brier_std)

    pd.DataFrame().from_dict(performance_dict).to_csv(RESULTS_PATH / "performance.csv", index=False)


if __name__ == "__main__":
    main()
