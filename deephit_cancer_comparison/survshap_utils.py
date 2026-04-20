import pickle

import numpy as np
import pandas as pd
import torch
from sksurv.functions import StepFunction

import deephit_cancer_comparison.constants as const

times_grid = np.array(const.EVAL_TIMES, dtype=float)


def _deephit_predict_pmf(model, X):
    model.eval()
    if torch.is_tensor(X):
        X_t = X.to(const.DEVICE).float()
    else:
        X_np = np.ascontiguousarray(np.asarray(X))  # silences the read-only warning
        X_t = torch.as_tensor(X_np, dtype=torch.float32, device=const.DEVICE)

    with torch.no_grad():
        out = model(X_t)
    return out.detach().cpu().numpy()


def _surv_on_grid(pmf_primary, target_times=times_grid, _time_bins=times_grid):
    cdf = np.cumsum(pmf_primary, axis=1)
    surv = 1.0 - cdf
    idx = np.searchsorted(_time_bins, target_times, side="right") - 1
    idx = np.clip(idx, 0, surv.shape[1] - 1)
    return np.clip(surv[:, idx], 1e-8, 1.0)


def predict_survival_function(m, d, PRIMARY_EVENT_IDX=const.PRIMARY_EVENT_LABEL):
    pmf = _deephit_predict_pmf(m, d)
    T = pmf.shape[-1]
    time_bins = np.arange(T, dtype=float)
    pmf_primary = pmf[:, PRIMARY_EVENT_IDX, :]
    surv = np.clip(1.0 - np.cumsum(pmf_primary, axis=1), 1e-8, 1.0)

    out = np.empty(surv.shape[0], dtype=object)
    for i in range(surv.shape[0]):
        out[i] = StepFunction(x=time_bins, y=surv[i])
    return out


def predict_cumulative_hazard_function(m, d):
    preds = predict_survival_function(m, d)
    out = np.empty(len(preds), dtype=object)
    for i, sf in enumerate(preds):
        out[i] = StepFunction(x=sf.x, y=-np.log(np.clip(sf.y, 1e-8, 1.0)))
    return out


def save_survshap(deephit_survshap, explain_idx, cohort, iteration, out_root):
    survshap_dir = out_root / cohort
    survshap_dir.mkdir(parents=True, exist_ok=True)

    with open(survshap_dir / "model_survshap.pkl", "wb") as f:
        pickle.dump(deephit_survshap, f)

    first = deephit_survshap.individual_explanations[0]
    time_cols = [c for c in first.result.columns if isinstance(c, str) and c.startswith("t = ")]

    tv_rows = []
    for local_i, expl in enumerate(deephit_survshap.individual_explanations):
        per_feature = expl.result.groupby("variable_name", as_index=False)[time_cols].mean()
        per_feature["obs_id"] = int(explain_idx[local_i])
        tv_rows.append(per_feature)

    tv_df = pd.concat(tv_rows, ignore_index=True)
    tv_df["cohort"] = cohort
    tv_df["iteration"] = iteration
    tv_df = tv_df[["cohort", "iteration", "obs_id", "variable_name"] + time_cols]
    tv_df.to_parquet(survshap_dir / "survshap_timevarying.parquet", index=False)

    scalar_rows = []
    for local_i, expl in enumerate(deephit_survshap.individual_explanations):
        df = expl.simplified_result.copy()
        df["obs_id"] = int(explain_idx[local_i])
        scalar_rows.append(df)

    scalar_df = pd.concat(scalar_rows, ignore_index=True)
    scalar_df["cohort"] = cohort
    scalar_df["iteration"] = iteration
    front = [
        "cohort",
        "iteration",
        "obs_id",
        "variable_name",
        "variable_value",
        "aggregated_change",
    ]
    scalar_df = scalar_df[front + [c for c in scalar_df.columns if c not in front]]
    scalar_df.to_parquet(survshap_dir / "survshap_aggregated.parquet", index=False)

    print(f"Saved SurvSHAP artifacts to {survshap_dir}")
    return survshap_dir
