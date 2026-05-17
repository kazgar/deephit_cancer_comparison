from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import deephit_cancer_comparison.constants as const

# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------


def load_all_cohorts(results_root: Path, artifact: str) -> pd.DataFrame:
    # Pattern matches one level deep: <results_root>/<cohort>/<artifact>.parquet
    paths = list(results_root.glob(f"*/{artifact}.parquet"))
    if not paths:
        raise FileNotFoundError(f"No {artifact}.parquet files found under {results_root}")
    frames = [pd.read_parquet(p) for p in paths]
    # ignore_index prevents duplicate row indices from the per-cohort frames
    return pd.concat(frames, ignore_index=True)


# -----------------------------------------------------------------------------
# Plot 1: Cross-cohort feature-importance heatmap (your headline figure)
# -----------------------------------------------------------------------------


def plot_cross_cohort_heatmap(
    scalar_df: pd.DataFrame,
    feature_col: str = "variable_name",
    value_col: str = "aggregated_change",
    agg: str = "mean",
    top_k: int | None = None,
    figsize: tuple = (10, 8),
    cmap: str = "viridis",
    save_path: Path | None = None,
) -> pd.DataFrame:
    """Heatmap of (feature x cohort) aggregated importance. Use `scalar_df` = survshap_aggregated
    (or _grouped) loaded across cohorts.

    Returns the pivoted matrix (useful for tables in your results chapter).
    """
    # 1. Patient-level -> cohort-level aggregation.
    cohort_level = scalar_df.groupby(["cohort", feature_col])[value_col].agg(agg).reset_index()

    # 2. Pivot to matrix.
    matrix = cohort_level.pivot(index=feature_col, columns="cohort", values=value_col)

    # 3. Order features by overall mean importance (most important at top).
    feature_order = matrix.mean(axis=1).sort_values(ascending=False).index
    matrix = matrix.loc[feature_order]

    # 4. Optional: keep only top_k features.
    if top_k is not None:
        matrix = matrix.head(top_k)

    # 5. Plot.
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        cbar_kws={"label": f"{agg.capitalize()} |SurvSHAP(t)| aggregated"},
        ax=ax,
    )
    ax.set_xlabel("Cancer cohort")
    ax.set_ylabel("Feature")
    ax.set_title(f"Cross-cohort feature importance ({agg} aggregated_change)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return matrix


# -----------------------------------------------------------------------------
# Plot 2: Per-cohort top-k feature ranking (bar chart)
# -----------------------------------------------------------------------------


def plot_cohort_top_features(
    scalar_df: pd.DataFrame,
    cohort: str,
    feature_col: str = "variable_name",
    value_col: str = "aggregated_change",
    top_k: int = 10,
    figsize: tuple = (8, 6),
    save_path: Path | None = None,
) -> pd.DataFrame:
    """Horizontal bar chart of top-k features for a single cohort.

    Error bars = patient-level variability (IQR).
    """
    cohort_data = scalar_df[scalar_df["cohort"] == cohort].copy()

    summary = (
        cohort_data.groupby(feature_col)[value_col]
        # Named lambdas are immediately renamed; pandas auto-names them "<lambda_0/1>"
        .agg(["median", "mean", lambda s: s.quantile(0.25), lambda s: s.quantile(0.75)])
        .rename(columns={"<lambda_0>": "q25", "<lambda_1>": "q75"})
        .sort_values("mean", ascending=False)
        .head(top_k)
    )

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(summary))
    ax.barh(
        y_pos,
        summary["mean"],
        # xerr expects [left_errors, right_errors]: distances from mean to q25 and q75
        xerr=[summary["mean"] - summary["q25"], summary["q75"] - summary["mean"]],
        color="steelblue",
        alpha=0.8,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(summary.index)
    ax.invert_yaxis()  # puts the highest-ranked feature at the top of the chart
    ax.set_xlabel("Mean |SurvSHAP(t)| aggregated (IQR error bars)")
    ax.set_title(f"Top {top_k} features — {cohort}")
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return summary


# -----------------------------------------------------------------------------
# Plot 3: Time-varying SurvSHAP(t) for a feature within a cohort
# -----------------------------------------------------------------------------
# This is the plot that justifies using SurvSHAP over plain SHAP — it shows
# that feature influence CHANGES over time, which is the paper's main selling
# point (Krzyzinski et al. 2023, Figs. 1, 11).


def plot_timevarying_feature(
    tv_df: pd.DataFrame,
    cohort: str,
    feature: str,
    show_individuals: bool = True,
    individual_alpha: float = 0.15,
    figsize: tuple = (10, 5),
    save_path: Path | None = None,
):
    """Plot SurvSHAP(t) values over time for one feature in one cohort.

    Thin lines = individual patients; bold line = cohort mean.
    """
    sub = tv_df[(tv_df["cohort"] == cohort) & (tv_df["variable_name"] == feature)].copy()
    if sub.empty:
        raise ValueError(f"No data for cohort={cohort}, feature={feature}")

    # Columns named "t = <float>" hold the per-time-bin SurvSHAP values; extract their order
    time_cols = [c for c in sub.columns if isinstance(c, str) and c.startswith("t = ")]
    times = np.array([float(c.removeprefix("t = ")) for c in time_cols])
    values = sub[time_cols].to_numpy()  # shape: (n_patients, n_times)

    fig, ax = plt.subplots(figsize=figsize)

    if show_individuals:
        for i in range(values.shape[0]):
            ax.plot(times, values[i], color="steelblue", alpha=individual_alpha, linewidth=0.8)

    ax.plot(times, values.mean(axis=0), color="darkblue", linewidth=2.2, label="Cohort mean")

    # Zero reference line: attribution above/below zero means the feature increases/decreases risk
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Time (bin index)")
    ax.set_ylabel(r"SurvSHAP$_t(x, d)$")
    ax.set_title(f"Time-varying attribution of '{feature}' — {cohort}")
    ax.legend()
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")


# -----------------------------------------------------------------------------
# Plot 4: Same feature across ALL cohorts (cross-cohort time-varying comparison)
# -----------------------------------------------------------------------------


def plot_feature_across_cohorts(
    tv_df: pd.DataFrame,
    feature: str,
    figsize: tuple = (10, 6),
    save_path: Path | None = None,
):
    """One line per cohort, showing cohort-mean SurvSHAP(t) over time for a given feature.

    Good for answering "does feature X's influence differ by cancer type?"
    """
    sub = tv_df[tv_df["variable_name"] == feature].copy()
    if sub.empty:
        raise ValueError(f"No data for feature={feature}")

    time_cols = [c for c in sub.columns if isinstance(c, str) and c.startswith("t = ")]
    times = np.array([float(c.removeprefix("t = ")) for c in time_cols])

    fig, ax = plt.subplots(figsize=figsize)
    cohorts = sorted(sub["cohort"].unique())
    # tab10 gives up to 10 perceptually distinct colours; linspace spreads them evenly
    colors = plt.cm.tab10(np.linspace(0, 1, len(cohorts)))

    for cohort, color in zip(cohorts, colors):
        cohort_data = sub[sub["cohort"] == cohort]
        # .values converts the Series to a plain array so ax.plot receives consistent types
        mean_curve = cohort_data[time_cols].mean(axis=0).values
        ax.plot(times, mean_curve, label=cohort, color=color, linewidth=1.8)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Time (bin index)")
    ax.set_ylabel(r"Mean SurvSHAP$_t(x, d)$")
    ax.set_title(f"Time-varying attribution of '{feature}' across cohorts")
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")


def main():

    scalar_df = load_all_cohorts(const.SURVSHAP_PATH, "survshap_aggregated")
    tv_df = load_all_cohorts(const.SURVSHAP_PATH, "survshap_timevarying")

    plot_cross_cohort_heatmap(
        scalar_df,
        top_k=15,
        save_path=Path("figures/cross_cohort_heatmap.pdf"),
    )

    for cohort in scalar_df["cohort"].unique():
        plot_cohort_top_features(
            scalar_df,
            cohort=cohort,
            top_k=10,
            save_path=Path(f"figures/top_features_{cohort}.pdf"),
        )

    # Illustrative calls — swap cohort/feature arguments to generate other plots
    plot_timevarying_feature(
        tv_df,
        cohort="breast",
        feature="age_at_diagnosis",
        save_path=Path("figures/timevarying_age_breast.pdf"),
    )

    plot_feature_across_cohorts(
        tv_df,
        feature="age_at_diagnosis",
        save_path=Path("figures/age_across_cohorts.pdf"),
    )


if __name__ == "__main__":
    main()
