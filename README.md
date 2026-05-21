# DeepHit Cancer Comparison

Competing-risks survival analysis using [DeepHit](https://ojs.aaai.org/index.php/AAAI/article/view/11842) applied to ten cancer cohorts from the SEER registry. For each cohort the model is trained with nested cross-validation and evaluated on cancer-specific death (primary event) and other-cause death (secondary event) using the weighted C-index and weighted Brier score. Time-dependent feature importances are computed via [SurvSHAP(t)](https://arxiv.org/abs/2208.11080).

## Dataset

The study uses data from the **SEER (Surveillance, Epidemiology, and End Results) Program, 2004–2021**, maintained by the National Cancer Institute (NCI).

> **The SEER dataset cannot be included in this repository and is not available for public download.** Access requires approval from the NCI, which was obtained for this study. Researchers wishing to replicate the work must apply for access independently through the [SEER Data Access](https://seer.cancer.gov/data/access.html) portal.

The ten cancer cohorts used are: Breast, Corpus (Uteri), Kidney Parenchyma, Melanoma of the Skin, Lung & Bronchus, Pancreas, Prostate, Thyroid, Urinary Bladder, and Colorectal (Colon & Rectum).

## Repository structure

```
deephit_cancer_comparison/          # main Python package
├── notebooks/
│   ├── seer_data_parsing.ipynb     # parse raw SEER data: filter, encode, and split into per-cohort train/test CSVs
│   ├── plot_performance.ipynb      # plot per-cohort C-index and Brier score curves and cross-cohort bar charts
│   └── plot_survshap.ipynb         # load SurvSHAP(t) artifacts and produce feature importance heatmaps and bar charts
├── constants.py                    # global paths, seeds, and time-grid settings
├── class_deephit.py                # DeepHit model (shared + cause-specific sub-networks)
├── import_data.py                  # data loading, normalization, and mask construction
├── get_main.py                     # training loop with early stopping
├── main_randomsearch.py            # outer CV + random hyperparameter search entrypoint
├── summarize_results.py            # test-set evaluation (C-index, Brier score)
├── extract_performance.py          # aggregate per-cohort metrics into performance.csv
├── calc_survshap.py                # SurvSHAP(t) computation for trained models
├── survshap_utils.py               # SurvSHAP helper functions (predict wrappers, I/O)
├── utils.py                        # general utilities (seeding, hyperparameter sampling, logging)
└── utils_eval.py                   # weighted C-index and weighted Brier score implementations
```

## Workflow

1. **Hyperparameter search** — for each cancer cohort, run nested cross-validation (5 outer iterations × 10 random-search trials):
   ```
   python -m deephit_cancer_comparison.main_randomsearch <cancer_type>
   ```

2. **Test-set evaluation** — load the best checkpoint from each outer iteration and compute C-index and Brier score on the held-out test set:
   ```
   python -m deephit_cancer_comparison.summarize_results <cancer_type>
   ```

3. **Aggregate performance** — collect results across all cohorts into a single CSV:
   ```
   python -m deephit_cancer_comparison.extract_performance
   ```

4. **SurvSHAP(t)** — compute time-dependent SHAP values for a trained model:
   ```
   python -m deephit_cancer_comparison.calc_survshap <cancer_type> [--iterations 0] [--explain-n 200] [--ref-n 100]
   ```

## Installation

Requires Python 3.12+. Install dependencies with [uv](https://github.com/astral-sh/uv):

```
uv sync
```

or with pip:

```
pip install -r requirements.txt
```

**Remember to run:**
```
uv pip install -e .
```

The package uses GPU acceleration automatically when CUDA or Apple MPS is available; it falls back to CPU otherwise.
