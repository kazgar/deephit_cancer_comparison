from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

DATA_PATH = PROJECT_ROOT / "data"

# Pick the best available accelerator.
# All tensors and the DeepHit model are moved to this device at construction time.
DEVICE = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps") if torch.mps.is_available() else torch.device("cpu")
)

# Index of the "primary" competing risk we care about (cancer-specific death).
PRIMARY_EVENT_LABEL = 0

# OUT_ITERATION: number of outer cross-validation.
# RS_ITERATION: number of random-search trials inside each outer iteration.
OUT_ITERATION = 5
RS_ITERATION = 10

# Single global seed reused by numpy / random / torch for reproducibility.
SEED = 1234

TIMESTEP = 1
T_MAX = 227
EVAL_TIMES = list(range(TIMESTEP, int(np.max(T_MAX) * 1.2), TIMESTEP))

EPSILON = 1e-08

DEEPHIT_DIR_PATH = PROJECT_ROOT / "deephit"
DATA_PATH = PROJECT_ROOT / "data"
RESULTS_PATH = PROJECT_ROOT / "results"  # trained models + CSV metrics per cohort
SURVSHAP_PATH = PROJECT_ROOT / "survshap_results"  # SurvSHAP parquet artifacts
GRAPH_PATH = PROJECT_ROOT / "graphs"  # exported figures
