from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

DATA_PATH = PROJECT_ROOT / "data"

DEVICE = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps") if torch.mps.is_available() else torch.device("cpu")
)

PRIMARY_EVENT_LABEL = 0

OUT_ITERATION = 5
RS_ITERATION = 10

SEED = 1234

TIMESTEP = 1
T_MAX = 227
EVAL_TIMES = list(range(TIMESTEP, int(np.max(T_MAX) * 1.2), TIMESTEP))

EPSILON = 1e-08

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DEEPHIT_DIR_PATH = PROJECT_ROOT / "deephit"
DATA_PATH = PROJECT_ROOT / "data"
RESULTS_PATH = PROJECT_ROOT / "results"
GRAPH_PATH = PROJECT_ROOT / "graphs"
