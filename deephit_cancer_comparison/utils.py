import os
import random

import numpy as np
import torch


def set_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def save_logging(dictionary, log_name):
    with open(log_name, "w") as f:
        for key, value in dictionary.items():
            f.write(f"{key}:{value}\n")


def load_logging(filename):
    data = {}
    with open(filename) as f:

        def is_float(input):
            try:
                float(input)
                return True
            except ValueError:
                return False

        for line in f.readlines():
            if ":" in line:
                key, value = line.strip().split(":", 1)
                if value.isdigit():
                    data[key] = int(value)
                elif is_float(value):
                    data[key] = float(value)
                elif value == "None":
                    data[key] = None
                else:
                    data[key] = value
            else:
                pass
    return data


def get_random_hyperparameters(out_path):
    SET_BATCH_SIZE = [32, 64, 128]
    SET_LAYERS = [1, 2, 3, 5]
    SET_NODES = [50, 100, 200, 300]
    SET_ACTIVATION_FN = ["relu", "elu", "tanh"]
    SET_BETA = [0.1, 0.5, 1.0, 3.0, 5.0]

    new_parser = {
        "mb_size": random.choice(SET_BATCH_SIZE),
        "iteration": 50000,
        "keep_prob": 0.6,
        "lr_train": 1e-4,
        "h_dim_shared": random.choice(SET_NODES),
        "h_dim_CS": random.choice(SET_NODES),
        "num_layers_shared": random.choice(SET_LAYERS),
        "num_layers_CS": random.choice(SET_LAYERS),
        "active_fn": random.choice(SET_ACTIVATION_FN),
        "alpha": 1.0,
        "beta": random.choice(SET_BETA),
        "gamma": 0,
        "out_path": out_path,
    }

    return new_parser


def get_hyperparameters(path):
    hyper_dict = {}
    for i, dir_path in enumerate(path.iterdir()):
        if dir_path.is_dir():
            hyperparams = dir_path / "hyperparameters_log.txt"
            with open(hyperparams) as f:
                hp_values = {hp: value for line in f for hp, value in [line.rstrip().split(":")]}
                hp_values.pop("out_path")
                hyper_dict[os.path.basename(dir_path)] = hp_values
    return hyper_dict
