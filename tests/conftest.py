"""Shared helpers: compose Hydra configs the way run.py does, without running."""
import itertools
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import bmf.config  # noqa: F401  (registers the structured base config)

CONF_DIR = Path(__file__).resolve().parents[1] / "conf"

OmegaConf.register_new_resolver(
    "linspace",
    lambda start, stop, num: ",".join(str(x) for x in np.linspace(start, stop, num)),
    replace=True,
)


def compose_config(name, overrides=()):
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        return compose(config_name=name, overrides=list(overrides))


def sweep_jobs(name):
    """One composed config per job of the sweep declared in conf/<name>.yaml."""
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        full = compose(config_name=name, return_hydra_config=True)
        params = OmegaConf.to_container(full.hydra.sweeper.params, resolve=True)
        keys = list(params)
        grids = [[v.strip() for v in str(params[k]).split(",")] for k in keys]
        return [
            compose(config_name=name, overrides=[f"{k}={v}" for k, v in zip(keys, combo)])
            for combo in itertools.product(*grids)
        ]


@pytest.fixture(autouse=True)
def workdir(tmp_path, monkeypatch):
    """Every test runs in a scratch directory (an Experiment creates its
    save_folder under the working directory) and without logging credentials."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("COMET_API_KEY", raising=False)
    monkeypatch.delenv("COMET_WORKSPACE", raising=False)
    return tmp_path
