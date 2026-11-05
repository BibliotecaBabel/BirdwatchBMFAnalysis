"""A tiny experiment end to end: run, save, load, estimate the anchor."""
import numpy as np

from bmf.experiment import Experiment
from bmf.gamma import gamma_estimates_grouped
from bmf.utils import load_project_data

from conftest import compose_config

TINY = ["dg.num_users=15", "dg.num_items=12", "replications=2", "k_values=[5]",
        "training.max_epochs=5", "project_name=smoke", "save_folder=data/smoke"]


def test_tiny_experiment_end_to_end(workdir):
    result = Experiment(compose_config("base", TINY)).run_experiment()
    name = result["name"]
    for kind in ("config", "summaries", "user", "item", "ratings"):
        assert list((workdir / "data" / "smoke").glob(f"{kind}_{name}.*")), kind

    params, metrics, items, users = load_project_data(["smoke"], base_data_path=str(workdir / "data"))
    assert len(params) == 1 and len(metrics) == 2
    assert len(users) == 2 * 15 and len(items) == 2 * 12

    gamma = gamma_estimates_grouped(users, items, group_cols=["experiment_name", "replication"])
    assert len(gamma) == 2
    assert np.isfinite(gamma[["gamma_item", "gamma_user"]].to_numpy()).all()

    # a finished experiment is skipped, not rerun
    assert Experiment(compose_config("base", TINY)).run_experiment()["status"] == "skipped"
