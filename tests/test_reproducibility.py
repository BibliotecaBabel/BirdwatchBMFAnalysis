"""Experiment names and seeds: what makes a rerun reproduce the paper's draws."""
import pytest

from bmf.experiment import RECORDED_SEED_OFFSETS, Experiment

from conftest import compose_config, sweep_jobs

# sweeps of the NeurIPS paper whose saved runs the current configs reproduce
PAPER_SWEEPS = [
    "agent-mean-sweep",
    "item-mean-sweep",
    "kappa-sweep",
    "robustness-distortion-500",
    "misspecification-sweep",
    "confounder-sweep",
]
MAIN_MU2 = ["dg.user_distribution.mean=2.0", "model.regularization_type=l2",
            "model.reg_lambda=0.01", "model.reg_intercept_multiplier=1.0"]


@pytest.mark.parametrize("name", PAPER_SWEEPS)
def test_every_paper_experiment_has_a_recorded_seed_offset(name, workdir):
    jobs = sweep_jobs(name)
    names = {Experiment(cfg)._build_experiment_name() for cfg in jobs}
    assert len(names) == len(jobs) > 0
    assert names <= set(RECORDED_SEED_OFFSETS)


def test_seeds_of_a_paper_experiment(workdir):
    exp = Experiment(compose_config("agent-mean-sweep", MAIN_MU2))
    assert exp._build_experiment_name() == "exp_ecd9344b"
    assert exp._compute_component_seeds(0) == {"data": 2642618813, "mechanism": 1700015148}
    assert exp._compute_component_seeds(1) == {"data": 2947929056, "mechanism": 3991352459}


@pytest.mark.parametrize("name, overrides", [
    ("agent-mean-sweep", MAIN_MU2),                 # offset from the record
    ("agent-mean-sweep-binary", MAIN_MU2),          # offset from the config hash
])
def test_logging_credentials_change_neither_names_nor_seeds(name, overrides, workdir, monkeypatch):
    plain = Experiment(compose_config(name, overrides))
    monkeypatch.setenv("COMET_API_KEY", "some-key")
    monkeypatch.setenv("COMET_WORKSPACE", "some-workspace")
    keyed = Experiment(compose_config(name, overrides))
    assert keyed._build_experiment_name() == plain._build_experiment_name()
    assert keyed._compute_component_seeds(0) == plain._compute_component_seeds(0)


def test_unrecorded_experiments_get_distinct_deterministic_seeds(workdir):
    jobs = sweep_jobs("agent-mean-sweep-binary")
    exps = [Experiment(cfg) for cfg in jobs]
    assert not {e._build_experiment_name() for e in exps} & set(RECORDED_SEED_OFFSETS)
    seeds = [e._compute_data_seed(0) for e in exps]
    assert seeds == [Experiment(cfg)._compute_data_seed(0) for cfg in jobs]
    assert len(set(seeds)) == len(seeds)
    assert exps[0]._compute_data_seed(1) - exps[0]._compute_data_seed(0) == 100000
