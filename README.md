# Can Ideologues Agree on Quality? Simulation and figure code

Code for the simulations and figures of

> David Gamba, Seura Ha, Daniel M. Romero, Grant Schoenebeck.
> *Can Ideologues Agree on Quality? From Non-identifiable Latent Factors to
> Collective Outcomes.* NeurIPS 2026.

The paper studies biased matrix factorization (BMF) as used by Community Notes
(formerly Birdwatch) to score notes: ratings are modeled as
`r_ij = q_j + u_i * v_j + noise`, with a note quality `q_j`, a note ideology
`v_j` and a rater ideology `u_i`. The fit is identified only up to a gauge
transformation, and the regularizer picks the point on that orbit: an *anchor*
`gamma` that follows the mean rater ideology and distorts the estimated
qualities. This repository generates synthetic ratings, fits L2-regularized
BMF, estimates the anchor, and draws every simulation figure of the paper
against the exact prediction of Theorem 3.

## Installation

Python 3.12 and [Poetry](https://python-poetry.org/):

```bash
poetry install
```

`poetry.lock` pins the package versions the paper's runs were made with.

## Reproducing the figures

```bash
bash run_all.sh                              # 1. run the experiments
poetry run python -m bmf.neurips_figures     # 2. draw the figures
```

Step 1 runs seven sweeps, 131 experiments of 10 replications each, and writes
them under `data/` (about 2.1 GB). It took 2 hours 20 minutes with 8 parallel
jobs on an Apple M1 Pro laptop. Finished experiments are skipped, so an
interrupted run can simply be restarted. `N_JOBS=4 bash run_all.sh` changes
the number of parallel jobs. A single sweep is run with

```bash
poetry run python run.py -m --config-name kappa-sweep hydra.launcher.n_jobs=8
```

Step 2 reads those runs and writes the figures to `results/figures/neurips/`
under the file names the paper includes. It prints, for every figure, the data
folder, the filter, the number of rows and the run settings read from the
saved configs, and it checks the data before writing anything.
`poetry run python -m bmf.neurips_figures main` draws one group only.

| Sweep (`conf/<name>.yaml`) | Experiments | Figures (`results/figures/neurips/`) |
|---|---|---|
| `agent-mean-sweep` | 33 | `main/gamma-estimate_both-estimators.pdf`, `main/quality-distortion_oq.pdf`, `main/gamma-robustness_lambda-vs-mu.pdf` |
| `agent-mean-sweep-binary` | 11 | `binary/gamma-estimate_both-estimators.pdf`, `binary/quality-distortion_oq.pdf` |
| `item-mean-sweep` | 11 | `item/gamma-estimate_both-estimators.pdf`, `item/quality-distortion_oq.pdf` |
| `kappa-sweep` | 14 | `main/gamma-robustness_multiplier.pdf` |
| `robustness-distortion-500` | 18 | `main/gamma-robustness_lambda-sweep.pdf`, `main/gamma-robustness_risk-reg.pdf`, `main/gamma-robustness_noise.pdf` |
| `misspecification-sweep` | 33 | `misspecification/misspecification-composite.pdf`, `misspecification/misspecification-table.tex` |
| `confounder-sweep` | 11 | the confounder series and row of the same figure and table |

Four of the configs sweep more values than the figures show (266 experiments
in all); `run_all.sh` runs only the values the figures read and says how to
run the rest. The two illustrations of the running example (Figures 1 and 2)
are not simulation results and are not produced by this code.

## Experimental settings

All sweeps share `conf/base.yaml`: 101 raters and 100 notes; rater ideology
`N(mu_u, 1)`, note ideology and quality `N(0, 1)`; observation noise with
standard deviation 0.5; every rater rates every note. BMF has one ideology
factor and an L2 regularizer on the mean of squares of each parameter block,
with weight `lambda` on the ideology factors and `kappa * lambda` on the
intercepts. Ideology factors are Xavier-uniform initialized and intercepts
start at zero. Training is full-batch Adam at learning rate 0.05 for at most
500 epochs, with early stopping on the training loss (patience 10, minimum
improvement 1e-5). The binary sweep draws Bernoulli ratings through a logistic
link and trains with binary cross-entropy at learning rate 0.01.

## Reproducibility

An experiment's name and its seeds are derived from its resolved config,
including `project_name` and `save_folder`, so renaming either changes the
draws. `src/bmf/seed_offsets.json` records the seed offsets of the paper's
runs, and with it a rerun draws exactly the paper's synthetic populations.
Fitted values depend on hardware and library versions: a full rerun on the
laptop above matched the paper's anchors to within 0.004 and regenerated the
misspecification table identically.

The binary sweep is the exception. The paper's binary runs were made with an
earlier version of the configs, so a rerun draws new populations; its two
figures agree with the paper's up to sampling noise.

## Layout

```
run.py, run_all.sh        entry points for the experiments
conf/                     Hydra configs: base.yaml and one file per sweep
src/bmf/
  data_generators.py      synthetic raters, notes and ratings
  mechanism.py            the BMF model and its training loop
  experiment.py           replications, seeds, metrics, saving
  metrics.py              selection metrics (quality gap, welfare, ...)
  gamma.py                the item-based and user-based anchor estimators
  neurips_figures.py      the figures and the exact anchor of Theorem 3
  seed_offsets.json       seed record of the paper's experiments
tests/                    pytest suite: poetry run pytest
```

## License and citation

The code is released under the MIT license (see `LICENSE`). If you use it,
please cite the paper; `CITATION.cff` has the reference.
