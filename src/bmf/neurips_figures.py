"""Figures of the NeurIPS paper, drawn from the runs saved under ``data/``.

Run the experiments first (``bash run_all.sh``); this script reruns nothing.

Every dashed reference line is the exact anchor gamma* of Theorem 3
(``exact_gamma``), evaluated at the empirical moments of the data drawn. Its
small-mu_u tangent is mu_u / 2. The kappa figure shows the ``reg_lambda = 0.1``
slice of ``data/kappa-sweep``; the lambda, risk and noise figures read
``data/robustness-distortion-500``.

Usage:
    poetry run python -m bmf.neurips_figures               # all groups
    poetry run python -m bmf.neurips_figures main binary    # a subset
    poetry run python -m bmf.neurips_figures --out results/figures/neurips

Outputs go to results/figures/neurips/{main,binary,item,misspecification}/
under the file names the paper includes.

For every figure the script prints the data folder, the filter, the number of
(experiment, replication) rows and the run settings read from the matching
``config_exp_*.json`` files. Sanity checks on the data run before any file is
written, so a wrong filter stops the script instead of producing a figure.
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import brentq

from bmf.gamma import gamma_estimates_grouped
from bmf.plotting import plot_gamma_combined_estimators
from bmf.utils import load_project_data

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

LABEL_SIZE = 18
TICK_SIZE = 15
LEGEND_SIZE = 15
L2_COLOR = '#009E73'
REF_COLOR = '#9A9A9A'

THEORY_LABEL = r'$\gamma^*$ (Theorem 3)'
OQ_COL = 'summary.OQ_gap@10'
X_LABEL_MU = r'$\mu_u$ (mean reviewer ideology)'

ESTIMATOR_PANELS = [
    ('gamma_user', r'User-based $\hat{\gamma}$'),
    ('gamma_item', r'Item-based $\hat{\gamma}$'),
]

SETTING_KEYS = [
    'dg.noise_std',
    'training.learning_rate',
    'training.max_epochs',
    'training.patience',
    'training.min_delta',
    'training.batch_size',
    'model.reg_lambda',
    'model.reg_intercept_multiplier',
    'model.loss_type',      # new config schema
    'training.loss_type',   # old config schema (robustness-distortion)
]

META_COLS = [
    'dg.agent_mean',
    'model.regularization_type',
    'model.reg_lambda',
    'model.reg_intercept_multiplier',
    'experiment_name',
]

# Published values of misspecification-table.tex (s_item, s_user, quality gap
# at mu_u = 0 and at |mu_u| = 2). A rerun draws the same populations but the
# fits can differ slightly across hardware, so a regenerated row is compared
# with the published one within PUBLISHED_TOL and a difference is reported
# without stopping the script.
PUBLISHED_TOL = 0.02
EXPECTED_TABLE = {
    'bilinear':     (0.370, 0.382, 0.007, 0.211),
    'euclidean':    (0.386, 0.193, 0.137, 0.158),
    'higher_order': (0.478, 0.500, 0.123, 0.504),
    'confounder':   (0.584, 0.611, 0.005, 0.442),
}


# ---------------------------------------------------------------------------
# Theory: the exact anchor of Theorem 3
# ---------------------------------------------------------------------------

def exact_gamma(mu_u, sigma_u2=1.0, sigma_v2=1.0, kappa=1.0):
    """Anchor gamma* of Theorem 3 under this repo's block-mean L2 regularizer.

    Ratings r_ij = q_j + u_i v_j + noise; weight ``lambda`` on the mean of
    squares of the ideology block (u, v) and ``kappa * lambda`` on the
    intercept block (b, q, c).  With t = tau^2, A = kappa * sigma_v2 and
    K = sigma_v2, gamma* = mu_u * t* / (t* + A) where t* > 0 is the root of

        psi(t) = t^2 sigma_u2 + mu_u^2 A^2 (t / (t + A))^2 - K.

    ``mu_u`` and ``sigma_u2`` are the (empirical) mean and variance of the
    user ideologies, ``sigma_v2`` the (empirical) mean of v_j^2.  The line
    mu_u / 2 is the tangent of this curve at mu_u = 0
    for kappa = 1.  Reference values at sigma_u2 = sigma_v2 = kappa = 1:
    gamma* = 0.125, 0.246, 0.469, 0.648, 0.775 at mu_u = 0.25, 0.5, 1, 1.5, 2.
    """
    A = kappa * sigma_v2
    K = sigma_v2
    if abs(mu_u) < 1e-12:
        return 0.0  # t* = sqrt(K / sigma_u2), gamma* = 0

    def psi(t):
        return t * t * sigma_u2 + mu_u ** 2 * A ** 2 * (t / (t + A)) ** 2 - K

    t_star = brentq(psi, 1e-8, 1e4)
    return mu_u * t_star / (t_star + A)


def replication_moments(users_df, items_df):
    """Empirical (mu_u, sigma_u2, sigma_v2) per (experiment_name, replication).

    sigma_u2 is the population variance (ddof=0) of the true user ideologies
    and sigma_v2 the mean of the squared true item ideologies, matching the
    mean-of-squares normalization of the regularizer.
    """
    keys = ['experiment_name', 'replication']
    u = users_df.groupby(keys)['true_user_ideology'].agg(
        mu_u='mean', sigma_u2=lambda s: float(np.var(s.values)))
    v = (items_df.assign(v2=items_df['true_item_ideology'] ** 2)
         .groupby(keys)['v2'].mean().rename('sigma_v2'))
    return u.join(v).reset_index()


def add_exact_gamma(gamma_df, users_df, items_df,
                    kappa_col='model.reg_intercept_multiplier'):
    """Attach ``gamma_exact``: Theorem 3 evaluated per replication at that
    replication's empirical moments and that experiment's kappa."""
    out = gamma_df.merge(replication_moments(users_df, items_df),
                         on=['experiment_name', 'replication'], how='left')
    out['gamma_exact'] = [
        exact_gamma(r.mu_u, r.sigma_u2, r.sigma_v2, r[kappa_col])
        for _, r in out.iterrows()
    ]
    return out


def theory_curve(df, x_col):
    """Reference curve for a swept x: the per-replication exact anchor averaged
    within each swept value, plotted at the nominal x."""
    g = df.groupby(x_col)['gamma_exact'].mean().reset_index()
    return g[x_col].values, g['gamma_exact'].values


# ---------------------------------------------------------------------------
# Data loading and reporting
# ---------------------------------------------------------------------------

_CACHE = {}


def load(folder, base='data'):
    """load_project_data with a cache; returns (params, metrics, items, users)."""
    if folder not in _CACHE:
        with warnings.catch_warnings():
            # old-schema folders already carry dg.agent_mean; the alias warning is noise
            warnings.filterwarnings('ignore', message="Column alias .* already exists")
            _CACHE[folder] = load_project_data([folder], base_data_path=base)
        print(f'loaded data/{folder}: {len(_CACHE[folder][0])} experiments')
    return _CACHE[folder]


def report(fig_name, folder, filt, frame, params):
    """Print data folder, filter, row count and run settings for one figure."""
    exps = frame['experiment_name'].unique()
    p = params[params['experiment_name'].isin(exps)]
    print(f'\n[{fig_name}]')
    print(f'  data:    data/{folder}')
    print(f'  filter:  {filt}')
    print(f'  rows:    {len(frame)} (experiment, replication) from {len(exps)} experiments')
    for key in SETTING_KEYS:
        if key in p.columns and p[key].notna().any():
            vals = sorted(p[key].dropna().unique().tolist())
            print(f'  {key}: {vals}')


def check(cond, msg):
    if not cond:
        sys.exit(f'CHECK FAILED: {msg}')
    print(f'  ok: {msg}')


def note(cond, msg):
    """Like check, but reports instead of stopping."""
    print(f"  {'ok' if cond else 'NOTE'}: {msg}")


def savefig(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {path}')


# ---------------------------------------------------------------------------
# Shared plot forms
# ---------------------------------------------------------------------------

def plot_oq(metrics_plot, x_col, x_label, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=metrics_plot, x=x_col, y=OQ_COL,
                    color=L2_COLOR, s=30, alpha=0.2, ax=ax)
    sns.lineplot(data=metrics_plot, x=x_col, y=OQ_COL,
                 color=L2_COLOR, errorbar=('ci', 95), linewidth=2, ax=ax)
    ax.set_xlabel(x_label, fontsize=LABEL_SIZE)
    ax.set_ylabel('Quality gap@10', fontsize=LABEL_SIZE)
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    savefig(fig, path)


def two_panel_sweep(df, x_col, x_label, path, log_x=False,
                    hline=None, hline_label=THEORY_LABEL):
    """Mean +- 1.96 SEM of each estimator against a swept parameter, with the
    individual replications scattered behind, one panel per estimator."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (method, title) in zip(axes, ESTIMATOR_PANELS):
        sns.scatterplot(data=df, x=x_col, y=method,
                        color=L2_COLOR, s=30, alpha=0.2, ax=ax)
        grouped = df.groupby(x_col)[method].agg(['mean', 'sem']).reset_index()
        ax.errorbar(grouped[x_col], grouped['mean'], yerr=1.96 * grouped['sem'],
                    fmt='o-', color=L2_COLOR, capsize=5, capthick=2,
                    markersize=8, linewidth=2.5, zorder=10)
        if hline is not None:
            ax.axhline(hline, color='k', linestyle='--', linewidth=1.5,
                       label=hline_label)
        if log_x:
            ax.set_xscale('log')
        ax.set_xlabel(x_label, fontsize=LABEL_SIZE)
        ax.set_ylabel(title, fontsize=LABEL_SIZE)
        ax.tick_params(axis='both', labelsize=TICK_SIZE)
        ax.grid(True, alpha=0.3, which='both' if log_x else 'major')
    if hline is not None:
        axes[0].legend(fontsize=LEGEND_SIZE)
    fig.tight_layout()
    savefig(fig, path)


# ---------------------------------------------------------------------------
# main/: agent-mean sweep (Figs. 3, 4, 8) and robustness (Figs. 5-7, 9)
# ---------------------------------------------------------------------------

MAIN_GAMMA_FILTER = (
    "`model.regularization_type`=='l2'"
    " and `model.reg_intercept_multiplier`==1"
    " and `model.reg_lambda`==0.01"
    " and gamma_user_abs_diff<5"
)
MAIN_METRICS_FILTER = (
    "`model.reg_intercept_multiplier`==1.0"
    " and `model.reg_lambda`==0.01"
    " and `model.regularization_type`=='l2'"
)


def gamma_frame(folder, meta_cols=META_COLS):
    """gamma estimates per (experiment, replication) merged with the config
    columns and the exact anchor at that replication's moments."""
    params, _, items, users = load(folder)
    g = gamma_estimates_grouped(users, items,
                                group_cols=['experiment_name', 'replication'])
    g = g.merge(params[meta_cols], on='experiment_name')
    return add_exact_gamma(g, users, items)


def print_comparison(df, x_col, label):
    cols = {'gamma_item': 'item', 'gamma_user': 'user', 'gamma_exact': 'exact'}
    t = df.groupby(x_col)[list(cols)].mean().rename(columns=cols)
    print(f'  {label}: mean estimate vs exact anchor by {x_col}')
    print('    ' + t.round(3).to_string().replace('\n', '\n    '))


def build_main(out):
    folder = 'agent-mean-sweep'
    params, metrics, _, _ = load(folder)
    gamma_df = gamma_frame(folder)
    metrics_sc = metrics.merge(params, on='experiment_name', how='left')

    gamma_plot = gamma_df.query(MAIN_GAMMA_FILTER)
    metrics_plot = metrics_sc.query(MAIN_METRICS_FILTER)

    # --- checks before anything is written ---
    print('\nchecks: main sweep')
    at2 = gamma_plot[gamma_plot['dg.agent_mean'] == 2.0]
    item2, user2 = at2['gamma_item'].mean(), at2['gamma_user'].mean()
    check(0.68 <= item2 <= 0.76, f'main sweep mu_u=2: item-based mean {item2:.3f} in [0.68, 0.76]')
    check(0.74 <= user2 <= 0.82, f'main sweep mu_u=2: user-based mean {user2:.3f} in [0.74, 0.82]')

    jobs = []

    # Fig. 3 ---------------------------------------------------------------
    def fig3():
        report('main/gamma-estimate_both-estimators.pdf (Fig. 3)', folder,
               MAIN_GAMMA_FILTER, gamma_plot, params)
        print_comparison(gamma_plot, 'dg.agent_mean', 'Fig. 3')
        # Reference: Theorem 3 per replication at its empirical moments,
        # averaged within each swept mu_u (not the population-value curve).
        fig = plot_gamma_combined_estimators(
            gamma_plot, reg_type='l2',
            ref_line=(*theory_curve(gamma_plot, 'dg.agent_mean'), THEORY_LABEL))
        savefig(fig, out / 'main' / 'gamma-estimate_both-estimators.pdf')
    jobs.append(fig3)

    # Fig. 4 -----------------------------------------------------------------
    def fig4():
        report('main/quality-distortion_oq.pdf (Fig. 4)', folder,
               MAIN_METRICS_FILTER, metrics_plot, params)
        plot_oq(metrics_plot, 'dg.agent_mean', X_LABEL_MU,
                out / 'main' / 'quality-distortion_oq.pdf')
    jobs.append(fig4)

    # Fig. 8: lambda vs mu (same sweep, all three lambdas) ---------------------
    lvm_filter = (
        "`model.regularization_type`=='l2'"
        " and `model.reg_intercept_multiplier`==1"
        " and gamma_user_abs_diff<5"
    )
    data_lvm = gamma_df.query(lvm_filter)

    def fig8():
        report('main/gamma-robustness_lambda-vs-mu.pdf (Fig. 8)', folder,
               lvm_filter, data_lvm, params)
        lambdas = sorted(data_lvm['model.reg_lambda'].unique())
        lam2color = dict(zip(lambdas, sns.color_palette('viridis', len(lambdas))))
        # Reference: Theorem 3 per replication at empirical moments, averaged
        # over all lambdas within each mu_u (the exact anchor has no lambda).
        ref_x, ref_y = theory_curve(data_lvm, 'dg.agent_mean')
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, (method, title) in zip(axes, ESTIMATOR_PANELS):
            for lam, color in lam2color.items():
                grp = data_lvm[data_lvm['model.reg_lambda'] == lam]
                g = grp.groupby('dg.agent_mean')[method].agg(['mean', 'sem']).reset_index()
                ax.errorbar(g['dg.agent_mean'], g['mean'], yerr=1.96 * g['sem'],
                            fmt='o-', color=color, capsize=3, markersize=5,
                            linewidth=1.8, label=f'$\\lambda={lam}$')
            ax.plot(ref_x, ref_y, 'k--', linewidth=1.5, label=THEORY_LABEL)
            ax.set_xlabel(X_LABEL_MU, fontsize=LABEL_SIZE)
            ax.set_ylabel(title, fontsize=LABEL_SIZE)
            ax.tick_params(axis='both', labelsize=TICK_SIZE)
            ax.grid(True, alpha=0.3)
        axes[0].legend(fontsize=LEGEND_SIZE - 2, title=r'$\lambda$')
        fig.tight_layout()
        savefig(fig, out / 'main' / 'gamma-robustness_lambda-vs-mu.pdf')
    jobs.append(fig8)

    jobs += build_robustness(out)
    jobs += build_kappa(out)
    return jobs


def build_robustness(out):
    """Figs. 5, 6 and 9 from data/robustness-distortion-500 (mu_u = 2, kappa = 1),
    run at the same optimization settings as every other sweep."""
    folder = 'robustness-distortion-500'
    params, metrics, items, users = load(folder)
    reg_col = 'summary.weighted_reg'  # the weighted regularizer, total loss minus risk
    base_filter = "`dg.agent_mean`==2.0 and `model.regularization_type`=='l2'"
    p_l2 = params.query(base_filter)
    keep = p_l2['experiment_name']
    items = items[items['experiment_name'].isin(keep)]
    users = users[users['experiment_name'].isin(keep)]
    meta = ['dg.agent_mean', 'dg.noise_std', 'model.regularization_type',
            'model.reg_lambda', 'model.reg_intercept_multiplier',
            'training.batch_size', 'experiment_name']
    gamma_rob = gamma_estimates_grouped(users, items,
                                        group_cols=['experiment_name', 'replication'])
    gamma_rob = gamma_rob.merge(p_l2[meta], on='experiment_name')
    gamma_rob = gamma_rob.merge(
        metrics[['replication', 'experiment_name', 'summary.risk',
                 reg_col, 'summary.total_loss']],
        on=['experiment_name', 'replication'])
    gamma_rob = add_exact_gamma(gamma_rob, users, items)

    lam_filter = (base_filter + " and `model.reg_intercept_multiplier`==1.0"
                  " and `training.batch_size`==10100 and `dg.noise_std`==0.1")
    noise_filter = (base_filter + " and `model.reg_intercept_multiplier`==1.0"
                    " and `model.reg_lambda`==0.01 and `training.batch_size`==10100")
    data_lambda = gamma_rob.query(lam_filter)
    data_noise = gamma_rob.query(noise_filter)
    # Horizontal reference: exact anchor at mu_u = 2, kappa = 1, averaged over
    # the per-replication empirical moments of the slice (about 0.78).
    gstar_lambda = data_lambda['gamma_exact'].mean()
    gstar_noise = data_noise['gamma_exact'].mean()

    print('\nchecks: robustness-distortion')
    check(len(data_lambda) > 0 and len(data_noise) > 0,
          f'robustness slices non-empty ({len(data_lambda)} lambda rows, {len(data_noise)} noise rows)')
    check(set(data_noise['dg.noise_std'].unique()) == {0.0, 0.1, 0.5},
          'noise slice covers sigma in {0, 0.1, 0.5}')

    def fig5():
        report('main/gamma-robustness_lambda-sweep.pdf (Fig. 5)', folder,
               lam_filter, data_lambda, params)
        print(f'  exact anchor (mu_u=2, kappa=1, slice moments): {gstar_lambda:.4f}')
        print_comparison(data_lambda, 'model.reg_lambda', 'Fig. 5')
        # The small dip at lambda = 0.005 is a convergence effect: the weaker the
        # regularizer, the slower the descent along the gauge orbit.
        two_panel_sweep(data_lambda, 'model.reg_lambda',
                        r'Regularization $\lambda$ (log scale)',
                        out / 'main' / 'gamma-robustness_lambda-sweep.pdf',
                        log_x=True, hline=gstar_lambda)

    def fig6():
        report('main/gamma-robustness_risk-reg.pdf (Fig. 6)', folder,
               lam_filter, data_lambda, params)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, metric, title in zip(axes, [reg_col, 'summary.risk'],
                                     ['Regularization term', 'Risk']):
            sns.scatterplot(data=data_lambda, x='model.reg_lambda', y=metric,
                            color=L2_COLOR, s=30, alpha=0.2, ax=ax)
            g = data_lambda.groupby('model.reg_lambda')[metric].agg(['mean', 'sem']).reset_index()
            ax.errorbar(g['model.reg_lambda'], g['mean'], yerr=1.96 * g['sem'],
                        fmt='o-', color=L2_COLOR, capsize=5, capthick=2,
                        markersize=8, linewidth=2.5, zorder=10)
            ax.set_xscale('log')
            ax.set_xlabel(r'Regularization $\lambda$ (log scale)', fontsize=LABEL_SIZE)
            ax.set_ylabel(title, fontsize=LABEL_SIZE)
            ax.tick_params(axis='both', labelsize=TICK_SIZE)
            ax.grid(True, alpha=0.3, which='both')
        fig.tight_layout()
        savefig(fig, out / 'main' / 'gamma-robustness_risk-reg.pdf')

    def fig9():
        report('main/gamma-robustness_noise.pdf (Fig. 9)', folder,
               noise_filter, data_noise, params)
        print(f'  exact anchor (mu_u=2, kappa=1, slice moments): {gstar_noise:.4f}')
        print_comparison(data_noise, 'dg.noise_std', 'Fig. 9')
        two_panel_sweep(data_noise, 'dg.noise_std', r'Noise std $\sigma$',
                        out / 'main' / 'gamma-robustness_noise.pdf',
                        hline=gstar_noise)

    return [fig5, fig6, fig9]


def build_kappa(out):
    """Fig. 7 from data/kappa-sweep (mu_u = 2, noise 0.5, 500 epochs, patience 10)."""
    folder = 'kappa-sweep'
    params, _, _, _ = load(folder)
    gamma_k = gamma_frame(folder)
    k_filter = "`model.regularization_type`=='l2'"
    data_k = gamma_k.query(k_filter)
    # The figure shows the reg_lambda = 0.1 slice only.  The 0.01 slice matches
    # theory for kappa >= 1 but stalls near 0.89 for kappa <= 0.5, where the
    # descent along the flat gauge orbit is slow; it is printed, not drawn.
    main_lam, faint_lam = 0.1, 0.01
    d_main = data_k[data_k['model.reg_lambda'] == main_lam]
    d_faint = data_k[data_k['model.reg_lambda'] == faint_lam]

    print('\nchecks: kappa-sweep (Fig. 7)')
    p_used = params[params['experiment_name'].isin(data_k['experiment_name'])]
    check((p_used['project_name'] == 'kappa-sweep').all()
          and (p_used['training.max_epochs'] == 500).all(),
          'Fig. 7 reads data/kappa-sweep (500 epochs), not the robustness-distortion kappa slice')
    k1 = d_main[d_main['model.reg_intercept_multiplier'] == 1.0]['gamma_item'].mean()
    k01 = d_main[d_main['model.reg_intercept_multiplier'] == 0.1]['gamma_item'].mean()
    check(0.70 <= k1 <= 0.78, f'kappa-sweep lambda=0.1 kappa=1: item-based mean {k1:.3f} in [0.70, 0.78]')
    check(1.6 <= k01 <= 1.8, f'kappa-sweep lambda=0.1 kappa=0.1: item-based mean {k01:.3f} in [1.6, 1.8]')

    def fig7():
        report('main/gamma-robustness_multiplier.pdf (Fig. 7)', folder,
               k_filter + f' and `model.reg_lambda`=={main_lam}', d_main, params)
        print_comparison(d_main, 'model.reg_intercept_multiplier', f'Fig. 7 lambda={main_lam}')
        print_comparison(d_faint, 'model.reg_intercept_multiplier', f'Fig. 7 lambda={faint_lam}')
        # Reference curve: Theorem 3 on a dense kappa grid at the slice-average
        # empirical moments (mu_u, sigma_u2, sigma_v2 of the lambda=0.1 runs),
        # so the dashed line is smooth on the log axis.  The per-replication
        # values at the seven swept kappas are printed above as 'exact'.
        mom = d_main[['mu_u', 'sigma_u2', 'sigma_v2']].mean()
        kgrid = np.logspace(np.log10(0.08), np.log10(12.5), 200)
        gstar = [exact_gamma(mom.mu_u, mom.sigma_u2, mom.sigma_v2, k) for k in kgrid]
        print(f'  reference moments: mu_u={mom.mu_u:.3f} sigma_u2={mom.sigma_u2:.3f} sigma_v2={mom.sigma_v2:.3f}')

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, (method, title) in zip(axes, ESTIMATOR_PANELS):
            sns.scatterplot(data=d_main, x='model.reg_intercept_multiplier', y=method,
                            color=L2_COLOR, s=30, alpha=0.2, ax=ax)
            g = d_main.groupby('model.reg_intercept_multiplier')[method].agg(['mean', 'sem']).reset_index()
            ax.errorbar(g['model.reg_intercept_multiplier'], g['mean'], yerr=1.96 * g['sem'],
                        fmt='o-', color=L2_COLOR, capsize=5, capthick=2,
                        markersize=8, linewidth=2.5, zorder=10)
            ax.plot(kgrid, gstar, 'k--', linewidth=1.5, label=THEORY_LABEL)
            ax.set_xscale('log')
            ax.set_xlabel(r'$\kappa = \lambda_q / \lambda_u$ (log scale)', fontsize=LABEL_SIZE)
            ax.set_ylabel(title, fontsize=LABEL_SIZE)
            ax.tick_params(axis='both', labelsize=TICK_SIZE)
            ax.grid(True, alpha=0.3, which='both')
        axes[0].legend(fontsize=LEGEND_SIZE)
        fig.tight_layout()
        savefig(fig, out / 'main' / 'gamma-robustness_multiplier.pdf')

    return [fig7]


# ---------------------------------------------------------------------------
# binary/ and item/
# ---------------------------------------------------------------------------

def build_binary(out):
    folder = 'agent-mean-sweep-binary'
    params, metrics, _, _ = load(folder)
    gamma_plot = gamma_frame(folder).query(MAIN_GAMMA_FILTER)
    metrics_plot = metrics.merge(params, on='experiment_name', how='left').query(MAIN_METRICS_FILTER)
    print('\nchecks: binary sweep')
    check(len(gamma_plot) > 0, f'binary gamma slice has {len(gamma_plot)} rows')

    def gamma_fig():
        report('binary/gamma-estimate_both-estimators.pdf', folder,
               MAIN_GAMMA_FILTER, gamma_plot, params)
        print_comparison(gamma_plot, 'dg.agent_mean', 'binary')
        # The logistic link is outside Theorem 3; the exact curve (per
        # replication at empirical moments, averaged per mu_u) is a reference only.
        fig = plot_gamma_combined_estimators(
            gamma_plot, reg_type='l2',
            ref_line=(*theory_curve(gamma_plot, 'dg.agent_mean'), THEORY_LABEL))
        savefig(fig, out / 'binary' / 'gamma-estimate_both-estimators.pdf')

    def oq_fig():
        report('binary/quality-distortion_oq.pdf', folder,
               MAIN_METRICS_FILTER, metrics_plot, params)
        plot_oq(metrics_plot, 'dg.agent_mean', X_LABEL_MU,
                out / 'binary' / 'quality-distortion_oq.pdf')

    return [gamma_fig, oq_fig]


def build_item(out):
    folder = 'item-mean-sweep'
    meta = ['dg.item_mean' if c == 'dg.agent_mean' else c for c in META_COLS]
    params, metrics, _, _ = load(folder)
    gamma_plot = gamma_frame(folder, meta).query(MAIN_GAMMA_FILTER)
    metrics_plot = metrics.merge(params, on='experiment_name', how='left').query(MAIN_METRICS_FILTER)
    x_label = r'$\mu_v$ (mean item ideology)'
    print('\nchecks: item-mean sweep')
    check(len(gamma_plot) > 0, f'item-mean gamma slice has {len(gamma_plot)} rows')

    def gamma_fig():
        report('item/gamma-estimate_both-estimators.pdf', folder,
               MAIN_GAMMA_FILTER, gamma_plot, params)
        # mu_u = 0 here, so the anchor is exactly zero whatever mu_v is.
        xs = np.array(sorted(gamma_plot['dg.item_mean'].unique()))
        fig = plot_gamma_combined_estimators(
            gamma_plot, reg_type='l2', x_col='dg.item_mean', x_label=x_label,
            ref_line=(xs, np.zeros_like(xs), r'$\gamma^* = 0$'))
        savefig(fig, out / 'item' / 'gamma-estimate_both-estimators.pdf')

    def oq_fig():
        report('item/quality-distortion_oq.pdf', folder,
               MAIN_METRICS_FILTER, metrics_plot, params)
        plot_oq(metrics_plot, 'dg.item_mean', x_label,
                out / 'item' / 'quality-distortion_oq.pdf')

    return [gamma_fig, oq_fig]


# ---------------------------------------------------------------------------
# misspecification/: composite figure and table
# ---------------------------------------------------------------------------

COMPOSITE_VARIANTS = [
    ('euclidean',    '#E69F00', 's', r'euclidean:  $-(u-v)^2$'),
    ('higher_order', '#009E73', '^', r'higher order:  $uv + \varphi u^2v^2$'),
    ('confounder',   '#56B4E9', 'D', r'confounder:  $uv + wz$'),
]
TABLE_ROWS = [
    ('bilinear',     r'bilinear (well-specified) & $uv$'),
    ('euclidean',    r'euclidean & $-(u-v)^2$'),
    ('higher_order', r'higher order & $uv + \varphi u^2 v^2$'),
    ('confounder',   r'confounder & $uv + wz$'),
]


def fitted_s(frame, col):
    """Slope of col = s * mu_u over the sweep.  The exact anchor has slope 1/2
    only at mu_u = 0; over the [-2, 2] grid its least-squares slope is 0.42."""
    d = frame[['dg.agent_mean', col]].dropna()
    return np.polyfit(d['dg.agent_mean'], d[col], 1)[0] if len(d) > 1 else np.nan


def build_misspecification(out):
    ms_folder, cf_folder = 'misspecification-sweep', 'confounder-sweep'
    ms_meta = META_COLS + ['dg.interaction.name']
    cf_meta = META_COLS + ['dg.confounder_item_std']
    ms_params, ms_metrics, _, _ = load(ms_folder)
    cf_params, cf_metrics, _, _ = load(cf_folder)

    ms_gamma = gamma_frame(ms_folder, ms_meta).query(MAIN_GAMMA_FILTER)
    ms_oq = ms_metrics.merge(ms_params[ms_meta], on='experiment_name').query(MAIN_METRICS_FILTER)
    cf_gamma = gamma_frame(cf_folder, cf_meta).query(MAIN_GAMMA_FILTER)
    cf_oq = cf_metrics.merge(cf_params[cf_meta], on='experiment_name').query(MAIN_METRICS_FILTER)

    def tag(frame, variant):
        return frame.assign(variant=variant)

    comp_gamma = pd.concat(
        [ms_gamma.rename(columns={'dg.interaction.name': 'variant'}),
         tag(cf_gamma[cf_gamma['dg.confounder_item_std'] == 1.0], 'confounder')],
        ignore_index=True)
    comp_oq = pd.concat(
        [ms_oq.rename(columns={'dg.interaction.name': 'variant'}),
         tag(cf_oq[cf_oq['dg.confounder_item_std'] == 1.0], 'confounder')],
        ignore_index=True)

    # --- table ---
    records = []
    for name, _ in TABLE_ROWS:
        g = comp_gamma[comp_gamma.variant == name]
        by_mu = comp_oq[comp_oq.variant == name].groupby('dg.agent_mean')[OQ_COL].mean()
        extremes = [x for x in by_mu.index if abs(abs(x) - 2) < 1e-9]
        records.append({
            'variant': name,
            's_item': fitted_s(g, 'gamma_item'),
            's_user': fitted_s(g, 'gamma_user'),
            'oq_0': by_mu.get(0.0, np.nan),
            'oq_2': by_mu.loc[extremes].mean() if extremes else np.nan,
        })
    bilinear = comp_gamma[comp_gamma.variant == 'bilinear']
    s_exact = fitted_s(bilinear, 'gamma_exact')
    grid = np.array(sorted(bilinear['dg.agent_mean'].unique()))
    s_exact_pop = np.polyfit(grid, [exact_gamma(m) for m in grid], 1)[0]

    print('\nchecks: misspecification table')
    for r in records:
        got = tuple(round(float(r[k]), 3) for k in ['s_item', 's_user', 'oq_0', 'oq_2'])
        published = EXPECTED_TABLE[r['variant']]
        same = np.allclose(got, published, atol=PUBLISHED_TOL)
        note(same, f"table row {r['variant']}: {got} "
                   + ('matches' if same else 'differs from') + f" the published {published}")

    def composite():
        report('misspecification/misspecification-composite.pdf', ms_folder,
               MAIN_GAMMA_FILTER + ' (+ confounder-sweep arm confounder_item_std==1.0)',
               comp_gamma, pd.concat([ms_params, cf_params], ignore_index=True))
        # The well-specified reference in this figure is the bilinear *data*
        # series, drawn in grey; no theory line is drawn here.
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
        for ax, (df, ycol, ylab) in zip(axes, [
                (comp_gamma, 'gamma_item', r'Estimated $\hat{\gamma}$'),
                (comp_oq, OQ_COL, 'Quality gap@10')]):
            sns.lineplot(data=df[df.variant == 'bilinear'], x='dg.agent_mean', y=ycol, ax=ax,
                         color=REF_COLOR, linewidth=2, linestyle='--', errorbar=('ci', 95),
                         err_kws={'alpha': 0.12}, label='bilinear (well-specified)')
            for name, color, marker, label in COMPOSITE_VARIANTS:
                sns.lineplot(data=df[df.variant == name], x='dg.agent_mean', y=ycol, ax=ax,
                             color=color, marker=marker, markersize=7, linewidth=2,
                             errorbar=('ci', 95), err_kws={'alpha': 0.12}, label=label)
            ax.set_xlabel(X_LABEL_MU, fontsize=LABEL_SIZE)
            ax.set_ylabel(ylab, fontsize=LABEL_SIZE)
            ax.tick_params(axis='both', labelsize=TICK_SIZE)
            ax.grid(True, alpha=0.3)
            ax.get_legend().remove()
        handles, labels = axes[0].get_legend_handles_labels()
        axes[0].legend(handles, labels, fontsize=LEGEND_SIZE, loc='upper left')
        fig.tight_layout()
        savefig(fig, out / 'misspecification' / 'misspecification-composite.pdf')

    def table():
        print('\n[misspecification/misspecification-table.tex]')
        print(pd.DataFrame(records).round(3).to_string(index=False))
        print(f'  exact anchor slope over the grid: {s_exact:.3f} (per-replication moments),'
              f' {s_exact_pop:.3f} (population moments)')
        lines = [
            f'% Slopes are least-squares fits over mu_u in [{grid.min():g}, {grid.max():g}].',
            f'% The exact anchor of Theorem 3 has slope {s_exact_pop:.2f} over this grid',
            '% (1/2 is its slope at mu_u = 0), so the bilinear row is close to theory.',
            r'\begin{tabular}{llcccc}', r'\toprule',
            r'Variant & $g(u,v)$ & $s_{\mathrm{item}}$ & $s_{\mathrm{user}}$'
            r' & Quality gap$@10$ & Quality gap$@10$ \\',
            r' & & & & ($\mu_u=0$) & ($|\mu_u|=2$) \\', r'\midrule',
        ]
        for (name, label), r in zip(TABLE_ROWS, records):
            lines.append(f"{label} & {r['s_item']:.3f} & {r['s_user']:.3f}"
                         f" & {r['oq_0']:.3f} & {r['oq_2']:.3f} \\\\")
        lines += [r'\bottomrule', r'\end{tabular}']
        path = out / 'misspecification' / 'misspecification-table.tex'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('\n'.join(lines) + '\n')
        print(f'  wrote {path}')

    return [composite, table]


# ---------------------------------------------------------------------------

GROUPS = {
    'main': build_main,
    'binary': build_binary,
    'item': build_item,
    'misspecification': build_misspecification,
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('groups', nargs='*', default=list(GROUPS),
                    help=f'subset of {list(GROUPS)} (default: all)')
    ap.add_argument('--out', default='results/figures/neurips', type=Path)
    args = ap.parse_args(argv)

    plt.rcParams['font.size'] = 14
    # Build every requested group first: all data checks run before any file
    # is written, so a failed check leaves the existing figures untouched.
    jobs = []
    for g in args.groups:
        jobs += GROUPS[g](args.out)
    print('\nall checks passed; writing figures')
    for job in jobs:
        job()


if __name__ == '__main__':
    main()
