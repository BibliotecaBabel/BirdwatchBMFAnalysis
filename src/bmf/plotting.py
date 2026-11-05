import numpy as np
import matplotlib.pyplot as plt


ESTIMATOR_STYLES = {
    'gamma_item': ('o', r'$\hat{\gamma}_{\mathrm{item}}$ (item-based)', '#009E73'),
    'gamma_user': ('^', r'$\hat{\gamma}_{\mathrm{user}}$ (user-based)', '#E69F00'),
    'gamma_neutral': ('s', r'$\hat{\gamma}_{\mathrm{neutral}}$ (neutral-agent)', '#56B4E9'),
}


def plot_gamma_combined_estimators(
    gamma_df,
    ideal_slope=0.5,
    ideal_intercept=0,
    reg_type='l2',
    x_col='dg.agent_mean',
    x_label=r'$\mu_u$ (mean reviewer ideology)',
    colors=('#009E73', '#E69F00'),
    methods=('gamma_item', 'gamma_user'),
    label_size=18,
    tick_size=15,
    legend_size=15,
    ref_line=None,
):
    """
    Single-panel figure with gamma estimators vs mu_u, filtered to one
    regularization type, with the theoretical prediction overlaid.

    Parameters
    ----------
    gamma_df : pd.DataFrame
        Output of gamma_estimates_grouped, merged with params.
    ideal_slope : float
        Slope of the theoretical gamma* line (default 0.5 for L2).
    ideal_intercept : float
        Intercept of the theoretical gamma* line.
    reg_type : str
        Regularization type to filter to (default 'l2').
    colors : tuple of str
        Colors for the estimators, positionally matched to ``methods``.  Falls
        back to the per-estimator default in ESTIMATOR_STYLES when shorter than
        ``methods``.
    methods : sequence of str
        Which gamma estimators to draw.  Defaults to the item- and user-based
        pair used in the paper figures; pass e.g.
        ``('gamma_item', 'gamma_user', 'gamma_neutral')`` to include the
        regression-free neutral-agent estimator.
    label_size, tick_size, legend_size : int
        Font sizes for axis labels, ticks, and legend.
    ref_line : tuple (x, y, label) or None
        When given, this curve is drawn as the dashed reference instead of
        the straight line ``ideal_slope * x + ideal_intercept``.  Used for the
        exact anchor of Theorem 3, which is not linear in mu_u.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if 'model.regularization_type' in gamma_df.columns:
        plot_data = gamma_df[gamma_df['model.regularization_type'] == reg_type]
    else:
        plot_data = gamma_df

    methods = list(methods)
    markers = [ESTIMATOR_STYLES.get(m, ('o', m, '#333333'))[0] for m in methods]
    labels = [ESTIMATOR_STYLES.get(m, ('o', m, '#333333'))[1] for m in methods]
    colors = list(colors) + [
        ESTIMATOR_STYLES.get(m, ('o', m, '#333333'))[2]
        for m in methods[len(colors):]
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    x_all = []

    for method, marker, label, color in zip(methods, markers, labels, colors):
        data = plot_data.dropna(subset=[method, x_col])
        if data.empty:
            continue
        grouped = data.groupby(x_col)[method].agg(['mean', 'std']).reset_index()
        x_all.extend(grouped[x_col].tolist())
        ax.errorbar(
            grouped[x_col], grouped['mean'],
            yerr=grouped['std'], fmt=f'{marker}-', color=color,
            capsize=4, capthick=1.5, markersize=7, linewidth=2,
            label=label,
        )

    if ref_line is not None:
        ref_x, ref_y, ref_label = ref_line
        ax.plot(ref_x, ref_y, 'k--', linewidth=1.5, label=ref_label)
    elif x_all:
        x_range = np.linspace(min(x_all), max(x_all), 100)
        ax.plot(x_range, ideal_slope * x_range + ideal_intercept,
                'k--', linewidth=1.5, label=r'$\gamma^* = \mu_u/2$')

    ax.set_xlabel(x_label, fontsize=label_size)
    ax.set_ylabel(r'Estimated $\hat{\gamma}$', fontsize=label_size)
    ax.tick_params(axis='both', labelsize=tick_size)
    ax.legend(fontsize=legend_size)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig
