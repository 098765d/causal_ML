# lasso_causal_pipeline.py
"""
Key stages
----------
Stage 1  LASSO (Y ~ X + M)  →  select candidate mediators / predictors.
Stage 2  For each selected mediator M, LASSO (M ~ X) → top X-drivers.
Stage 3  OLS (Y ~ all M + all X_drivers) → α, β, γ paths and derived IE, DE, TE.

Outputs
-------
* PNG coefficient plots for Stage 1 and each mediator-specific Stage 2 fit
* CSV file ``causal_effects.csv`` summarising α, β, γ, IE & TE with p-value bands

All random operations are seeded (``random_state``) for full determinism.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.patches import Patch
from sklearn.impute import KNNImputer
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ----------------------------------------------------------------------------
# CONFIGURATION & GLOBALS
# ----------------------------------------------------------------------------
RANDOM_STATE: int = 42
PLOT_DPI: int = 300
PLOT_WIDTH: float = 3.5  # inches
PLOT_BAR_HEIGHT: float = 0.2  # inches per bar

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)

# ----------------------------------------------------------------------------
# UTILITY FUNCTIONS
# ----------------------------------------------------------------------------

def wrap_label(text: str, width: int = 15) -> str:
    """Wrap *text* to the given *width* for compact horizontal-bar labels."""
    import textwrap

    return "\n".join(textwrap.wrap(text, width=width))


def plot_lasso_coefs(
    coef_series: pd.Series,
    highlight_vars: List[str],
    title: str,
    filename: Path,
    fmt: str = "{:.3f}",
    wrap_width: int = 15,
) -> None:
    """Horizontal bar plot of LASSO coefficients.

    Parameters
    ----------
    coef_series : pd.Series
        Index = variable names; values = coefficients.
    highlight_vars : List[str]
        Variables coloured differently (e.g., mediators).
    title : str
        Figure title.
    filename : Path
        File path where PNG will be saved.
    fmt : str, optional
        Python format spec for annotations.
    wrap_width : int, optional
        Character width for wrapping long variable names.
    """

    coef_series = coef_series.sort_values()
    if coef_series.empty:
        raise ValueError("No coefficients to plot.")

    # --- Figure sizing ---
    n_bars = coef_series.size
    fig_h = PLOT_BAR_HEIGHT * n_bars + 1.0  # + padding
    fig, ax = plt.subplots(figsize=(PLOT_WIDTH, fig_h), dpi=PLOT_DPI)

    # --- Bar plot ---
    colours = ["tab:blue" if v in highlight_vars else "tab:gray" for v in coef_series.index]
    bars = ax.barh(range(n_bars), coef_series.values, color=colours, alpha=0.8)

    # --- Axes & grid ---
    ax.set_yticks(range(n_bars))
    ax.set_yticklabels([wrap_label(lbl, wrap_width) for lbl in coef_series.index], fontsize=7)
    ax.axvline(0, color="black", lw=0.7)
    ax.set_xlabel("Coefficient value", fontsize=8)
    ax.set_title(title, fontsize=9, pad=6)

    # --- Annotations ---
    span = coef_series.max() - coef_series.min()
    offset = span * 0.02 if span != 0 else 0.02
    for bar, (coef, lbl) in zip(bars, coef_series.items()):
        ha = "left" if coef >= 0 else "right"
        x = coef + offset if coef >= 0 else coef - offset
        ax.text(x, bar.get_y() + bar.get_height() / 2, fmt.format(coef), va="center", ha=ha, fontsize=7)

    # Legend
    legend_handles = [
        Patch(color="tab:blue", label="Mediator (M)"),
        Patch(color="tab:gray", label="Background (X)"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=7, framealpha=0.8)

    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved plot → %s", filename)

# ----------------------------------------------------------------------------
# PIPELINE FUNCTIONS
# ----------------------------------------------------------------------------

def load_and_preprocess(csv_path: Path) -> tuple[pd.Series, pd.DataFrame]:
    """Load CSV, handle NA, encode categoricals, impute/scale numerics."""

    df = pd.read_csv(csv_path, sep=";")

    # Drop unused columns + rows with missing target
    df = (
        df.drop(columns=["G1", "G2"])
        .dropna(subset=["G3"])
    )

    y = df["G3"].astype(float)
    X = df.drop(columns=["G3"]).copy()

    # Label-encode categoricals
    for col in X.select_dtypes("object"):
        X[col] = LabelEncoder().fit_transform(X[col])

    # Impute + scale numerics only
    num_cols = X.select_dtypes(include=[np.number]).columns
    imputer = KNNImputer(n_neighbors=5)
    scaler = StandardScaler()
    X[num_cols] = imputer.fit_transform(X[num_cols])
    X[num_cols] = scaler.fit_transform(X[num_cols])

    return y, X


def stage1_lasso(y: pd.Series, X: pd.DataFrame, mediators: List[str]) -> List[str]:
    """Run LASSO on Y ~ X + M and return selected variable names."""
    features = mediators + [c for c in X.columns if c not in mediators]
    model = LassoCV(cv=10, random_state=RANDOM_STATE).fit(X[features], y)
    coefs = pd.Series(model.coef_, index=features).round(3)
    selected = coefs[coefs != 0].sort_values(key=abs).index.tolist()

    # Limit to top 8 (journal space)
    top_vars = selected[:8]
    logging.info("Stage 1 selected variables → %s", top_vars)

    # Plot
    plot_lasso_coefs(
        coef_series=coefs[top_vars],
        highlight_vars=[v for v in top_vars if v in mediators],
        title="Stage 1 LASSO (Y ~ X + M)",
        filename=Path("stage1_lasso_selected.png"),
    )
    return top_vars


def stage2_drivers(X: pd.DataFrame, mediators: List[str], background_vars: List[str]) -> Dict[str, List[str]]:
    """For each selected mediator, fit LASSO(M ~ X) and return top 5 drivers."""

    drivers: Dict[str, List[str]] = {}
    for m in mediators:
        lasso = LassoCV(cv=10, random_state=RANDOM_STATE).fit(X[background_vars], X[m])
        coefs = pd.Series(lasso.coef_, index=background_vars)
        top5 = coefs[coefs != 0].abs().nlargest(5).index.tolist()
        drivers[m] = top5
        logging.info("Mediator %s ← drivers %s", m, top5)

        if top5:
            plot_lasso_coefs(
                coef_series=coefs[top5],
                highlight_vars=[],
                title=f"Stage 2 LASSO (M ~ X), M='{m}'",
                filename=Path(f"stage2_lasso_{m}.png"),
            )
    return drivers


def stage3_effects(y: pd.Series, X: pd.DataFrame, drivers: Dict[str, List[str]]) -> pd.DataFrame:
    """Estimate α, β, γ paths and compute indirect, direct, total effects."""

    # α paths (M ~ X_drivers)
    alpha: Dict[tuple[str, str], tuple[float, float]] = {}
    for m, x_vars in drivers.items():
        if not x_vars:
            continue
        mod = sm.OLS(X[m], sm.add_constant(X[x_vars])).fit()
        for x_var in x_vars:
            alpha[(m, x_var)] = (mod.params[x_var], mod.pvalues[x_var])

    # β and γ in a single model Y ~ all M + all X_drivers
    all_M = sorted(set(drivers.keys()))
    all_X = sorted({x for xs in drivers.values() for x in xs})
    mod_y = sm.OLS(y, sm.add_constant(pd.concat([X[all_M], X[all_X]], axis=1))).fit()

    # Compile results
    records = []
    for (m, x), (a_coef, a_p) in alpha.items():
        beta = mod_y.params[m]
        beta_p = mod_y.pvalues[m]
        gamma = mod_y.params[x]
        gamma_p = mod_y.pvalues[x]

        ie = a_coef * beta
        te = gamma + ie

        records.append({
            "X": x,
            "M": m,
            "Y": "G3",
            "alpha": a_coef,
            "p_alpha": a_p,
            "beta": beta,
            "p_beta": beta_p,
            "gamma": gamma,
            "p_gamma": gamma_p,
            "IE": ie,
            "TE": te,
        })

    df_out = pd.DataFrame.from_records(records)
    df_out = df_out.round({"alpha": 3, "beta": 3, "gamma": 3, "IE": 3, "TE": 3})

    # p-value category helper
    def cat(p: float) -> str:
        return "<0.01" if p < 0.01 else "<0.05" if p < 0.05 else ">0.05"

    for col in ["p_alpha", "p_beta", "p_gamma"]:
        df_out[col] = df_out[col].apply(cat)

    return df_out

# ----------------------------------------------------------------------------
# MAIN GLUE
# ----------------------------------------------------------------------------

def main(csv: Path) -> None:
    y, X = load_and_preprocess(csv)

    mediator_vars = [
        "studytime", "failures", "paid", "activities", "romantic",
        "freetime", "Dalc", "Walc", "absences",
    ]
    background_vars = [c for c in X.columns if c not in mediator_vars]

    # --- Stage 1 ---
    selected = stage1_lasso(y, X, mediator_vars)
    selected_mediators = [v for v in selected if v in mediator_vars]

    # --- Stage 2 ---
    drv_map = stage2_drivers(X, selected_mediators, background_vars)

    # --- Stage 3 ---
    df_effects = stage3_effects(y, X, drv_map)
    df_effects.to_csv("causal_effects.csv", index=False)
    logging.info("Saved causal effects table → causal_effects.csv")
    logging.info("\n%s", df_effects)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Three-stage LASSO causal pipeline for student performance data.")
    parser.add_argument("csv", type=Path, help="Path to student-mat.csv (semicolon-delimited).")
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(args.csv)

    main(args.csv)
