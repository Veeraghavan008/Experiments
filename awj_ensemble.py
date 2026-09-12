from __future__ import annotations
import sys, logging, warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib;

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor,
    GradientBoostingRegressor, AdaBoostRegressor,
    VotingRegressor, HistGradientBoostingRegressor,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")

DATA_CSV = Path("data/awj/awj_data.csv")
DATA_XLSX = Path("data/awj/awj_data.xlsx")
RESULTS_DIR = Path("data/awj/results")
FEATURE_COLS = ["Pressure", "TR", "AFR", "SOD"]
TARGET_COL = "DOC"
RANDOM_STATE = 42
TEST_SIZE = 0.30
ENABLE_TUNING = True
N_ITER = 100
CV_FOLDS = 5
FIGURE_DPI = 150

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    name: str
    pipeline: Pipeline
    param_grid: dict[str, list[Any]] = field(default_factory=dict)
    supports_feature_importance: bool = True


def _xgb_or_hist() -> Any:
    try:
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4,
                            subsample=0.8, colsample_bytree=0.8,
                            random_state=RANDOM_STATE, verbosity=0)
    except ImportError:
        log.warning("xgboost not found — falling back to HistGradientBoostingRegressor.")
        return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05,
                                             max_depth=4, random_state=RANDOM_STATE)


def build_model_configs() -> list[ModelConfig]:
    tree_grid = {
        "model__n_estimators": [50, 100, 150, 200, 250, 300, 400, 500, 700, 1000],
        "model__max_depth": [None, 3, 5, 7, 10, 12, 15, 18, 22, 30],
        "model__max_features": ["sqrt", "log2", 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
        "model__min_samples_leaf": [1, 2, 3, 4, 5, 6, 7, 8, 10, 12],
        "model__min_samples_split": [2, 3, 4, 5, 6, 7, 8, 10, 12, 15],
    }

    rf = ModelConfig(
        name="Random Forest",
        pipeline=Pipeline([("scaler", StandardScaler()),
                           ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1))]),
        param_grid=tree_grid,
    )

    et = ModelConfig(
        name="Extra Trees",
        pipeline=Pipeline([("scaler", StandardScaler()),
                           ("model", ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1))]),
        param_grid=tree_grid,
    )

    gb = ModelConfig(
        name="Gradient Boosting",
        pipeline=Pipeline([("scaler", StandardScaler()),
                           ("model", GradientBoostingRegressor(random_state=RANDOM_STATE))]),
        param_grid={
            "model__n_estimators": [50, 100, 150, 200, 250, 300, 400, 500, 700, 1000],
            "model__learning_rate": [0.001, 0.005, 0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3],
            "model__max_depth": [2, 3, 4, 5, 6, 7, 8, 9, 10, 12],
            "model__subsample": [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
            "model__min_samples_leaf": [1, 2, 3, 4, 5, 6, 7, 8, 10, 12],
            "model__min_samples_split": [2, 3, 4, 5, 6, 7, 8, 10, 12, 15],
        },
    )

    ada = ModelConfig(
        name="AdaBoost",
        pipeline=Pipeline([("scaler", StandardScaler()),
                           ("model", AdaBoostRegressor(estimator=DecisionTreeRegressor(),
                                                       random_state=RANDOM_STATE))]),
        param_grid={
            "model__n_estimators": [25, 50, 75, 100, 150, 200, 300, 400, 500, 700],
            "model__learning_rate": [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5],
            "model__loss": ["linear", "square", "exponential",
                            "linear", "square", "exponential",
                            "linear", "square", "exponential", "linear"],
            "model__estimator__max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 10, 12],
        },
        supports_feature_importance=False,
    )

    voting = ModelConfig(
        name="Voting Regressor",
        pipeline=Pipeline([("scaler", StandardScaler()),
                           ("model", VotingRegressor(estimators=[
                               ("rf", RandomForestRegressor(n_estimators=200,
                                                            random_state=RANDOM_STATE, n_jobs=-1)),
                               ("et", ExtraTreesRegressor(n_estimators=200,
                                                          random_state=RANDOM_STATE, n_jobs=-1)),
                               ("gb", _xgb_or_hist()),
                           ], n_jobs=-1))]),
        param_grid={
            "model__weights": [
                [1, 1, 1], [2, 1, 1], [1, 2, 1], [1, 1, 2],
                [3, 1, 1], [1, 3, 1], [1, 1, 3],
                [2, 2, 1], [2, 1, 2], [1, 2, 2],
            ],
        },
        supports_feature_importance=False,
    )

    return [rf, et, gb, ada, voting]


def load_data() -> pd.DataFrame:
    if DATA_CSV.exists():
        df = pd.read_csv(DATA_CSV)
    elif DATA_XLSX.exists():
        df = pd.read_excel(DATA_XLSX)
    else:
        raise FileNotFoundError(f"No data file at {DATA_CSV} or {DATA_XLSX}")

    df.columns = df.columns.str.strip()
    required = FEATURE_COLS + [TARGET_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Found: {df.columns.tolist()}")

    df = df[required].apply(pd.to_numeric, errors="coerce")
    before = len(df)
    df.dropna(inplace=True)
    if len(df) < before:
        log.warning("Dropped %d rows with NaN.", before - len(df))

    log.info("Loaded %d rows | features=%s | target=%s", len(df), FEATURE_COLS, TARGET_COL)
    return df.reset_index(drop=True)


def compute_metrics(pipe, X_train, y_train, X_test, y_test) -> dict:
    return {
        "Train MAE": round(float(mean_absolute_error(y_train, pipe.predict(X_train))), 4),
        "Train R2": round(float(r2_score(y_train, pipe.predict(X_train))), 4),
        "Test MAE": round(float(mean_absolute_error(y_test, pipe.predict(X_test))), 4),
        "Test R2": round(float(r2_score(y_test, pipe.predict(X_test))), 4),
    }


def print_param_grids(configs: list[ModelConfig]) -> None:
    print("\nHyperparameter Search Spaces  (10 values per parameter)")
    print("-" * 60)
    for cfg in configs:
        if not cfg.param_grid:
            continue
        print(f"\n  {cfg.name}")
        for k, v in cfg.param_grid.items():
            label = k.replace("model__", "")
            display = list(dict.fromkeys(v)) if isinstance(v[0], str) else v
            print(f"    {label:<35} {display}")
    print()


def train_and_evaluate(configs, X_train, y_train, X_test, y_test):
    records, fitted, best_params = [], [], {}
    n_cv = min(CV_FOLDS, len(y_train))

    print("Training & Tuning")
    print("-" * 60)
    for cfg in configs:
        print(f"\n  [{cfg.name}]")
        if ENABLE_TUNING and cfg.param_grid:
            search = RandomizedSearchCV(
                cfg.pipeline, cfg.param_grid,
                n_iter=N_ITER, scoring="r2", cv=n_cv,
                n_jobs=-1, refit=True, random_state=RANDOM_STATE, verbose=0,
            )
            search.fit(X_train, y_train)
            pipe, params = search.best_estimator_, search.best_params_
            print(f"  CV R2        : {search.best_score_:.4f}")
            print(f"  Best params  : {params}")
        else:
            pipe = cfg.pipeline
            pipe.fit(X_train, y_train)
            params = {}

        m = compute_metrics(pipe, X_train, y_train, X_test, y_test)
        m["Model"] = cfg.name
        records.append(m)
        fitted.append((cfg.name, pipe))
        best_params[cfg.name] = params
        print(f"  Train R2={m['Train R2']:.4f}  Test R2={m['Test R2']:.4f} "
              f"| Train MAE={m['Train MAE']:.4f}  Test MAE={m['Test MAE']:.4f}")

    print()
    cols = ["Model", "Train MAE", "Train R2", "Test MAE", "Test R2"]
    df = (pd.DataFrame(records)[cols]
          .sort_values("Test R2", ascending=False)
          .reset_index(drop=True))
    df.index += 1
    df.index.name = "Rank"
    return df, fitted, best_params


def print_results(results_df, best_params) -> None:
    w = 78
    print("=" * w)
    print(f"  Results  |  Split {int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)}"
          f"  |  Tuning={'ON  n_iter=' + str(N_ITER) if ENABLE_TUNING else 'OFF'}")
    print("=" * w)
    print(f"  {'Rank':<5} {'Model':<22} {'Train MAE':>10} {'Train R2':>10} "
          f"{'Test MAE':>10} {'Test R2':>10}")
    print("  " + "-" * 74)
    for rank, row in results_df.iterrows():
        tag = "  <-- BEST" if rank == 1 else ""
        print(f"  {rank:<5} {row['Model']:<22} {row['Train MAE']:>10.4f} "
              f"{row['Train R2']:>10.4f} {row['Test MAE']:>10.4f} {row['Test R2']:>10.4f}{tag}")
    print("=" * w)

    best = results_df.iloc[0]
    name = best["Model"]
    params = best_params.get(name, {})
    print(f"\n  Winner : {name}")
    print(f"  Test R2={best['Test R2']:.4f}  Test MAE={best['Test MAE']:.4f} mm"
          f"  |  Train R2={best['Train R2']:.4f}  Train MAE={best['Train MAE']:.4f} mm")
    if params:
        print("\n  Fixed Hyperparameters")
        print("  " + "-" * 40)
        for k, v in params.items():
            print(f"  {k.replace('model__', ''):<35} = {v}")
    print()


def save_outputs(results_df, best_params) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_DIR / "paper_comparative_metrics.csv")
    log.info("Saved: paper_comparative_metrics.csv")
    if ENABLE_TUNING:
        rows = [{"Model": m, "Parameter": k.replace("model__", ""), "Value": v}
                for m, p in best_params.items() for k, v in p.items()]
        pd.DataFrame(rows).to_csv(RESULTS_DIR / "best_hyperparameters.csv", index=False)
        log.info("Saved: best_hyperparameters.csv")


COLORS = {"train": "#4C72B0", "test": "#DD8452", "imp": "#55A868"}


def plot_r2_comparison(results_df, path) -> None:
    models, train_r2, test_r2 = (results_df["Model"].tolist(),
                                 results_df["Train R2"].tolist(),
                                 results_df["Test R2"].tolist())
    x, w = np.arange(len(models)), 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    b1 = ax.bar(x - w / 2, train_r2, w, label="Train R²", color=COLORS["train"], edgecolor="white")
    b2 = ax.bar(x + w / 2, test_r2, w, label="Test R²", color=COLORS["test"], edgecolor="white")
    for bar in (*b1, *b2):
        h = bar.get_height()
        ax.text(bar.get_x() + w / 2, h + 0.005, f"{h:.3f}",
                ha="center", va="bottom", fontsize=8, color="dimgrey")
    ax.set_xticks(x);
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("R² Score", fontsize=12)
    ax.set_title("AWJ DOC — Train vs Test R²", fontsize=13, fontweight="bold")
    ax.set_ylim(0, min(1.08, max(*train_r2, *test_r2) + 0.12))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.legend(fontsize=11);
    ax.grid(axis="y", linestyle="--", alpha=0.4);
    ax.set_axisbelow(True)
    fig.tight_layout();
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight");
    plt.close(fig)
    log.info("Saved: %s", path)


def plot_feature_importance(model_name, pipe, path) -> None:
    est = pipe.named_steps["model"]
    if not hasattr(est, "feature_importances_"):
        return
    imp, order = est.feature_importances_, np.argsort(est.feature_importances_)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([FEATURE_COLS[i] for i in order], imp[order], color=COLORS["imp"], edgecolor="white")
    for i, v in enumerate(imp[order]):
        ax.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=9, color="dimgrey")
    ax.set_xlabel("Feature Importance (MDI)", fontsize=12)
    ax.set_title(f"Feature Importance — {model_name}", fontsize=13, fontweight="bold")
    ax.set_xlim(0, max(imp) + 0.12);
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout();
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight");
    plt.close(fig)
    log.info("Saved: %s", path)


def plot_fitting_curves(fitted_models, X_train, y_train, X_test, y_test, path) -> None:
    n = len(fitted_models)
    ncols = min(3, n);
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)

    for idx, (name, pipe) in enumerate(fitted_models):
        ax = axes[idx // ncols][idx % ncols]
        ytr_p, yte_p = pipe.predict(X_train), pipe.predict(X_test)
        ax.scatter(y_train, ytr_p, color=COLORS["train"], alpha=0.75, s=45,
                   label="Train", edgecolors="white", linewidths=0.4, zorder=3)
        ax.scatter(y_test, yte_p, color=COLORS["test"], alpha=0.85, s=55,
                   label="Test", edgecolors="white", linewidths=0.4, zorder=3)

        all_a = np.concatenate([y_train, y_test]);
        all_p = np.concatenate([ytr_p, yte_p])
        lo, hi = min(all_a.min(), all_p.min()), max(all_a.max(), all_p.max())
        mg = (hi - lo) * 0.05
        ax.plot([lo - mg, hi + mg], [lo - mg, hi + mg], "grey", linestyle="--",
                linewidth=1.2, alpha=0.7, label="Perfect fit", zorder=2)
        xs = all_a[np.argsort(all_a)];
        ys = all_p[np.argsort(all_a)]
        ax.plot(np.linspace(xs[0], xs[-1], 200),
                np.poly1d(np.polyfit(xs, ys, 2))(np.linspace(xs[0], xs[-1], 200)),
                color="#C44E52", linewidth=2.0, label="Poly fit (deg 2)", zorder=4)

        ax.text(0.04, 0.96,
                f"Train R²={r2_score(y_train, ytr_p):.3f}\n"
                f"Test  R²={r2_score(y_test, yte_p):.3f}\n"
                f"Test MAE={mean_absolute_error(y_test, yte_p):.3f}",
                transform=ax.transAxes, va="top", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#ccc", alpha=0.85))
        ax.set_xlim(lo - mg, hi + mg);
        ax.set_ylim(lo - mg, hi + mg)
        ax.set_xlabel("Actual DOC (mm)", fontsize=10);
        ax.set_ylabel("Predicted DOC (mm)", fontsize=10)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="lower right");
        ax.grid(linestyle="--", alpha=0.35)

    for i in range(n, nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)

    fig.suptitle("AWJ DOC — Actual vs Predicted", fontsize=14, fontweight="bold")
    fig.tight_layout();
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight");
    plt.close(fig)
    log.info("Saved: %s", path)


def main():
    log.info("AWJ Ensemble  |  ENABLE_TUNING=%s  |  N_ITER=%s\n",
             ENABLE_TUNING, N_ITER if ENABLE_TUNING else "N/A")

    df = load_data()
    X = df[FEATURE_COLS].values.astype(float)
    y = df[TARGET_COL].values.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    log.info("Split: %d train / %d test\n", len(y_train), len(y_test))

    configs = build_model_configs()
    print_param_grids(configs)

    results_df, fitted_models, best_params = train_and_evaluate(
        configs, X_train, y_train, X_test, y_test
    )

    print_results(results_df, best_params)
    save_outputs(results_df, best_params)

    print("Saving plots ...")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_r2_comparison(results_df, RESULTS_DIR / "r2_comparison.png")

    tree_model_map = dict(fitted_models)
    best_tree_name = next(
        (name for name in results_df["Model"]
         if hasattr(tree_model_map[name].named_steps["model"], "feature_importances_")),
        None,
    )
    if best_tree_name:
        plot_feature_importance(best_tree_name, tree_model_map[best_tree_name],
                                RESULTS_DIR / "feature_importance.png")

    plot_fitting_curves(fitted_models, X_train, y_train, X_test, y_test,
                        RESULTS_DIR / "fitting_curves.png")
    print()
    log.info("Done. All outputs -> %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
