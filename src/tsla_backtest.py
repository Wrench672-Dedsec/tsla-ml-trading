import argparse
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False

from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV, lasso_path
from sklearn.metrics import accuracy_score, roc_auc_score
from glob import glob
from collections import defaultdict
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    import optuna

    HAS_OPTUNA = True
except Exception:
    HAS_OPTUNA = False


LOGGER = logging.getLogger("tsla_backtest")


def _setup_logging(out_dir: Path, level: str = "INFO", log_file: str = "backtest.log") -> Path:
    level_name = str(level).upper()
    log_level = getattr(logging, level_name, logging.INFO)
    LOGGER.setLevel(log_level)
    LOGGER.handlers.clear()
    LOGGER.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    LOGGER.addHandler(console_handler)

    file_path = _safe_path(out_dir / log_file)
    file_handler = logging.FileHandler(file_path, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)

    LOGGER.info("Logging enabled. level=%s, file=%s", level_name, file_path)
    return file_path


@dataclass
class BacktestResult:
    name: str
    n_train: int
    n_val: int
    n_test: int
    n_features: int
    selected_features: List[str]
    val_cum_return: float
    test_cum_return: float
    test_sharpe: float
    test_max_dd: float
    test_accuracy: float
    test_auc: float


class TimeSeriesStackingClassifier(BaseEstimator):
    """Time-series-safe stacking classifier.

    This avoids sklearn StackingClassifier's internal cross_val_predict partition
    requirement, and instead builds OOF meta-features with TimeSeriesSplit.
    """

    def __init__(
        self,
        estimators: List[Tuple[str, BaseEstimator]],
        final_estimator: BaseEstimator,
        passthrough: bool = False,
        n_splits: int = 3,
    ) -> None:
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.passthrough = passthrough
        self.n_splits = n_splits

    @staticmethod
    def _predict_score(model: BaseEstimator, X: np.ndarray) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        pred = model.predict(X)
        return np.asarray(pred, dtype=float)

    def fit(self, X, y):
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)
        n = X_arr.shape[0]
        if n < 4:
            raise ValueError("Not enough samples for TimeSeriesStackingClassifier")

        n_splits = min(int(self.n_splits), max(2, n - 1))
        tscv = TimeSeriesSplit(n_splits=n_splits)

        m = len(self.estimators)
        meta_oof = np.full((n, m), np.nan, dtype=float)

        # Build OOF meta-features using forward-only splits.
        for tr_idx, va_idx in tscv.split(np.arange(n)):
            X_tr, y_tr = X_arr[tr_idx], y_arr[tr_idx]
            X_va = X_arr[va_idx]
            for j, (_, est) in enumerate(self.estimators):
                em = clone(est)
                em.fit(X_tr, y_tr)
                meta_oof[va_idx, j] = self._predict_score(em, X_va)

        valid = ~np.isnan(meta_oof).any(axis=1)
        if int(np.sum(valid)) < max(20, m * 5):
            raise ValueError("Not enough valid OOF rows for stacking meta-model")

        if self.passthrough:
            X_meta = np.hstack([meta_oof[valid], X_arr[valid]])
        else:
            X_meta = meta_oof[valid]

        self.final_estimator_ = clone(self.final_estimator)
        self.final_estimator_.fit(X_meta, y_arr[valid])

        # Refit base estimators on the full dataset for inference.
        self.base_estimators_: List[Tuple[str, BaseEstimator]] = []
        for name, est in self.estimators:
            em = clone(est)
            em.fit(X_arr, y_arr)
            self.base_estimators_.append((name, em))

        self.n_features_in_ = X_arr.shape[1]
        return self

    def _meta_features(self, X) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        meta = np.column_stack([self._predict_score(est, X_arr) for _, est in self.base_estimators_])
        if self.passthrough:
            return np.hstack([meta, X_arr])
        return meta

    def predict_proba(self, X) -> np.ndarray:
        X_meta = self._meta_features(X)
        if hasattr(self.final_estimator_, "predict_proba"):
            return self.final_estimator_.predict_proba(X_meta)

        score = self.final_estimator_.predict(X_meta).astype(float)
        score = np.clip(score, 0.0, 1.0)
        return np.column_stack([1.0 - score, score])

    def predict(self, X):
        p = self.predict_proba(X)[:, 1]
        return (p >= 0.5).astype(int)


def _safe_path(path: Path) -> Path:
    """Return a writable path; if locked, fall back to a timestamped sibling."""
    try:
        # Try opening for exclusive-ish write.
        with open(path, "a", encoding="utf-8"):
            pass
        return path
    except PermissionError:
        ts = time.strftime("%Y%m%d_%H%M%S")
        return path.with_name(f"{path.stem}_{ts}{path.suffix}")


def _filter_ai_features_for_backtest(
    X_train: pd.DataFrame,
    ai_feature_cols: List[str],
    missing_threshold: float,
    corr_threshold: float,
) -> Tuple[List[str], pd.DataFrame]:
    """Drop sparse and redundant AI features using train-only statistics.

    The filter is intentionally conservative:
    - remove AI columns with train missing rate above `missing_threshold`
    - from the remaining AI columns, greedily remove columns that are highly
      correlated with already-kept AI columns above `corr_threshold`
    """

    if not ai_feature_cols:
        return [], pd.DataFrame(columns=["feature", "train_missing_rate", "train_std", "reason"])

    missing_rate = X_train[ai_feature_cols].isna().mean().sort_values(ascending=False)
    train_std = X_train[ai_feature_cols].std(numeric_only=True, ddof=0).fillna(0.0)

    kept: List[str] = []
    rows: List[Dict[str, object]] = []
    dropped_due_to_missing = 0
    dropped_due_to_corr = 0

    candidate_cols = [c for c in ai_feature_cols if float(missing_rate.get(c, 1.0)) <= float(missing_threshold)]
    candidate_cols.sort(key=lambda c: (float(missing_rate.get(c, 1.0)), -float(train_std.get(c, 0.0)), str(c)))

    X_num = X_train[candidate_cols].apply(pd.to_numeric, errors="coerce")

    for col in ai_feature_cols:
        mr = float(missing_rate.get(col, 1.0))
        sd = float(train_std.get(col, 0.0))
        if mr > float(missing_threshold):
            rows.append({"feature": col, "train_missing_rate": mr, "train_std": sd, "reason": "high_missing"})
            dropped_due_to_missing += 1
            continue

        is_redundant = False
        for kept_col in kept:
            s1 = pd.to_numeric(X_num[col], errors="coerce")
            s2 = pd.to_numeric(X_num[kept_col], errors="coerce")
            valid = s1.notna() & s2.notna()
            if int(valid.sum()) < 15:
                continue
            corr = float(s1[valid].corr(s2[valid]))
            if np.isfinite(corr) and abs(corr) >= float(corr_threshold):
                rows.append(
                    {
                        "feature": col,
                        "train_missing_rate": mr,
                        "train_std": sd,
                        "reason": f"high_corr_with:{kept_col}",
                        "corr_with_kept": kept_col,
                        "abs_corr": abs(corr),
                    }
                )
                dropped_due_to_corr += 1
                is_redundant = True
                break

        if not is_redundant:
            kept.append(col)
            rows.append({"feature": col, "train_missing_rate": mr, "train_std": sd, "reason": "kept"})

    report = pd.DataFrame(rows)
    report["dropped_due_to_missing"] = dropped_due_to_missing
    report["dropped_due_to_corr"] = dropped_due_to_corr
    return kept, report


def _model_label(model_name: str) -> str:
    name = str(model_name)
    if name == "baseline_long":
        return "baseline"
    if "_best_" in name:
        return name.split("_best_", 1)[0]
    return name


def _save_visualizations(
    summary: pd.DataFrame,
    out_dir: Path,
    model_trades_test: Dict[str, pd.DataFrame],
) -> Dict[str, Path]:
    """Save CV/test comparison plots and return created artifact paths."""

    artifacts: Dict[str, Path] = {}
    if not HAS_MATPLOTLIB:
        LOGGER.warning("matplotlib unavailable; skip visualization outputs.")
        return artifacts

    df = summary.copy()
    df["model_label"] = df["model"].map(_model_label)

    cv_cols = ["model_label", "val_opt_method", "val_ann", "val_cum"]
    cv_df = df[cv_cols].copy()
    cv_df = cv_df.sort_values("val_ann", ascending=False).reset_index(drop=True)
    cv_csv_path = _safe_path(out_dir / "cv_summary.csv")
    cv_df.to_csv(cv_csv_path, index=False, encoding="utf-8-sig")
    artifacts["cv_summary"] = cv_csv_path

    order = cv_df["model_label"].tolist()
    idx = np.arange(len(order))

    fig_cv, axes_cv = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes_cv[0].bar(idx, cv_df["val_ann"].astype(float) * 100.0, color="#1f77b4")
    axes_cv[0].set_ylabel("Val annualized return (%)")
    axes_cv[0].set_title("Walk-forward CV performance by model")
    axes_cv[0].axhline(0.0, color="black", linewidth=0.8)

    axes_cv[1].bar(idx, cv_df["val_cum"].astype(float) * 100.0, color="#2ca02c")
    axes_cv[1].set_ylabel("Val cumulative return (%)")
    axes_cv[1].set_xticks(idx)
    axes_cv[1].set_xticklabels(order, rotation=25, ha="right")
    axes_cv[1].axhline(0.0, color="black", linewidth=0.8)

    fig_cv.tight_layout()
    cv_plot_path = _safe_path(out_dir / "cv_model_comparison.png")
    fig_cv.savefig(cv_plot_path, dpi=140)
    plt.close(fig_cv)
    artifacts["cv_plot"] = cv_plot_path

    df_test = df.set_index("model_label").reindex(order).reset_index()
    fig_test, axes_test = plt.subplots(1, 3, figsize=(16, 4.8))
    axes_test[0].bar(np.arange(len(order)), df_test["test_cum"].astype(float) * 100.0, color="#ff7f0e")
    axes_test[0].set_title("Test cumulative return")
    axes_test[0].set_ylabel("Return (%)")
    axes_test[0].axhline(0.0, color="black", linewidth=0.8)

    axes_test[1].bar(np.arange(len(order)), df_test["test_sharpe"].astype(float), color="#17becf")
    axes_test[1].set_title("Test Sharpe")
    axes_test[1].axhline(0.0, color="black", linewidth=0.8)

    axes_test[2].bar(np.arange(len(order)), df_test["test_max_dd"].astype(float) * 100.0, color="#d62728")
    axes_test[2].set_title("Test max drawdown")
    axes_test[2].set_ylabel("Drawdown (%)")
    axes_test[2].axhline(0.0, color="black", linewidth=0.8)

    for ax in axes_test:
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels(order, rotation=25, ha="right")

    fig_test.tight_layout()
    test_plot_path = _safe_path(out_dir / "test_model_comparison.png")
    fig_test.savefig(test_plot_path, dpi=140)
    plt.close(fig_test)
    artifacts["test_plot"] = test_plot_path

    fig_eq, ax_eq = plt.subplots(figsize=(12, 5))
    plotted = 0
    for name, trades in model_trades_test.items():
        if trades is None or trades.empty or "strategy_return" not in trades.columns:
            continue
        r = pd.to_numeric(trades["strategy_return"], errors="coerce").fillna(0.0)
        eq = (1.0 + r).cumprod()
        x = pd.to_datetime(trades.get("date"), errors="coerce")
        ax_eq.plot(x, eq, linewidth=1.4, label=name)
        plotted += 1

    if plotted > 0:
        ax_eq.set_title("Test equity curves")
        ax_eq.set_ylabel("Equity (start=1)")
        ax_eq.grid(alpha=0.25)
        ax_eq.legend(loc="best", ncol=2)
        fig_eq.tight_layout()
        eq_plot_path = _safe_path(out_dir / "test_equity_curves.png")
        fig_eq.savefig(eq_plot_path, dpi=140)
        artifacts["equity_plot"] = eq_plot_path

    plt.close(fig_eq)
    return artifacts


def _save_feature_selector_comparison(
    selected_lasso: List[str],
    selected_ridge: List[str],
    out_dir: Path,
    *,
    top_k: int = 40,
) -> Dict[str, Path]:
    """Save LASSO vs Ridge feature-selection comparison tables and plots."""

    artifacts: Dict[str, Path] = {}

    # Keep insertion order while dropping duplicates.
    lasso_order = list(dict.fromkeys(selected_lasso))
    ridge_order = list(dict.fromkeys(selected_ridge))

    lasso_set = set(lasso_order)
    ridge_set = set(ridge_order)
    overlap = lasso_set.intersection(ridge_set)
    only_lasso = lasso_set - ridge_set
    only_ridge = ridge_set - lasso_set

    lasso_rank = {f: i + 1 for i, f in enumerate(lasso_order)}
    ridge_rank = {f: i + 1 for i, f in enumerate(ridge_order)}

    all_features = sorted(lasso_set.union(ridge_set))
    detail_rows = []
    for feat in all_features:
        lr = lasso_rank.get(feat)
        rr = ridge_rank.get(feat)
        rank_score = min(x for x in [lr, rr] if x is not None)
        detail_rows.append(
            {
                "feature": feat,
                "in_lasso": int(feat in lasso_set),
                "in_ridge": int(feat in ridge_set),
                "lasso_rank": lr,
                "ridge_rank": rr,
                "best_rank": rank_score,
            }
        )

    detail_df = pd.DataFrame(detail_rows).sort_values(["best_rank", "feature"], ascending=[True, True]).reset_index(drop=True)
    detail_path = _safe_path(out_dir / "feature_selector_detail.csv")
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    artifacts["feature_selector_detail"] = detail_path

    overlap_ratio = (len(overlap) / max(1, len(lasso_set)))
    jaccard = len(overlap) / max(1, len(lasso_set.union(ridge_set)))
    summary_df = pd.DataFrame(
        [
            {
                "lasso_n": len(lasso_set),
                "ridge_n": len(ridge_set),
                "overlap_n": len(overlap),
                "lasso_only_n": len(only_lasso),
                "ridge_only_n": len(only_ridge),
                "overlap_vs_lasso": overlap_ratio,
                "jaccard": jaccard,
            }
        ]
    )
    summary_path = _safe_path(out_dir / "feature_selector_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    artifacts["feature_selector_summary"] = summary_path

    if not HAS_MATPLOTLIB:
        LOGGER.warning("matplotlib unavailable; skip feature-selector comparison plots.")
        return artifacts

    plot_df = detail_df.head(max(1, int(top_k))).copy()
    matrix = plot_df[["in_lasso", "in_ridge"]].to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={"width_ratios": [1.0, 1.9]})

    count_labels = ["LASSO only", "Overlap", "Ridge only"]
    count_vals = [len(only_lasso), len(overlap), len(only_ridge)]
    count_colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]
    bars = axes[0].bar(np.arange(3), count_vals, color=count_colors)
    axes[0].set_xticks(np.arange(3))
    axes[0].set_xticklabels(count_labels, rotation=20, ha="right")
    axes[0].set_ylabel("Feature count")
    axes[0].set_title("LASSO vs Ridge overlap")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    for b in bars:
        axes[0].text(
            b.get_x() + b.get_width() / 2.0,
            b.get_height() + 0.2,
            f"{int(b.get_height())}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    im = axes[1].imshow(matrix, aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["LASSO", "Ridge"])
    axes[1].set_yticks(np.arange(len(plot_df)))
    axes[1].set_yticklabels(plot_df["feature"].tolist(), fontsize=8)
    axes[1].set_title(f"Top-{len(plot_df)} feature membership (by best rank)")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axes[1].text(j, i, str(int(matrix[i, j])), ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(im, ax=axes[1], fraction=0.045, pad=0.03, label="Selected (1=yes)")

    fig.tight_layout()
    plot_path = _safe_path(out_dir / "feature_selector_comparison.png")
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)
    artifacts["feature_selector_plot"] = plot_path

    return artifacts


def _max_drawdown(equity_curve: pd.Series) -> float:
    """Max drawdown on an equity curve (1.0 based)."""
    running_max = equity_curve.cummax()
    dd = equity_curve / running_max - 1.0
    return float(dd.min())


def _perf_stats(returns: pd.Series, periods_per_year: int = 252) -> Dict[str, float]:
    r = returns.dropna().astype(float)
    if r.empty:
        return {"cum": 0.0, "ann": 0.0, "vol": 0.0, "sharpe": 0.0, "max_dd": 0.0}

    equity = (1.0 + r).cumprod()
    cum = float(equity.iloc[-1] - 1.0)

    n = len(r)
    ann = float((1.0 + cum) ** (periods_per_year / max(1, n)) - 1.0)
    vol = float(r.std(ddof=0) * math.sqrt(periods_per_year))
    sharpe = float((r.mean() / (r.std(ddof=0) + 1e-12)) * math.sqrt(periods_per_year))
    max_dd = _max_drawdown(equity)
    return {"cum": cum, "ann": ann, "vol": vol, "sharpe": sharpe, "max_dd": max_dd}


def _apply_costs_and_turnover(
    raw_returns: np.ndarray,
    positions: np.ndarray,
    *,
    cost_per_unit: float,
    prev_position: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Apply linear costs based on position changes.

    cost_per_unit is in return units (e.g. 1bp = 0.0001).
    A move 0->1 costs 1*cost_per_unit, -1->1 costs 2*cost_per_unit.
    """

    pos = np.asarray(positions, dtype=float)
    r = np.asarray(raw_returns, dtype=float)
    if len(pos) != len(r):
        raise ValueError("positions and raw_returns must have same length")

    prev = np.concatenate([[float(prev_position)], pos[:-1]])
    turnover = np.abs(pos - prev)
    costs = turnover * float(cost_per_unit)
    net = r - costs
    return net, turnover, costs, float(pos[-1]) if len(pos) else float(prev_position)


def _positions_from_proba(
    proba_up: np.ndarray,
    *,
    positioning: str,
    threshold: Optional[float] = None,
    quantile: Optional[float] = None,
    quantile_threshold: Optional[float] = None,
) -> np.ndarray:
    p = np.asarray(proba_up, dtype=float)

    if positioning in {"long_short", "long_flat"}:
        if threshold is None:
            raise ValueError("threshold is required for positioning long_short/long_flat")
        if positioning == "long_short":
            return np.where(p >= float(threshold), 1.0, -1.0)
        return np.where(p >= float(threshold), 1.0, 0.0)

    if positioning == "long_flat_quantile":
        if quantile_threshold is not None:
            thr = float(quantile_threshold)
        else:
            if quantile is None:
                raise ValueError("quantile is required for positioning long_flat_quantile")
            q = float(quantile)
            if not (0.0 < q < 1.0):
                raise ValueError("quantile must be in (0, 1)")
            thr = float(np.quantile(p, q))
        return np.where(p >= thr, 1.0, 0.0)

    raise ValueError("positioning must be one of: long_short, long_flat, long_flat_quantile")


def _parse_float_grid(s: Optional[str]) -> Optional[List[float]]:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _sample_params_from_space_random(
    rng: np.random.Generator,
    space: Dict[str, Dict],
) -> Dict:
    params: Dict = {}
    for name, spec in space.items():
        kind = spec["type"]
        if kind == "int":
            low, high = int(spec["low"]), int(spec["high"])
            step = int(spec.get("step", 1))
            choices = list(range(low, high + 1, step))
            params[name] = int(rng.choice(choices))
        elif kind == "float":
            low, high = float(spec["low"]), float(spec["high"])
            log = bool(spec.get("log", False))
            if log:
                params[name] = float(np.exp(rng.uniform(np.log(low), np.log(high))))
            else:
                params[name] = float(rng.uniform(low, high))
        elif kind == "categorical":
            picked = rng.choice(spec["choices"])
            params[name] = picked.item() if hasattr(picked, "item") else picked
        else:
            raise ValueError(f"Unknown search space type: {kind}")
    return params


def _sample_params_from_space_optuna(
    trial,
    space: Dict[str, Dict],
) -> Dict:
    params: Dict = {}
    for name, spec in space.items():
        kind = spec["type"]
        if kind == "int":
            params[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]), step=int(spec.get("step", 1)))
        elif kind == "float":
            params[name] = trial.suggest_float(
                name,
                float(spec["low"]),
                float(spec["high"]),
                log=bool(spec.get("log", False)),
            )
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, list(spec["choices"]))
        else:
            raise ValueError(f"Unknown search space type: {kind}")
    return params


def _optimize_params_walk_forward(
    *,
    name: str,
    model_factory,
    search_space: Dict[str, Dict],
    n_trials: int,
    startup_trials: int,
    seed: int,
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    next_ret_dev: pd.Series,
    next_date_dev: pd.Series,
    cv_splits: int,
    positioning: str,
    threshold_mode: str,
    threshold_grid: List[float],
    quantile_grid: List[float],
    selection_metric: str,
    cost_per_unit: float,
) -> Tuple[Dict, Optional[float], Optional[float], Dict[str, float], str]:
    """Optimize model params with Bayesian TPE (Optuna) or random-search fallback."""

    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")

    if HAS_OPTUNA:
        sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=min(startup_trials, n_trials))
        study = optuna.create_study(direction="maximize", sampler=sampler)

        def _objective(trial) -> float:
            params = _sample_params_from_space_optuna(trial, search_space)
            _, thr, q, wf_stats = _walk_forward_select(
                name=f"{name}_trial",
                model_factory=model_factory,
                param_grid=[params],
                X_dev=X_dev,
                y_dev=y_dev,
                next_ret_dev=next_ret_dev,
                next_date_dev=next_date_dev,
                cv_splits=cv_splits,
                positioning=positioning,
                threshold_mode=threshold_mode,
                threshold_grid=threshold_grid,
                quantile_grid=quantile_grid,
                selection_metric=selection_metric,
                cost_per_unit=cost_per_unit,
                seed=seed,
                verbose=False,
            )
            trial.set_user_attr("best_threshold", thr)
            trial.set_user_attr("best_quantile", q)
            trial.set_user_attr("wf_stats", wf_stats)
            return float(wf_stats[selection_metric])

        study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)
        best_trial = study.best_trial
        best_params = dict(best_trial.params)
        best_thr = best_trial.user_attrs.get("best_threshold")
        best_q = best_trial.user_attrs.get("best_quantile")
        best_stats = dict(best_trial.user_attrs.get("wf_stats"))
        method = "bayes_optuna_tpe"
    else:
        rng = np.random.default_rng(seed)
        best_score = -1e18
        best_params = None
        best_thr = None
        best_q = None
        best_stats = None
        for _ in range(n_trials):
            params = _sample_params_from_space_random(rng, search_space)
            _, thr, q, wf_stats = _walk_forward_select(
                name=f"{name}_trial",
                model_factory=model_factory,
                param_grid=[params],
                X_dev=X_dev,
                y_dev=y_dev,
                next_ret_dev=next_ret_dev,
                next_date_dev=next_date_dev,
                cv_splits=cv_splits,
                positioning=positioning,
                threshold_mode=threshold_mode,
                threshold_grid=threshold_grid,
                quantile_grid=quantile_grid,
                selection_metric=selection_metric,
                cost_per_unit=cost_per_unit,
                seed=seed,
                verbose=False,
            )
            score = float(wf_stats[selection_metric])
            if score > best_score:
                best_score = score
                best_params = params
                best_thr = thr
                best_q = q
                best_stats = wf_stats

        assert best_params is not None and best_stats is not None
        method = "random_fallback"

    LOGGER.info(
        "%s optimized via %s: params=%s, thr=%s, q=%s, val_ann=%.2f%%, val_cum=%.2f%%, val_sharpe=%.2f",
        name,
        method,
        best_params,
        best_thr,
        best_q,
        best_stats["ann"] * 100.0,
        best_stats["cum"] * 100.0,
        best_stats["sharpe"],
    )
    return best_params, best_thr, best_q, best_stats, method


def _walk_forward_select(
    *,
    name: str,
    model_factory,
    param_grid: List[Dict],
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    next_ret_dev: pd.Series,
    next_date_dev: pd.Series,
    cv_splits: int,
    positioning: str,
    threshold_mode: str,
    threshold_grid: List[float],
    quantile_grid: List[float],
    selection_metric: str,
    cost_per_unit: float,
    seed: int,
    verbose: bool = True,
) -> Tuple[Dict, Optional[float], Optional[float], Dict[str, float]]:
    """Walk-forward selection over params and (threshold or quantile).

    Returns: best_params, best_threshold, best_quantile, walk_forward_stats
    where walk_forward_stats is computed on concatenated out-of-sample fold segments.
    """

    if cv_splits < 2:
        raise ValueError("walk-forward cv_splits must be >= 2")
    if selection_metric not in {"cum", "sharpe", "ann"}:
        raise ValueError("selection_metric must be one of: cum, sharpe, ann")
    if threshold_mode not in {"fixed", "auto"}:
        raise ValueError("threshold_mode must be one of: fixed, auto")

    n = len(X_dev)
    effective_splits = min(int(cv_splits), max(2, n - 1))
    cv = TimeSeriesSplit(n_splits=effective_splits)

    # For fixed threshold, don't search.
    if positioning in {"long_short", "long_flat"} and threshold_mode == "fixed":
        candidates: List[Tuple[Optional[float], Optional[float]]] = [(threshold_grid[0], None)]
    elif positioning in {"long_short", "long_flat"}:
        candidates = [(t, None) for t in threshold_grid]
    else:
        candidates = [(None, q) for q in quantile_grid]

    best_score = -1e18
    best_params: Optional[Dict] = None
    best_thr: Optional[float] = None
    best_q: Optional[float] = None
    best_stats: Optional[Dict[str, float]] = None

    for params in param_grid:
        # For each params, fit once per fold, then evaluate all candidates.
        oos_returns: Dict[Tuple[Optional[float], Optional[float]], List[float]] = {c: [] for c in candidates}
        oos_turnover: Dict[Tuple[Optional[float], Optional[float]], List[float]] = {c: [] for c in candidates}
        oos_costs: Dict[Tuple[Optional[float], Optional[float]], List[float]] = {c: [] for c in candidates}
        prev_pos: Dict[Tuple[Optional[float], Optional[float]], float] = {c: 0.0 for c in candidates}

        for fold_train, fold_val in cv.split(np.arange(n)):
            X_tr = X_dev.iloc[fold_train]
            y_tr = y_dev.iloc[fold_train]
            X_va = X_dev.iloc[fold_val]
            nr_va = next_ret_dev.iloc[fold_val]

            model = model_factory(params)
            pipe = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                    ("scaler", StandardScaler()),
                    ("clf", model),
                ]
            )
            pipe.fit(X_tr, y_tr)

            if hasattr(pipe, "predict_proba"):
                p_va = pipe.predict_proba(X_va)[:, 1]
                p_tr = pipe.predict_proba(X_tr)[:, 1]
            else:
                p_va = pipe.predict(X_va).astype(float)
                p_tr = pipe.predict(X_tr).astype(float)

            nr_va_arr = np.asarray(nr_va, dtype=float)

            for cand in candidates:
                thr, q = cand
                q_thr = None
                if positioning == "long_flat_quantile":
                    if q is None:
                        raise ValueError("quantile candidate cannot be None for long_flat_quantile")
                    q_thr = float(np.quantile(p_tr, float(q)))
                pos_va = _positions_from_proba(
                    p_va,
                    positioning=positioning,
                    threshold=thr,
                    quantile=q,
                    quantile_threshold=q_thr,
                )
                net, turnover, costs, prev_pos[cand] = _apply_costs_and_turnover(
                    raw_returns=nr_va_arr * pos_va,
                    positions=pos_va,
                    cost_per_unit=cost_per_unit,
                    prev_position=prev_pos[cand],
                )
                oos_returns[cand].extend(net.tolist())
                oos_turnover[cand].extend(turnover.tolist())
                oos_costs[cand].extend(costs.tolist())

        for thr, q in candidates:
            r = oos_returns[(thr, q)]
            to = oos_turnover[(thr, q)]
            co = oos_costs[(thr, q)]
            s = _perf_stats(pd.Series(r))
            s["turnover_mean"] = float(np.mean(to)) if to else 0.0
            s["turnover_total"] = float(np.sum(to)) if to else 0.0
            s["cost_total"] = float(np.sum(co)) if co else 0.0

            score = float(s[selection_metric])
            if score > best_score:
                best_score = score
                best_params = params
                best_thr = thr
                best_q = q
                best_stats = s

    assert best_params is not None and best_stats is not None
    if verbose:
        LOGGER.info(
            "%s walk-forward best: params=%s, thr=%s, q=%s, oos_ann=%.2f%%, oos_cum=%.2f%%, oos_sharpe=%.2f, oos_turnover_mean=%.3g, cost_total=%.4g",
            name,
            best_params,
            best_thr,
            best_q,
            best_stats["ann"] * 100.0,
            best_stats["cum"] * 100.0,
            best_stats["sharpe"],
            best_stats["turnover_mean"],
            best_stats["cost_total"],
        )
    return best_params, best_thr, best_q, best_stats


def _time_split_index(n: int, train_frac: float, val_frac: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    idx = np.arange(n)
    return idx[:train_end], idx[train_end:val_end], idx[val_end:]


def select_features_lasso(
    X_train: pd.DataFrame,
    y_train_reg: pd.Series,
    X_val: pd.DataFrame,
    y_val_reg: pd.Series,
    *,
    alpha_grid: Optional[List[float]] = None,
    n_alphas: int = 60,
    cv_splits: int = 5,
    max_features: int = 40,
    min_selected_features: int = 0,
    random_state: int = 42,
    lasso_plot_path: Optional[Path] = None,
) -> List[str]:
    """Select features via LASSO regression (non-zero coefficients).

    Alpha is chosen by time-series cross-validation on the training set.

    Notes:
    - "Penalty steps" corresponds to the number of alphas along the regularization path
      (controlled by ``n_alphas``) when ``alpha_grid`` is not provided.
    - We optionally prefer alphas that yield at least ``min_selected_features`` non-zero
      coefficients, using CV MSE as the selection criterion.
    """

    if cv_splits < 2:
        raise ValueError("cv_splits must be >= 2")
    if n_alphas < 2:
        raise ValueError("n_alphas must be >= 2")

    lasso_max_iter = 20000
    lasso_tol = 1e-3

    num_cols = list(X_train.columns)

    preprocess = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("var", VarianceThreshold(threshold=0.0)),
        ]
    )

    # Fit preprocess once on train.
    Xt_train = preprocess.fit_transform(X_train)
    Xt_val = preprocess.transform(X_val)

    y_tr = y_train_reg.to_numpy(dtype=float)
    y_va = y_val_reg.to_numpy(dtype=float)

    n_train = Xt_train.shape[0]
    effective_splits = min(cv_splits, max(2, n_train - 1))
    if effective_splits < 2:
        raise ValueError(f"Not enough training samples for CV: n_train={n_train}")

    cv = TimeSeriesSplit(n_splits=effective_splits)

    # Use LassoCV to obtain CV MSE across a regularization path.
    if alpha_grid is not None:
        alphas = np.asarray(alpha_grid, dtype=float)
        lcv = LassoCV(alphas=alphas, cv=cv, max_iter=lasso_max_iter, tol=lasso_tol, n_jobs=-1)
    else:
        # sklearn>=1.7: passing an int to `alphas` replaces deprecated `n_alphas`.
        lcv = LassoCV(alphas=int(n_alphas), cv=cv, max_iter=lasso_max_iter, tol=lasso_tol, n_jobs=-1)

    lcv.fit(Xt_train, y_tr)
    alphas = np.asarray(lcv.alphas_, dtype=float)

    # lcv.mse_path_: shape (n_alphas, n_folds)
    mse_mean = np.mean(lcv.mse_path_, axis=1)

    coefs_path: Optional[np.ndarray] = None
    nnz_per_alpha: Optional[List[int]] = None
    if min_selected_features or lasso_plot_path is not None:
        # Count how many features each alpha selects on the full training set.
        # Use lasso_path to compute coefficients for all alphas efficiently.
        _, coefs_path, _ = lasso_path(Xt_train, y_tr, alphas=alphas, max_iter=lasso_max_iter, tol=lasso_tol)
        # coefs_path: (n_features, n_alphas)
        if min_selected_features:
            nnz_per_alpha = np.sum(np.abs(coefs_path) > 1e-10, axis=0).astype(int).tolist()

    best_idx = int(np.argmin(mse_mean))
    chosen_idx = best_idx

    if min_selected_features and nnz_per_alpha is not None:
        feasible = [i for i, nnz in enumerate(nnz_per_alpha) if nnz >= int(min_selected_features)]
        if feasible:
            feasible_mse = [(i, float(mse_mean[i])) for i in feasible]
            chosen_idx = min(feasible_mse, key=lambda t: t[1])[0]

    chosen_alpha = float(alphas[chosen_idx])
    final_model = Lasso(alpha=chosen_alpha, max_iter=lasso_max_iter, tol=lasso_tol)
    final_model.fit(Xt_train, y_tr)
    best_coef = final_model.coef_.copy()
    best_nnz = int(np.sum(np.abs(best_coef) > 1e-10))

    pred_val = final_model.predict(Xt_val)
    val_mse = float(np.mean((pred_val - y_va) ** 2))
    cv_mse = float(mse_mean[chosen_idx])

    # Map coefficients back to surviving feature names after VarianceThreshold.
    kept_mask = preprocess.named_steps["var"].get_support()
    kept_cols = [c for c, keep in zip(num_cols, kept_mask) if keep]
    coef = pd.Series(best_coef, index=kept_cols)

    non_zero = coef[coef.abs() > 1e-10].sort_values(key=lambda s: s.abs(), ascending=False)
    if non_zero.empty:
        # Fallback: pick top absolute coefficients even if small.
        non_zero = coef.sort_values(key=lambda s: s.abs(), ascending=False)

    selected = non_zero.index[:max_features].tolist()

    if lasso_plot_path is not None:
        if HAS_MATPLOTLIB and coefs_path is not None:
            fig, ax = plt.subplots(figsize=(10, 6))
            log_lambda = np.log(alphas)
            for j in range(coefs_path.shape[0]):
                ax.plot(log_lambda, coefs_path[j, :], linewidth=0.9, alpha=0.65)
            ax.axvline(np.log(chosen_alpha), color="black", linestyle="--", linewidth=1.2, label="chosen lambda")
            ax.set_xlabel("log(lambda)")
            ax.set_ylabel("coefficient")
            ax.set_title("LASSO coefficient path")
            ax.grid(alpha=0.25)
            ax.legend(loc="best")
            fig.tight_layout()
            safe_plot_path = _safe_path(Path(lasso_plot_path))
            fig.savefig(safe_plot_path, dpi=140)
            plt.close(fig)
            LOGGER.info("Saved LASSO coefficient path plot: %s", safe_plot_path)
        else:
            LOGGER.warning("Skip LASSO coefficient path plot (matplotlib unavailable).")

    if min_selected_features:
        LOGGER.info(
            "LASSO selected %d features (alpha=%s, cv_mse=%.6g, val_mse=%.6g, nnz=%d, min_required=%d).",
            len(selected),
            chosen_alpha,
            cv_mse,
            val_mse,
            best_nnz,
            min_selected_features,
        )
    else:
        LOGGER.info(
            "LASSO selected %d features (alpha=%s, cv_mse=%.6g, val_mse=%.6g).",
            len(selected),
            chosen_alpha,
            cv_mse,
            val_mse,
        )
    return selected


def select_features_ridge(
    X_train: pd.DataFrame,
    y_train_reg: pd.Series,
    X_val: pd.DataFrame,
    y_val_reg: pd.Series,
    *,
    alphas: Optional[List[float]] = None,
    cv_splits: int = 5,
    max_features: int = 60,
    min_selected_features: int = 20,
    min_coef_ratio: float = 0.10,
    coef_detail_path: Optional[Path] = None,
    random_state: int = 42,
) -> List[str]:
    """Select features via Ridge with strength threshold + count constraints.

    Ridge does not produce exact zeros, so we use two constraints:
    1) keep coefficients with |coef| >= min_coef_ratio * max(|coef|)
    2) enforce min/max selected feature counts
    """

    if cv_splits < 2:
        raise ValueError("cv_splits must be >= 2")
    if max_features < 1:
        raise ValueError("max_features must be >= 1")
    if min_selected_features < 0:
        raise ValueError("min_selected_features must be >= 0")
    if not (0.0 <= float(min_coef_ratio) < 1.0):
        raise ValueError("min_coef_ratio must be in [0, 1)")

    if alphas is None:
        alphas = np.logspace(-4, 4, 30).tolist()

    num_cols = list(X_train.columns)

    preprocess = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("var", VarianceThreshold(threshold=0.0)),
        ]
    )

    Xt_train = preprocess.fit_transform(X_train)
    Xt_val = preprocess.transform(X_val)

    y_tr = y_train_reg.to_numpy(dtype=float)
    y_va = y_val_reg.to_numpy(dtype=float)

    n_train = Xt_train.shape[0]
    effective_splits = min(cv_splits, max(2, n_train - 1))
    cv = TimeSeriesSplit(n_splits=effective_splits)

    rcv = RidgeCV(alphas=alphas, cv=cv, scoring="neg_mean_squared_error")
    rcv.fit(Xt_train, y_tr)

    best_alpha = rcv.alpha_
    final_model = Ridge(alpha=best_alpha)
    final_model.fit(Xt_train, y_tr)
    best_coef = final_model.coef_.copy()

    pred_val = final_model.predict(Xt_val)
    val_mse = float(np.mean((pred_val - y_va) ** 2))

    kept_mask = preprocess.named_steps["var"].get_support()
    kept_cols = [c for c, keep in zip(num_cols, kept_mask) if keep]
    coef = pd.Series(best_coef, index=kept_cols)

    ranked_abs = coef.abs().sort_values(ascending=False)
    n_ranked = len(ranked_abs)
    cap = min(int(max_features), n_ranked)
    target_min = min(int(min_selected_features), cap)

    max_abs = float(ranked_abs.iloc[0]) if n_ranked else 0.0
    abs_threshold = float(max_abs * float(min_coef_ratio)) if max_abs > 0.0 else 0.0

    if abs_threshold > 0.0:
        threshold_keep = ranked_abs[ranked_abs >= abs_threshold].index.tolist()
    else:
        threshold_keep = ranked_abs.index.tolist()

    selected = threshold_keep[:cap]

    # If threshold is too strict, back-fill by rank to satisfy minimum count.
    if len(selected) < target_min:
        for feat in ranked_abs.index.tolist():
            if feat not in selected:
                selected.append(feat)
            if len(selected) >= target_min:
                break

    selected = selected[:cap]

    if coef_detail_path is not None:
        selected_set = set(selected)
        detail_df = pd.DataFrame(
            {
                "feature": ranked_abs.index,
                "coef": coef.loc[ranked_abs.index].to_numpy(),
                "abs_coef": ranked_abs.to_numpy(),
                "rank": np.arange(1, n_ranked + 1),
                "selected_by_threshold": [int(f in threshold_keep) for f in ranked_abs.index],
                "selected_final": [int(f in selected_set) for f in ranked_abs.index],
            }
        )
        safe_detail_path = _safe_path(Path(coef_detail_path))
        detail_df.to_csv(safe_detail_path, index=False, encoding="utf-8-sig")
        LOGGER.info("Saved Ridge coefficient ranking: %s", safe_detail_path)

    LOGGER.info(
        "Ridge selected %d features (alpha=%s, val_mse=%.6g, abs_threshold=%.6g, min_coef_ratio=%s, min_required=%d, max_cap=%d).",
        len(selected),
        best_alpha,
        val_mse,
        abs_threshold,
        min_coef_ratio,
        target_min,
        cap,
    )
    return selected


def select_features_slasso(
    X_train: pd.DataFrame,
    y_train_reg: pd.Series,
    X_val: pd.DataFrame,
    y_val_reg: pd.Series,
    *,
    alpha_grid: Optional[List[float]] = None,
    n_alphas: int = 60,
    cv_splits: int = 5,
    max_features: int = 60,
    min_selected_features: int = 20,
    n_bootstrap: int = 80,
    sample_frac: float = 0.75,
    stability_threshold: float = 0.60,
    detail_path: Optional[Path] = None,
    random_state: int = 42,
) -> List[str]:
    """Select features via Stable LASSO (SLASSO).

    Steps:
    1) Select LASSO alpha by time-series CV on training set.
    2) Run bootstrap LASSO fits and compute feature selection frequency.
    3) Keep features with frequency >= stability_threshold, then apply min/max bounds.
    """

    if cv_splits < 2:
        raise ValueError("cv_splits must be >= 2")
    if n_alphas < 2:
        raise ValueError("n_alphas must be >= 2")
    if max_features < 1:
        raise ValueError("max_features must be >= 1")
    if min_selected_features < 0:
        raise ValueError("min_selected_features must be >= 0")
    if n_bootstrap < 5:
        raise ValueError("n_bootstrap must be >= 5")
    if not (0.0 < float(sample_frac) <= 1.0):
        raise ValueError("sample_frac must be in (0, 1]")
    if not (0.0 <= float(stability_threshold) <= 1.0):
        raise ValueError("stability_threshold must be in [0, 1]")

    lasso_max_iter = 20000
    lasso_tol = 1e-3

    num_cols = list(X_train.columns)
    preprocess = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("var", VarianceThreshold(threshold=0.0)),
        ]
    )

    Xt_train = preprocess.fit_transform(X_train)
    Xt_val = preprocess.transform(X_val)
    y_tr = y_train_reg.to_numpy(dtype=float)
    y_va = y_val_reg.to_numpy(dtype=float)

    n_train = Xt_train.shape[0]
    effective_splits = min(cv_splits, max(2, n_train - 1))
    if effective_splits < 2:
        raise ValueError(f"Not enough training samples for CV: n_train={n_train}")
    cv = TimeSeriesSplit(n_splits=effective_splits)

    if alpha_grid is not None:
        alphas = np.asarray(alpha_grid, dtype=float)
        lcv = LassoCV(alphas=alphas, cv=cv, max_iter=lasso_max_iter, tol=lasso_tol, n_jobs=-1)
    else:
        lcv = LassoCV(alphas=int(n_alphas), cv=cv, max_iter=lasso_max_iter, tol=lasso_tol, n_jobs=-1)

    lcv.fit(Xt_train, y_tr)
    chosen_alpha = float(lcv.alpha_)

    final_model = Lasso(alpha=chosen_alpha, max_iter=lasso_max_iter, tol=lasso_tol)
    final_model.fit(Xt_train, y_tr)
    pred_val = final_model.predict(Xt_val)
    val_mse = float(np.mean((pred_val - y_va) ** 2))

    kept_mask = preprocess.named_steps["var"].get_support()
    kept_cols = [c for c, keep in zip(num_cols, kept_mask) if keep]
    coef_full = pd.Series(final_model.coef_, index=kept_cols)

    # Bootstrap stability selection.
    rng = np.random.default_rng(random_state)
    n_kept = Xt_train.shape[1]
    freq_count = np.zeros(n_kept, dtype=float)

    bs_n = max(8, int(round(n_train * float(sample_frac))))
    for _ in range(int(n_bootstrap)):
        idx = rng.choice(n_train, size=bs_n, replace=True)
        xb = Xt_train[idx]
        yb = y_tr[idx]
        m = Lasso(alpha=chosen_alpha, max_iter=lasso_max_iter, tol=lasso_tol)
        m.fit(xb, yb)
        freq_count += (np.abs(m.coef_) > 1e-10).astype(float)

    stability_freq = freq_count / float(n_bootstrap)
    stable_series = pd.Series(stability_freq, index=kept_cols)

    ranked = pd.DataFrame(
        {
            "feature": kept_cols,
            "coef": coef_full.values,
            "abs_coef": np.abs(coef_full.values),
            "stability_freq": stable_series.values,
        }
    ).sort_values(["stability_freq", "abs_coef"], ascending=[False, False])

    n_ranked = len(ranked)
    cap = min(int(max_features), n_ranked)
    target_min = min(int(min_selected_features), cap)

    stable_keep = ranked.loc[ranked["stability_freq"] >= float(stability_threshold), "feature"].tolist()
    selected = stable_keep[:cap]

    if len(selected) < target_min:
        for feat in ranked["feature"].tolist():
            if feat not in selected:
                selected.append(feat)
            if len(selected) >= target_min:
                break

    if not selected:
        selected = ranked["feature"].tolist()[:cap]

    selected = selected[:cap]

    if detail_path is not None:
        selected_set = set(selected)
        detail_df = ranked.copy()
        detail_df["rank"] = np.arange(1, len(detail_df) + 1)
        detail_df["selected_by_threshold"] = (detail_df["stability_freq"] >= float(stability_threshold)).astype(int)
        detail_df["selected_final"] = detail_df["feature"].isin(selected_set).astype(int)
        safe_detail_path = _safe_path(Path(detail_path))
        detail_df.to_csv(safe_detail_path, index=False, encoding="utf-8-sig")
        LOGGER.info("Saved SLASSO stability ranking: %s", safe_detail_path)

    LOGGER.info(
        "SLASSO selected %d features (alpha=%s, val_mse=%.6g, bootstrap=%d, sample_frac=%s, threshold=%s, min_required=%d, max_cap=%d).",
        len(selected),
        chosen_alpha,
        val_mse,
        n_bootstrap,
        sample_frac,
        stability_threshold,
        target_min,
        cap,
    )
    return selected


def backtest_classifier(
    name: str,
    model: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    next_ret_train: pd.Series,
    next_ret_val: pd.Series,
    next_ret_test: pd.Series,
    next_date_val: pd.Series,
    next_date_test: pd.Series,
    *,
    threshold: Optional[float] = 0.5,
    quantile: Optional[float] = None,
    positioning: str = "long_short",
    cost_per_unit: float = 0.0,
) -> Tuple[Dict[str, float], Dict[str, float], pd.DataFrame, pd.DataFrame]:
    """Fit on train, choose thresholded long/short positions, return val/test perf."""

    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("clf", model),
        ]
    )

    pipe.fit(X_train, y_train)

    def _predict_proba(X: pd.DataFrame) -> np.ndarray:
        if hasattr(pipe, "predict_proba"):
            return pipe.predict_proba(X)[:, 1]
        # Fallback for estimators without predict_proba.
        pred = pipe.predict(X)
        return pred.astype(float)

    p_val = _predict_proba(X_val)
    p_test = _predict_proba(X_test)
    p_train = _predict_proba(X_train)

    q_thr = None
    if positioning == "long_flat_quantile":
        if quantile is None:
            raise ValueError("quantile is required for long_flat_quantile")
        q_thr = float(np.quantile(p_train, float(quantile)))

    pos_val = _positions_from_proba(
        p_val,
        positioning=positioning,
        threshold=threshold,
        quantile=quantile,
        quantile_threshold=q_thr,
    )
    pos_test = _positions_from_proba(
        p_test,
        positioning=positioning,
        threshold=threshold,
        quantile=quantile,
        quantile_threshold=q_thr,
    )

    nr_val = np.asarray(next_ret_val, dtype=float)
    nr_test = np.asarray(next_ret_test, dtype=float)

    net_val, to_val, cost_val, _ = _apply_costs_and_turnover(
        raw_returns=pos_val * nr_val,
        positions=pos_val,
        cost_per_unit=cost_per_unit,
        prev_position=0.0,
    )
    net_test, to_test, cost_test, _ = _apply_costs_and_turnover(
        raw_returns=pos_test * nr_test,
        positions=pos_test,
        cost_per_unit=cost_per_unit,
        prev_position=0.0,
    )

    strat_val = pd.Series(net_val, index=X_val.index)
    strat_test = pd.Series(net_test, index=X_test.index)

    val_stats = _perf_stats(strat_val)
    test_stats = _perf_stats(strat_test)

    val_stats.update(
        {
            "turnover_mean": float(np.mean(to_val)) if len(to_val) else 0.0,
            "turnover_total": float(np.sum(to_val)) if len(to_val) else 0.0,
            "cost_total": float(np.sum(cost_val)) if len(cost_val) else 0.0,
        }
    )
    test_stats.update(
        {
            "turnover_mean": float(np.mean(to_test)) if len(to_test) else 0.0,
            "turnover_total": float(np.sum(to_test)) if len(to_test) else 0.0,
            "cost_total": float(np.sum(cost_test)) if len(cost_test) else 0.0,
        }
    )

    # Classification metrics on test.
    metric_threshold = 0.5 if threshold is None else float(threshold)
    yhat_test = (p_test >= metric_threshold).astype(int)
    acc = float(accuracy_score(y_test, yhat_test))
    try:
        auc = float(roc_auc_score(y_test, p_test))
    except Exception:
        auc = float("nan")

    trades_val = pd.DataFrame(
        {
            "date": X_val.index,
            "next_date": next_date_val.values,
            "prob_up": p_val,
            "position": pos_val,
            "next_return": nr_val,
            "turnover": to_val,
            "cost": cost_val,
            "strategy_return": strat_val.to_numpy(),
        }
    ).reset_index(drop=True)

    trades_test = pd.DataFrame(
        {
            "date": X_test.index,
            "next_date": next_date_test.values,
            "prob_up": p_test,
            "position": pos_test,
            "next_return": nr_test,
            "turnover": to_test,
            "cost": cost_test,
            "strategy_return": strat_test.to_numpy(),
        }
    ).reset_index(drop=True)

    test_stats = {
        **test_stats,
        "accuracy": acc,
        "auc": auc,
    }

    LOGGER.info(
        "%s: val_cum=%.2f%%, test_cum=%.2f%%, test_sharpe=%.2f, to_mean=%.3g, cost=%.4g, acc=%.2f%%, auc=%.3f",
        name,
        val_stats["cum"] * 100.0,
        test_stats["cum"] * 100.0,
        test_stats["sharpe"],
        test_stats["turnover_mean"],
        test_stats["cost_total"],
        acc * 100.0,
        auc,
    )
    return val_stats, test_stats, trades_val, trades_test


def backtest_classifier_test_only(
    name: str,
    model: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    next_ret_test: pd.Series,
    next_date_test: pd.Series,
    *,
    threshold: Optional[float] = 0.5,
    quantile: Optional[float] = None,
    positioning: str = "long_short",
    cost_per_unit: float = 0.0,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Fit on X_train/y_train, backtest on X_test only."""

    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("clf", model),
        ]
    )

    pipe.fit(X_train, y_train)

    if hasattr(pipe, "predict_proba"):
        p_test = pipe.predict_proba(X_test)[:, 1]
        p_train = pipe.predict_proba(X_train)[:, 1]
    else:
        p_test = pipe.predict(X_test).astype(float)
        p_train = pipe.predict(X_train).astype(float)

    q_thr = None
    if positioning == "long_flat_quantile":
        if quantile is None:
            raise ValueError("quantile is required for long_flat_quantile")
        q_thr = float(np.quantile(p_train, float(quantile)))

    pos_test = _positions_from_proba(
        p_test,
        positioning=positioning,
        threshold=threshold,
        quantile=quantile,
        quantile_threshold=q_thr,
    )

    nr_test = np.asarray(next_ret_test, dtype=float)
    net_test, to_test, cost_test, _ = _apply_costs_and_turnover(
        raw_returns=pos_test * nr_test,
        positions=pos_test,
        cost_per_unit=cost_per_unit,
        prev_position=0.0,
    )

    strat_test = pd.Series(net_test, index=X_test.index)
    test_stats = _perf_stats(strat_test)
    test_stats.update(
        {
            "turnover_mean": float(np.mean(to_test)) if len(to_test) else 0.0,
            "turnover_total": float(np.sum(to_test)) if len(to_test) else 0.0,
            "cost_total": float(np.sum(cost_test)) if len(cost_test) else 0.0,
        }
    )

    yhat_test = (p_test >= 0.5).astype(int)
    acc = float(accuracy_score(y_test, yhat_test))
    try:
        auc = float(roc_auc_score(y_test, p_test))
    except Exception:
        auc = float("nan")

    test_stats.update({"accuracy": acc, "auc": auc})

    trades_test = pd.DataFrame(
        {
            "date": X_test.index,
            "next_date": next_date_test.values,
            "prob_up": p_test,
            "position": pos_test,
            "next_return": nr_test,
            "turnover": to_test,
            "cost": cost_test,
            "strategy_return": strat_test.to_numpy(),
        }
    ).reset_index(drop=True)

    LOGGER.info(
        "%s: test_cum=%.2f%%, test_sharpe=%.2f, to_mean=%.3g, cost=%.4g, acc=%.2f%%, auc=%.3f",
        name,
        test_stats["cum"] * 100.0,
        test_stats["sharpe"],
        test_stats["turnover_mean"],
        test_stats["cost_total"],
        acc * 100.0,
        auc,
    )
    return test_stats, trades_test


def main() -> None:
    ap = argparse.ArgumentParser(description="TSLA factor selection + ML backtest (LASSO/SLASSO/Ridge + RF/MLP/SVM + Bagging/GBDT/Stacking + Bayesian tuning)")
    ap.add_argument("--data", default="data/TSLA_Merged_V3.csv")
    ap.add_argument(
        "--ai-data",
        default="data/tsla_ai_narrative_daily_from_rpa.csv",
        help="Daily AI narrative factor file; its columns identify the AI features in --data.",
    )
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--threshold", type=float, default=0.5, help="Fixed threshold when --threshold-mode=fixed")
    ap.add_argument(
        "--threshold-mode",
        choices=["fixed", "auto"],
        default="auto",
        help="Use a fixed threshold or select it on validation via walk-forward.",
    )
    ap.add_argument(
        "--threshold-grid",
        default="0.4,0.45,0.5,0.55,0.6",
        help="Comma-separated threshold candidates when --threshold-mode=auto.",
    )
    ap.add_argument(
        "--quantile-grid",
        default="0.6,0.7,0.8,0.9",
        help="Comma-separated quantile candidates when positioning=long_flat_quantile.",
    )
    ap.add_argument(
        "--walkforward-splits",
        type=int,
        default=5,
        help="Number of walk-forward splits for model/threshold selection on train+val.",
    )
    ap.add_argument(
        "--selection-metric",
        choices=["cum", "sharpe", "ann"],
        default="ann",
        help="Metric used to pick hyperparameters/threshold on walk-forward validation.",
    )
    ap.add_argument(
        "--bayes-trials",
        type=int,
        default=12,
        help="Number of Bayesian optimization trials for ensemble models (Bagging/GBDT/Stacking).",
    )
    ap.add_argument(
        "--bayes-startup-trials",
        type=int,
        default=4,
        help="Number of startup random trials before TPE Bayesian sampling.",
    )
    ap.add_argument(
        "--positioning",
        choices=["long_short", "long_flat", "long_flat_quantile"],
        default="long_flat",
    )
    ap.add_argument(
        "--cost-bps",
        type=float,
        default=5.0,
        help="Transaction cost + slippage in bps per 1.0 unit position change (0->1 costs 1x, -1->1 costs 2x).",
    )
    ap.add_argument("--max-features", type=int, default=120)
    ap.add_argument(
        "--ai-missing-threshold",
        type=float,
        default=0.60,
        help="Drop AI columns with train missing rate above this threshold before feature selection.",
    )
    ap.add_argument(
        "--ai-corr-threshold",
        type=float,
        default=0.95,
        help="Drop AI columns whose absolute train correlation with a kept AI column is above this threshold.",
    )
    ap.add_argument(
        "--lasso-min-features",
        type=int,
        default=30,
        help="Prefer LASSO alphas that yield at least this many non-zero coefficients (falls back to best MSE if impossible).",
    )
    ap.add_argument(
        "--lasso-cv-splits",
        type=int,
        default=5,
        help="Number of time-series CV splits used to select LASSO alpha on the training set.",
    )
    ap.add_argument(
        "--lasso-n-alphas",
        type=int,
        default=60,
        help="Number of penalty steps (alphas) along the LASSO regularization path when alpha grid is not explicitly provided.",
    )
    ap.add_argument(
        "--ridge-cv-splits",
        type=int,
        default=5,
        help="Number of time-series CV splits used to select Ridge alpha on the training set.",
    )
    ap.add_argument(
        "--ridge-max-features",
        type=int,
        default=60,
        help="Maximum number of Ridge-selected features.",
    )
    ap.add_argument(
        "--ridge-min-features",
        type=int,
        default=20,
        help="Minimum number of Ridge-selected features after threshold filtering.",
    )
    ap.add_argument(
        "--ridge-min-coef-ratio",
        type=float,
        default=0.10,
        help="Keep Ridge factors with |coef| >= ratio * max(|coef|), then apply min/max feature bounds.",
    )
    ap.add_argument(
        "--feature-selector",
        choices=["lasso", "slasso", "ridge"],
        default="lasso",
        help="Which method to use to select features.",
    )
    ap.add_argument(
        "--slasso-min-features",
        type=int,
        default=30,
        help="Minimum number of SLASSO-selected features after stability thresholding.",
    )
    ap.add_argument(
        "--slasso-cv-splits",
        type=int,
        default=5,
        help="Number of time-series CV splits used to select SLASSO alpha on the training set.",
    )
    ap.add_argument(
        "--slasso-n-alphas",
        type=int,
        default=60,
        help="Number of alpha steps used by LassoCV inside SLASSO when alpha grid is not provided.",
    )
    ap.add_argument(
        "--slasso-bootstrap",
        type=int,
        default=80,
        help="Number of bootstrap resamples for SLASSO stability frequencies.",
    )
    ap.add_argument(
        "--slasso-sample-frac",
        type=float,
        default=0.75,
        help="Bootstrap sample fraction of train rows for each SLASSO resample.",
    )
    ap.add_argument(
        "--slasso-threshold",
        type=float,
        default=0.60,
        help="Stability frequency threshold for SLASSO feature inclusion.",
    )
    ap.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity for console and file outputs.",
    )
    ap.add_argument(
        "--log-file",
        default="backtest.log",
        help="Log file name saved under --out directory.",
    )
    ap.add_argument(
        "--compute-r2",
        action="store_true",
        help="Compute R^2 from saved *_trades_test.csv files in the output folder and exit.",
    )
    ap.add_argument(
        "--monthly-r2",
        action="store_true",
        help="Compute monthly out-of-sample R^2 pooled across the panel of saved *_trades_test.csv files and exit.",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = _setup_logging(out_dir, level=args.log_level, log_file=args.log_file)
    LOGGER.info("Run started with args: %s", vars(args))

    df = pd.read_csv(args.data, parse_dates=["date"])
    df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True).copy()
    LOGGER.info("Loaded data: rows=%d, cols=%d, data_path=%s", df.shape[0], df.shape[1], args.data)
    # If user only wants to compute R^2 from saved predictions, do that and exit.
    if args.compute_r2:
        def _compute_r2_from_saved(out_dir: Path) -> Dict[str, float]:
            results: Dict[str, float] = {}
            files = glob(str(out_dir / "*_trades_test.csv"))
            if not files:
                LOGGER.warning("No *_trades_test.csv files found in %s", out_dir)
                return results
            for fp in sorted(files):
                try:
                    dfp = pd.read_csv(fp)
                except Exception as e:
                    LOGGER.error("Failed to read %s: %s", fp, e)
                    continue
                base = Path(fp).stem
                model_name = base.replace("_trades_test", "")
                if "next_return" not in dfp.columns:
                    LOGGER.warning("File %s missing 'next_return' column; skip.", fp)
                    continue
                # Use 'prob_up' if available as the prediction; else use 'position' or 'strategy_return'.
                if "prob_up" in dfp.columns:
                    y_pred = pd.to_numeric(dfp["prob_up"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                elif "position" in dfp.columns:
                    y_pred = pd.to_numeric(dfp["position"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                elif "strategy_return" in dfp.columns:
                    y_pred = pd.to_numeric(dfp["strategy_return"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                else:
                    LOGGER.warning("No suitable prediction column in %s; skip.", fp)
                    continue

                y_true = pd.to_numeric(dfp["next_return"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                # Compute R^2 = 1 - SS_res / SS_tot
                ss_res = float(((y_true - y_pred) ** 2).sum())
                y_mean = float(y_true.mean()) if len(y_true) else 0.0
                ss_tot = float(((y_true - y_mean) ** 2).sum())
                r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")
                results[model_name] = r2
                LOGGER.info("R^2 for %s (file=%s): %.6f", model_name, fp, r2)
            return results

        r2_results = _compute_r2_from_saved(out_dir)
        if not r2_results:
            LOGGER.info("No R^2 results computed.")
        else:
            for m, r in r2_results.items():
                LOGGER.info("%s: R^2=%.6f", m, r)
        return
    if args.monthly_r2:
        def _compute_monthly_r2_from_saved(out_dir: Path) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
            # Returns (per_model_dfs, panel_df)
            per_model: Dict[str, pd.DataFrame] = {}
            panel_month = defaultdict(lambda: {"y_true": [], "y_pred": []})
            files = glob(str(out_dir / "*_trades_test.csv"))
            if not files:
                LOGGER.warning("No *_trades_test.csv files found in %s", out_dir)
                return per_model, pd.DataFrame()

            for fp in sorted(files):
                try:
                    dfp = pd.read_csv(fp)
                except Exception as e:
                    LOGGER.error("Failed to read %s: %s", fp, e)
                    continue
                base = Path(fp).stem
                model_name = base.replace("_trades_test", "")

                # Determine date column to use for grouping (use next_date if present)
                if "next_date" in dfp.columns:
                    dates = pd.to_datetime(dfp["next_date"], errors="coerce")
                elif "date" in dfp.columns:
                    dates = pd.to_datetime(dfp["date"], errors="coerce")
                else:
                    LOGGER.warning("File %s missing date columns; skip.", fp)
                    continue

                if "next_return" not in dfp.columns:
                    LOGGER.warning("File %s missing 'next_return' column; skip.", fp)
                    continue

                # choose prediction column
                if "prob_up" in dfp.columns:
                    preds = pd.to_numeric(dfp["prob_up"], errors="coerce")
                elif "position" in dfp.columns:
                    preds = pd.to_numeric(dfp["position"], errors="coerce")
                elif "strategy_return" in dfp.columns:
                    preds = pd.to_numeric(dfp["strategy_return"], errors="coerce")
                else:
                    LOGGER.warning("No suitable prediction column in %s; skip.", fp)
                    continue

                truths = pd.to_numeric(dfp["next_return"], errors="coerce")

                df_local = pd.DataFrame({"date": dates, "y_true": truths, "y_pred": preds})
                df_local = df_local.dropna(subset=["date", "y_true"]).copy()
                if df_local.empty:
                    LOGGER.warning("No valid rows in %s after dropping NaNs; skip.", fp)
                    continue

                df_local["month"] = df_local["date"].dt.to_period("M").astype(str)

                rows = []
                for month, g in df_local.groupby("month"):
                    y_true = g["y_true"].to_numpy(dtype=float)
                    y_pred = g["y_pred"].fillna(0.0).to_numpy(dtype=float)
                    ss_res = float(((y_true - y_pred) ** 2).sum())
                    y_mean = float(y_true.mean()) if len(y_true) else 0.0
                    ss_tot = float(((y_true - y_mean) ** 2).sum())
                    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")
                    rows.append({"month": month, "r2": r2, "n": len(y_true)})

                    # append to panel aggregator
                    panel_month[month]["y_true"].extend(y_true.tolist())
                    panel_month[month]["y_pred"].extend(y_pred.tolist())

                per_model_df = pd.DataFrame(rows).sort_values("month")
                per_model[model_name] = per_model_df
                outp = _safe_path(out_dir / f"{model_name}_monthly_r2.csv")
                per_model_df.to_csv(outp, index=False, encoding="utf-8-sig")
                LOGGER.info("Saved monthly R^2 for %s -> %s", model_name, outp)

            # compute panel-level monthly R2
            panel_rows = []
            for month in sorted(panel_month.keys()):
                y_true = np.asarray(panel_month[month]["y_true"], dtype=float)
                y_pred = np.asarray(panel_month[month]["y_pred"], dtype=float)
                if y_true.size == 0:
                    panel_rows.append({"month": month, "r2": float("nan"), "n": 0})
                    continue
                ss_res = float(((y_true - y_pred) ** 2).sum())
                y_mean = float(y_true.mean())
                ss_tot = float(((y_true - y_mean) ** 2).sum())
                r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")
                panel_rows.append({"month": month, "r2": r2, "n": int(y_true.size)})

            panel_df = pd.DataFrame(panel_rows).sort_values("month")
            panel_out = _safe_path(out_dir / "panel_monthly_r2.csv")
            panel_df.to_csv(panel_out, index=False, encoding="utf-8-sig")
            LOGGER.info("Saved panel monthly R^2 -> %s", panel_out)

            return per_model, panel_df

        per_model_results, panel_df = _compute_monthly_r2_from_saved(out_dir)
        if panel_df.empty:
            LOGGER.info("No panel monthly R^2 computed.")
        else:
            LOGGER.info("Panel monthly R^2 computed with %d months.", len(panel_df))
        return

    # Target: next-day simple return sign.
    # Use assign + copy to avoid pandas fragmented-frame warnings on wide data.
    df = df.assign(
        next_return=df["returns"].shift(-1),
        next_date=df["date"].shift(-1),
    ).copy()
    df = df.dropna(subset=["next_return"]).copy()

    # Binary classification label: up (1) if next_return > 0.
    df["y_up"] = (df["next_return"] > 0).astype(int)

    # Features: numeric columns excluding the explicit target/label.
    numeric = df.select_dtypes(include=[np.number]).copy()
    drop_cols = {"next_return", "y_up"}
    feature_cols = [c for c in numeric.columns if c not in drop_cols]
    X_all = numeric[feature_cols].dropna(axis=1, how="all")

    # Keep date as index for reporting.
    dates = df["date"]
    X_all.index = dates
    y_all = df["y_up"].astype(int)
    next_ret_all = df["next_return"].astype(float)
    next_date_all = df["next_date"]

    n = len(df)
    tr_idx, va_idx, te_idx = _time_split_index(n, args.train_frac, args.val_frac)

    X_train, X_val, X_test = X_all.iloc[tr_idx], X_all.iloc[va_idx], X_all.iloc[te_idx]
    y_train, y_val, y_test = y_all.iloc[tr_idx], y_all.iloc[va_idx], y_all.iloc[te_idx]
    nr_train, nr_val, nr_test = next_ret_all.iloc[tr_idx], next_ret_all.iloc[va_idx], next_ret_all.iloc[te_idx]
    nd_train, nd_val, nd_test = next_date_all.iloc[tr_idx], next_date_all.iloc[va_idx], next_date_all.iloc[te_idx]

    ai_source_cols = set(pd.read_csv(args.ai_data, nrows=1).columns)
    ai_feature_cols = [c for c in X_all.columns if c in ai_source_cols and c != "date"]
    selected_ai_cols, ai_filter_report = _filter_ai_features_for_backtest(
        X_train=X_train,
        ai_feature_cols=ai_feature_cols,
        missing_threshold=float(args.ai_missing_threshold),
        corr_threshold=float(args.ai_corr_threshold),
    )
    removed_ai_cols = [c for c in ai_feature_cols if c not in selected_ai_cols]
    kept_cols = [c for c in X_all.columns if c not in ai_feature_cols] + selected_ai_cols
    X_all = X_all[kept_cols].copy()
    X_train = X_train[kept_cols].copy()
    X_val = X_val[kept_cols].copy()
    X_test = X_test[kept_cols].copy()
    ai_filter_path = _safe_path(out_dir / "ai_feature_filter_report.csv")
    ai_filter_report.to_csv(ai_filter_path, index=False, encoding="utf-8-sig")
    LOGGER.info(
        "AI feature filter: kept=%d removed=%d (missing_threshold=%.2f, corr_threshold=%.2f)",
        len(selected_ai_cols),
        len(removed_ai_cols),
        float(args.ai_missing_threshold),
        float(args.ai_corr_threshold),
    )
    if removed_ai_cols:
        LOGGER.info("Removed AI features: %s", removed_ai_cols)
    LOGGER.info("Saved AI feature filter report -> %s", ai_filter_path)
    LOGGER.info(
        "Split sizes: train=%d, val=%d, test=%d; features=%d",
        len(X_train),
        len(X_val),
        len(X_test),
        X_all.shape[1],
    )

    # Development set used for walk-forward selection.
    X_dev = pd.concat([X_train, X_val], axis=0)
    y_dev = pd.concat([y_train, y_val], axis=0)
    nr_dev = pd.concat([nr_train, nr_val], axis=0)
    nd_dev = pd.concat([nd_train, nd_val], axis=0)

    # Always run both explicitly to compare factors, then use the requested selector.
    lasso_coef_plot_path = _safe_path(out_dir / "lasso_coef_path_log_lambda.png")
    selected_lasso = select_features_lasso(
        X_train,
        nr_train,
        X_val,
        nr_val,
        max_features=args.max_features,
        min_selected_features=args.lasso_min_features,
        cv_splits=args.lasso_cv_splits,
        n_alphas=args.lasso_n_alphas,
        random_state=args.seed,
        lasso_plot_path=lasso_coef_plot_path,
    )

    ridge_coef_detail_path = _safe_path(out_dir / "ridge_coef_ranked.csv")
    selected_ridge = select_features_ridge(
        X_train,
        nr_train,
        X_val,
        nr_val,
        max_features=args.ridge_max_features,
        min_selected_features=args.ridge_min_features,
        min_coef_ratio=args.ridge_min_coef_ratio,
        cv_splits=args.ridge_cv_splits,
        coef_detail_path=ridge_coef_detail_path,
        random_state=args.seed,
    )

    slasso_detail_path = _safe_path(out_dir / "slasso_stability_ranked.csv")
    selected_slasso = select_features_slasso(
        X_train,
        nr_train,
        X_val,
        nr_val,
        max_features=args.max_features,
        min_selected_features=args.slasso_min_features,
        cv_splits=args.slasso_cv_splits,
        n_alphas=args.slasso_n_alphas,
        n_bootstrap=args.slasso_bootstrap,
        sample_frac=args.slasso_sample_frac,
        stability_threshold=args.slasso_threshold,
        detail_path=slasso_detail_path,
        random_state=args.seed,
    )

    lasso_set = set(selected_lasso)
    slasso_set = set(selected_slasso)
    ridge_set = set(selected_ridge)
    overlap = lasso_set.intersection(ridge_set)
    overlap_pct = (len(overlap) / max(1, len(lasso_set))) * 100.0
    LOGGER.info("--- Feature Selection Comparison ---")
    LOGGER.info("LASSO features: %d", len(lasso_set))
    LOGGER.info("SLASSO features: %d", len(slasso_set))
    LOGGER.info("Ridge features: %d", len(ridge_set))
    LOGGER.info("Overlap size: %d (%.1f%% of LASSO)", len(overlap), overlap_pct)
    LOGGER.info("LASSO ∩ SLASSO: %d", len(lasso_set.intersection(slasso_set)))
    LOGGER.info("SLASSO ∩ Ridge: %d", len(slasso_set.intersection(ridge_set)))
    LOGGER.info("LASSO only: %s", lasso_set - ridge_set)
    LOGGER.info("SLASSO only: %s", slasso_set - lasso_set - ridge_set)
    LOGGER.info("Ridge only: %s", ridge_set - lasso_set)

    fs_compare_artifacts = _save_feature_selector_comparison(
        selected_lasso=selected_lasso,
        selected_ridge=selected_ridge,
        out_dir=out_dir,
        top_k=min(50, max(len(selected_lasso), len(selected_ridge))),
    )

    if args.feature_selector == "lasso":
        selected = selected_lasso
        LOGGER.info(">> Using LASSO features for down-stream modeling.")
    elif args.feature_selector == "slasso":
        selected = selected_slasso
        LOGGER.info(">> Using SLASSO features for down-stream modeling.")
    else:
        selected = selected_ridge
        LOGGER.info(">> Using Ridge features for down-stream modeling.")

    if not selected:
        raise RuntimeError(f"{args.feature_selector.upper()} selected 0 features; check data scaling/missingness.")

    X_train_s = X_train[selected]
    X_val_s = X_val[selected]
    X_test_s = X_test[selected]

    cost_per_unit = float(args.cost_bps) / 10000.0

    # Baseline: always long.
    # - validation metric: walk-forward OOS on dev (train+val)
    # - test metric: holdout test
    n_dev = len(nr_dev)
    cv_dev = TimeSeriesSplit(n_splits=min(int(args.walkforward_splits), max(2, n_dev - 1)))
    oos_base: List[float] = []
    oos_to: List[float] = []
    oos_cost: List[float] = []
    prev_pos = 0.0
    for _, fold_val in cv_dev.split(np.arange(n_dev)):
        nr_va = np.asarray(nr_dev.iloc[fold_val], dtype=float)
        pos_va = np.ones_like(nr_va, dtype=float)
        net, turnover, costs, prev_pos = _apply_costs_and_turnover(
            raw_returns=nr_va,
            positions=pos_va,
            cost_per_unit=cost_per_unit,
            prev_position=prev_pos,
        )
        oos_base.extend(net.tolist())
        oos_to.extend(turnover.tolist())
        oos_cost.extend(costs.tolist())
    baseline_val = _perf_stats(pd.Series(oos_base))
    baseline_val.update(
        {
            "turnover_mean": float(np.mean(oos_to)) if oos_to else 0.0,
            "turnover_total": float(np.sum(oos_to)) if oos_to else 0.0,
            "cost_total": float(np.sum(oos_cost)) if oos_cost else 0.0,
        }
    )

    nr_te = np.asarray(nr_test, dtype=float)
    pos_te = np.ones_like(nr_te, dtype=float)
    net_te, to_te, cost_te, _ = _apply_costs_and_turnover(
        raw_returns=nr_te,
        positions=pos_te,
        cost_per_unit=cost_per_unit,
        prev_position=0.0,
    )
    baseline_test = _perf_stats(pd.Series(net_te, index=X_test.index))
    baseline_test.update(
        {
            "turnover_mean": float(np.mean(to_te)) if len(to_te) else 0.0,
            "turnover_total": float(np.sum(to_te)) if len(to_te) else 0.0,
            "cost_total": float(np.sum(cost_te)) if len(cost_te) else 0.0,
        }
    )

    LOGGER.info(
        "Baseline (always long): wf_oos_ann=%.2f%%, wf_oos_cum=%.2f%%, test_cum=%.2f%%, test_sharpe=%.2f, cost=%.4g",
        baseline_val["ann"] * 100.0,
        baseline_val["cum"] * 100.0,
        baseline_test["cum"] * 100.0,
        baseline_test["sharpe"],
        baseline_test["cost_total"],
    )

    results: List[BacktestResult] = []

    threshold_grid = _parse_float_grid(args.threshold_grid) or [args.threshold]
    quantile_grid = _parse_float_grid(args.quantile_grid) or [0.8]

    # ---- Random Forest (small grid, pick best by validation cumulative return) ----
    rf_grid = [
        {"n_estimators": 500, "max_depth": 4, "min_samples_leaf": 50},
        {"n_estimators": 800, "max_depth": 6, "min_samples_leaf": 30},
        {"n_estimators": 800, "max_depth": None, "min_samples_leaf": 20},
    ]

    best_rf = None
    best_rf_stats = None
    best_rf_trades_test = None
    best_rf_val_ann = -1e9

    def _rf_factory(params: Dict) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            random_state=args.seed,
            n_jobs=-1,
        )

    best_rf, best_rf_thr, best_rf_q, rf_wf_stats = _walk_forward_select(
        name="RF",
        model_factory=_rf_factory,
        param_grid=rf_grid,
        X_dev=X_dev[selected],
        y_dev=y_dev,
        next_ret_dev=nr_dev,
        next_date_dev=nd_dev,
        cv_splits=args.walkforward_splits,
        positioning=args.positioning,
        threshold_mode=args.threshold_mode,
        threshold_grid=[args.threshold] if args.threshold_mode == "fixed" else threshold_grid,
        quantile_grid=quantile_grid,
        selection_metric=args.selection_metric,
        cost_per_unit=cost_per_unit,
        seed=args.seed,
    )

    rf_best_model = _rf_factory(best_rf)
    best_rf_stats, best_rf_trades_test = backtest_classifier_test_only(
        name=f"RF_best({best_rf})",
        model=rf_best_model,
        X_train=X_dev[selected],
        y_train=y_dev,
        X_test=X_test_s,
        y_test=y_test,
        next_ret_test=nr_test,
        next_date_test=nd_test,
        threshold=best_rf_thr,
        quantile=best_rf_q,
        positioning=args.positioning,
        cost_per_unit=cost_per_unit,
    )
    best_rf_val_ann = rf_wf_stats["ann"]

    assert best_rf is not None and best_rf_stats is not None and best_rf_trades_test is not None

    # ---- MLP (neural network) with Bayesian optimization ----

    best_mlp = None
    best_mlp_stats = None
    best_mlp_trades_test = None
    best_mlp_val_ann = -1e9
    mlp_opt_method = ""

    _mlp_hidden_map = {
        "32": (32,),
        "64": (64,),
        "128": (128,),
        "64_32": (64, 32),
        "128_64": (128, 64),
        "256_128": (256, 128),
        "128_64_32": (128, 64, 32),
        "256_128_64": (256, 128, 64),
        "256_128_64_32": (256, 128, 64, 32),
        "256_128_64_32_16": (256, 128, 64, 32, 16),
    }

    def _mlp_factory(params: Dict) -> MLPClassifier:
        hidden_key = params["hidden_layer_key"]
        return MLPClassifier(
            hidden_layer_sizes=_mlp_hidden_map[hidden_key],
            alpha=params["alpha"],
            activation=params["activation"],
            learning_rate_init=params["learning_rate_init"],
            batch_size=params["batch_size"],
            random_state=args.seed,
            max_iter=800,
            early_stopping=False,
            n_iter_no_change=12,
        )

    mlp_space = {
        "hidden_layer_key": {
            "type": "categorical",
            "choices": [
                "32",
                "64",
                "128",
                "64_32",
                "128_64",
                "256_128",
                "128_64_32",
                "256_128_64",
                "256_128_64_32",
                "256_128_64_32_16",
            ],
        },
        "alpha": {"type": "float", "low": 1e-6, "high": 1e-2, "log": True},
        "learning_rate_init": {"type": "float", "low": 1e-4, "high": 5e-2, "log": True},
        "batch_size": {"type": "categorical", "choices": [32, 64, 128, 256]},
        "activation": {"type": "categorical", "choices": ["relu", "tanh"]},
    }

    best_mlp, best_mlp_thr, best_mlp_q, mlp_wf_stats, mlp_opt_method = _optimize_params_walk_forward(
        name="MLP",
        model_factory=_mlp_factory,
        search_space=mlp_space,
        n_trials=args.bayes_trials,
        startup_trials=args.bayes_startup_trials,
        seed=args.seed,
        X_dev=X_dev[selected],
        y_dev=y_dev,
        next_ret_dev=nr_dev,
        next_date_dev=nd_dev,
        cv_splits=args.walkforward_splits,
        positioning=args.positioning,
        threshold_mode=args.threshold_mode,
        threshold_grid=[args.threshold] if args.threshold_mode == "fixed" else threshold_grid,
        quantile_grid=quantile_grid,
        selection_metric=args.selection_metric,
        cost_per_unit=cost_per_unit,
    )

    mlp_best_model = _mlp_factory(best_mlp)
    best_mlp_stats, best_mlp_trades_test = backtest_classifier_test_only(
        name=f"MLP_best({best_mlp})",
        model=mlp_best_model,
        X_train=X_dev[selected],
        y_train=y_dev,
        X_test=X_test_s,
        y_test=y_test,
        next_ret_test=nr_test,
        next_date_test=nd_test,
        threshold=best_mlp_thr,
        quantile=best_mlp_q,
        positioning=args.positioning,
        cost_per_unit=cost_per_unit,
    )
    best_mlp_val_ann = mlp_wf_stats["ann"]

    assert best_mlp is not None and best_mlp_stats is not None and best_mlp_trades_test is not None

    # ---- SVM (SVC) ----
    svm_grid = [
        {"kernel": "rbf", "C": 1.0, "gamma": "scale"},
        {"kernel": "rbf", "C": 3.0, "gamma": "scale"},
        {"kernel": "linear", "C": 1.0},
    ]

    best_svm = None
    best_svm_stats = None
    best_svm_trades_test = None
    best_svm_val_ann = -1e9

    def _svm_factory(params: Dict) -> SVC:
        return SVC(
            kernel=params["kernel"],
            C=params["C"],
            gamma=params.get("gamma", "scale"),
            probability=True,
            random_state=args.seed,
        )

    best_svm, best_svm_thr, best_svm_q, svm_wf_stats = _walk_forward_select(
        name="SVM",
        model_factory=_svm_factory,
        param_grid=svm_grid,
        X_dev=X_dev[selected],
        y_dev=y_dev,
        next_ret_dev=nr_dev,
        next_date_dev=nd_dev,
        cv_splits=args.walkforward_splits,
        positioning=args.positioning,
        threshold_mode=args.threshold_mode,
        threshold_grid=[args.threshold] if args.threshold_mode == "fixed" else threshold_grid,
        quantile_grid=quantile_grid,
        selection_metric=args.selection_metric,
        cost_per_unit=cost_per_unit,
        seed=args.seed,
    )

    svm_best_model = _svm_factory(best_svm)
    best_svm_stats, best_svm_trades_test = backtest_classifier_test_only(
        name=f"SVM_best({best_svm})",
        model=svm_best_model,
        X_train=X_dev[selected],
        y_train=y_dev,
        X_test=X_test_s,
        y_test=y_test,
        next_ret_test=nr_test,
        next_date_test=nd_test,
        threshold=best_svm_thr,
        quantile=best_svm_q,
        positioning=args.positioning,
        cost_per_unit=cost_per_unit,
    )
    best_svm_val_ann = svm_wf_stats["ann"]

    assert best_svm is not None and best_svm_stats is not None and best_svm_trades_test is not None

    # ---- Boosting (GBDT) with Bayesian optimization ----

    best_gbdt = None
    best_gbdt_stats = None
    best_gbdt_trades_test = None
    best_gbdt_val_ann = -1e9
    gbdt_opt_method = ""

    def _gbdt_factory(params: Dict) -> GradientBoostingClassifier:
        return GradientBoostingClassifier(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            subsample=params["subsample"],
            random_state=args.seed,
        )

    gbdt_space = {
        "n_estimators": {"type": "int", "low": 150, "high": 600, "step": 50},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.2, "log": True},
        "max_depth": {"type": "int", "low": 2, "high": 4, "step": 1},
        "subsample": {"type": "float", "low": 0.6, "high": 1.0},
    }

    best_gbdt, best_gbdt_thr, best_gbdt_q, gbdt_wf_stats, gbdt_opt_method = _optimize_params_walk_forward(
        name="GBDT",
        model_factory=_gbdt_factory,
        search_space=gbdt_space,
        n_trials=args.bayes_trials,
        startup_trials=args.bayes_startup_trials,
        seed=args.seed,
        X_dev=X_dev[selected],
        y_dev=y_dev,
        next_ret_dev=nr_dev,
        next_date_dev=nd_dev,
        cv_splits=args.walkforward_splits,
        positioning=args.positioning,
        threshold_mode=args.threshold_mode,
        threshold_grid=[args.threshold] if args.threshold_mode == "fixed" else threshold_grid,
        quantile_grid=quantile_grid,
        selection_metric=args.selection_metric,
        cost_per_unit=cost_per_unit,
    )

    gbdt_best_model = _gbdt_factory(best_gbdt)
    best_gbdt_stats, best_gbdt_trades_test = backtest_classifier_test_only(
        name=f"GBDT_best({best_gbdt})",
        model=gbdt_best_model,
        X_train=X_dev[selected],
        y_train=y_dev,
        X_test=X_test_s,
        y_test=y_test,
        next_ret_test=nr_test,
        next_date_test=nd_test,
        threshold=best_gbdt_thr,
        quantile=best_gbdt_q,
        positioning=args.positioning,
        cost_per_unit=cost_per_unit,
    )
    best_gbdt_val_ann = gbdt_wf_stats["ann"]

    assert best_gbdt is not None and best_gbdt_stats is not None and best_gbdt_trades_test is not None

    # ---- Bagging with Bayesian optimization ----
    best_bag = None
    best_bag_stats = None
    best_bag_trades_test = None
    best_bag_val_ann = -1e9
    bag_opt_method = ""

    def _bagging_factory(params: Dict) -> BaggingClassifier:
        tree = DecisionTreeClassifier(
            max_depth=params["tree_max_depth"],
            min_samples_leaf=params["tree_min_samples_leaf"],
            random_state=args.seed,
        )
        return BaggingClassifier(
            estimator=tree,
            n_estimators=params["n_estimators"],
            max_samples=params["max_samples"],
            max_features=params["max_features"],
            bootstrap=True,
            random_state=args.seed,
            n_jobs=-1,
        )

    bag_space = {
        "n_estimators": {"type": "int", "low": 100, "high": 500, "step": 50},
        "max_samples": {"type": "float", "low": 0.5, "high": 1.0},
        "max_features": {"type": "float", "low": 0.4, "high": 1.0},
        "tree_max_depth": {"type": "categorical", "choices": [None, 3, 5, 8]},
        "tree_min_samples_leaf": {"type": "int", "low": 5, "high": 60, "step": 5},
    }

    best_bag, best_bag_thr, best_bag_q, bag_wf_stats, bag_opt_method = _optimize_params_walk_forward(
        name="Bagging",
        model_factory=_bagging_factory,
        search_space=bag_space,
        n_trials=args.bayes_trials,
        startup_trials=args.bayes_startup_trials,
        seed=args.seed,
        X_dev=X_dev[selected],
        y_dev=y_dev,
        next_ret_dev=nr_dev,
        next_date_dev=nd_dev,
        cv_splits=args.walkforward_splits,
        positioning=args.positioning,
        threshold_mode=args.threshold_mode,
        threshold_grid=[args.threshold] if args.threshold_mode == "fixed" else threshold_grid,
        quantile_grid=quantile_grid,
        selection_metric=args.selection_metric,
        cost_per_unit=cost_per_unit,
    )

    bag_best_model = _bagging_factory(best_bag)
    best_bag_stats, best_bag_trades_test = backtest_classifier_test_only(
        name=f"Bagging_best({best_bag})",
        model=bag_best_model,
        X_train=X_dev[selected],
        y_train=y_dev,
        X_test=X_test_s,
        y_test=y_test,
        next_ret_test=nr_test,
        next_date_test=nd_test,
        threshold=best_bag_thr,
        quantile=best_bag_q,
        positioning=args.positioning,
        cost_per_unit=cost_per_unit,
    )
    best_bag_val_ann = bag_wf_stats["ann"]

    assert best_bag is not None and best_bag_stats is not None and best_bag_trades_test is not None

    # ---- Stacking with Bayesian optimization ----
    best_stack = None
    best_stack_stats = None
    best_stack_trades_test = None
    best_stack_val_ann = -1e9
    stack_opt_method = ""

    def _stacking_factory(params: Dict) -> TimeSeriesStackingClassifier:
        estimators = [
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=params["rf_n_estimators"],
                    max_depth=params["rf_max_depth"],
                    min_samples_leaf=params["rf_min_samples_leaf"],
                    random_state=args.seed,
                    n_jobs=-1,
                ),
            ),
            (
                "gbdt",
                GradientBoostingClassifier(
                    n_estimators=params["gbdt_n_estimators"],
                    learning_rate=params["gbdt_learning_rate"],
                    max_depth=params["gbdt_max_depth"],
                    subsample=params["gbdt_subsample"],
                    random_state=args.seed,
                ),
            ),
        ]
        final_estimator = LogisticRegression(
            C=params["meta_c"],
            max_iter=1000,
            solver="lbfgs",
        )
        return TimeSeriesStackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            passthrough=params["passthrough"],
            n_splits=3,
        )

    stack_space = {
        "rf_n_estimators": {"type": "int", "low": 200, "high": 600, "step": 100},
        "rf_max_depth": {"type": "categorical", "choices": [None, 4, 6]},
        "rf_min_samples_leaf": {"type": "int", "low": 10, "high": 50, "step": 10},
        "gbdt_n_estimators": {"type": "int", "low": 150, "high": 400, "step": 50},
        "gbdt_learning_rate": {"type": "float", "low": 0.01, "high": 0.15, "log": True},
        "gbdt_max_depth": {"type": "int", "low": 2, "high": 3, "step": 1},
        "gbdt_subsample": {"type": "float", "low": 0.6, "high": 1.0},
        "meta_c": {"type": "float", "low": 0.1, "high": 10.0, "log": True},
        "passthrough": {"type": "categorical", "choices": [False, True]},
    }

    best_stack, best_stack_thr, best_stack_q, stack_wf_stats, stack_opt_method = _optimize_params_walk_forward(
        name="Stacking",
        model_factory=_stacking_factory,
        search_space=stack_space,
        n_trials=args.bayes_trials,
        startup_trials=args.bayes_startup_trials,
        seed=args.seed,
        X_dev=X_dev[selected],
        y_dev=y_dev,
        next_ret_dev=nr_dev,
        next_date_dev=nd_dev,
        cv_splits=args.walkforward_splits,
        positioning=args.positioning,
        threshold_mode=args.threshold_mode,
        threshold_grid=[args.threshold] if args.threshold_mode == "fixed" else threshold_grid,
        quantile_grid=quantile_grid,
        selection_metric=args.selection_metric,
        cost_per_unit=cost_per_unit,
    )

    stack_best_model = _stacking_factory(best_stack)
    best_stack_stats, best_stack_trades_test = backtest_classifier_test_only(
        name=f"Stacking_best({best_stack})",
        model=stack_best_model,
        X_train=X_dev[selected],
        y_train=y_dev,
        X_test=X_test_s,
        y_test=y_test,
        next_ret_test=nr_test,
        next_date_test=nd_test,
        threshold=best_stack_thr,
        quantile=best_stack_q,
        positioning=args.positioning,
        cost_per_unit=cost_per_unit,
    )
    best_stack_val_ann = stack_wf_stats["ann"]

    assert best_stack is not None and best_stack_stats is not None and best_stack_trades_test is not None

    # Save artifacts.
    feature_path = _safe_path(out_dir / f"{args.feature_selector}_selected_features.txt")
    feature_path.write_text("\n".join(selected), encoding="utf-8")
    lasso_feature_path = _safe_path(out_dir / "lasso_selected_features.txt")
    slasso_feature_path = _safe_path(out_dir / "slasso_selected_features.txt")
    ridge_feature_path = _safe_path(out_dir / "ridge_selected_features.txt")
    lasso_feature_path.write_text("\n".join(selected_lasso), encoding="utf-8")
    slasso_feature_path.write_text("\n".join(selected_slasso), encoding="utf-8")
    ridge_feature_path.write_text("\n".join(selected_ridge), encoding="utf-8")

    # Save test trades.
    rf_trades_path = _safe_path(out_dir / "rf_trades_test.csv")
    mlp_trades_path = _safe_path(out_dir / "mlp_trades_test.csv")
    svm_trades_path = _safe_path(out_dir / "svm_trades_test.csv")
    gbdt_trades_path = _safe_path(out_dir / "gbdt_trades_test.csv")
    bagging_trades_path = _safe_path(out_dir / "bagging_trades_test.csv")
    stacking_trades_path = _safe_path(out_dir / "stacking_trades_test.csv")
    best_rf_trades_test.to_csv(rf_trades_path, index=False, encoding="utf-8-sig")
    best_mlp_trades_test.to_csv(mlp_trades_path, index=False, encoding="utf-8-sig")
    best_svm_trades_test.to_csv(svm_trades_path, index=False, encoding="utf-8-sig")
    best_gbdt_trades_test.to_csv(gbdt_trades_path, index=False, encoding="utf-8-sig")
    best_bag_trades_test.to_csv(bagging_trades_path, index=False, encoding="utf-8-sig")
    best_stack_trades_test.to_csv(stacking_trades_path, index=False, encoding="utf-8-sig")

    # Summary table.
    summary = pd.DataFrame(
        [
            {
                "model": "baseline_long",
                "val_opt_method": "baseline",
                "val_ann": baseline_val["ann"],
                "val_cum": baseline_val["cum"],
                "test_cum": baseline_test["cum"],
                "test_sharpe": baseline_test["sharpe"],
                "test_max_dd": baseline_test["max_dd"],
                "test_turnover_mean": baseline_test.get("turnover_mean"),
                "test_turnover_total": baseline_test.get("turnover_total"),
                "test_cost_total": baseline_test.get("cost_total"),
            },
            {
                "model": f"rf_best_{best_rf}",
                "val_opt_method": "grid_walkforward",
                "val_ann": best_rf_val_ann,
                "val_cum": rf_wf_stats["cum"],
                "test_cum": best_rf_stats["cum"],
                "test_sharpe": best_rf_stats["sharpe"],
                "test_max_dd": best_rf_stats["max_dd"],
                "test_turnover_mean": best_rf_stats.get("turnover_mean"),
                "test_turnover_total": best_rf_stats.get("turnover_total"),
                "test_cost_total": best_rf_stats.get("cost_total"),
                "test_acc": best_rf_stats.get("accuracy"),
                "test_auc": best_rf_stats.get("auc"),
            },
            {
                "model": f"mlp_best_{best_mlp}",
                "val_opt_method": mlp_opt_method,
                "val_ann": best_mlp_val_ann,
                "val_cum": mlp_wf_stats["cum"],
                "test_cum": best_mlp_stats["cum"],
                "test_sharpe": best_mlp_stats["sharpe"],
                "test_max_dd": best_mlp_stats["max_dd"],
                "test_turnover_mean": best_mlp_stats.get("turnover_mean"),
                "test_turnover_total": best_mlp_stats.get("turnover_total"),
                "test_cost_total": best_mlp_stats.get("cost_total"),
                "test_acc": best_mlp_stats.get("accuracy"),
                "test_auc": best_mlp_stats.get("auc"),
            },
            {
                "model": f"svm_best_{best_svm}",
                "val_opt_method": "grid_walkforward",
                "val_ann": best_svm_val_ann,
                "val_cum": svm_wf_stats["cum"],
                "test_cum": best_svm_stats["cum"],
                "test_sharpe": best_svm_stats["sharpe"],
                "test_max_dd": best_svm_stats["max_dd"],
                "test_turnover_mean": best_svm_stats.get("turnover_mean"),
                "test_turnover_total": best_svm_stats.get("turnover_total"),
                "test_cost_total": best_svm_stats.get("cost_total"),
                "test_acc": best_svm_stats.get("accuracy"),
                "test_auc": best_svm_stats.get("auc"),
            },
            {
                "model": f"gbdt_best_{best_gbdt}",
                "val_opt_method": gbdt_opt_method,
                "val_ann": best_gbdt_val_ann,
                "val_cum": gbdt_wf_stats["cum"],
                "test_cum": best_gbdt_stats["cum"],
                "test_sharpe": best_gbdt_stats["sharpe"],
                "test_max_dd": best_gbdt_stats["max_dd"],
                "test_turnover_mean": best_gbdt_stats.get("turnover_mean"),
                "test_turnover_total": best_gbdt_stats.get("turnover_total"),
                "test_cost_total": best_gbdt_stats.get("cost_total"),
                "test_acc": best_gbdt_stats.get("accuracy"),
                "test_auc": best_gbdt_stats.get("auc"),
            },
            {
                "model": f"bagging_best_{best_bag}",
                "val_opt_method": bag_opt_method,
                "val_ann": best_bag_val_ann,
                "val_cum": bag_wf_stats["cum"],
                "test_cum": best_bag_stats["cum"],
                "test_sharpe": best_bag_stats["sharpe"],
                "test_max_dd": best_bag_stats["max_dd"],
                "test_turnover_mean": best_bag_stats.get("turnover_mean"),
                "test_turnover_total": best_bag_stats.get("turnover_total"),
                "test_cost_total": best_bag_stats.get("cost_total"),
                "test_acc": best_bag_stats.get("accuracy"),
                "test_auc": best_bag_stats.get("auc"),
            },
            {
                "model": f"stacking_best_{best_stack}",
                "val_opt_method": stack_opt_method,
                "val_ann": best_stack_val_ann,
                "val_cum": stack_wf_stats["cum"],
                "test_cum": best_stack_stats["cum"],
                "test_sharpe": best_stack_stats["sharpe"],
                "test_max_dd": best_stack_stats["max_dd"],
                "test_turnover_mean": best_stack_stats.get("turnover_mean"),
                "test_turnover_total": best_stack_stats.get("turnover_total"),
                "test_cost_total": best_stack_stats.get("cost_total"),
                "test_acc": best_stack_stats.get("accuracy"),
                "test_auc": best_stack_stats.get("auc"),
            },
        ]
    )
    summary_path = _safe_path(out_dir / "model_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    viz_artifacts = _save_visualizations(
        summary=summary,
        out_dir=out_dir,
        model_trades_test={
            "rf": best_rf_trades_test,
            "mlp": best_mlp_trades_test,
            "svm": best_svm_trades_test,
            "gbdt": best_gbdt_trades_test,
            "bagging": best_bag_trades_test,
            "stacking": best_stack_trades_test,
        },
    )

    LOGGER.info("=== Saved outputs ===")
    LOGGER.info("- %s", summary_path)
    LOGGER.info("- %s", feature_path)
    LOGGER.info("- %s", lasso_feature_path)
    LOGGER.info("- %s", slasso_feature_path)
    LOGGER.info("- %s", ridge_feature_path)
    LOGGER.info("- %s", slasso_detail_path)
    LOGGER.info("- %s", lasso_coef_plot_path)
    LOGGER.info("- %s", ridge_coef_detail_path)
    LOGGER.info("- %s", rf_trades_path)
    LOGGER.info("- %s", mlp_trades_path)
    LOGGER.info("- %s", svm_trades_path)
    LOGGER.info("- %s", gbdt_trades_path)
    LOGGER.info("- %s", bagging_trades_path)
    LOGGER.info("- %s", stacking_trades_path)
    LOGGER.info("- %s", log_path)
    if fs_compare_artifacts:
        for key, path in fs_compare_artifacts.items():
            LOGGER.info("- %s: %s", key, path)
    if viz_artifacts:
        for key, path in viz_artifacts.items():
            LOGGER.info("- %s: %s", key, path)


if __name__ == "__main__":
    main()
