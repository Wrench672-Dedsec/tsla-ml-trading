import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    import yfinance as yf

    HAS_YFINANCE = True
except Exception:
    HAS_YFINANCE = False

from tsla_backtest import (
    TimeSeriesStackingClassifier,
    _optimize_params_walk_forward,
    _parse_float_grid,
    _positions_from_proba,
    _safe_path,
    _setup_logging,
    _time_split_index,
    _walk_forward_select,
    select_features_lasso,
    select_features_ridge,
    select_features_slasso,
)


LOGGER = logging.getLogger("tsla_signal_v4")

AI_NARRATIVE_DEFAULT_PATH = Path("data/tsla_ai_narrative_daily_from_rpa.csv")
AI_NARRATIVE_COLUMNS = ["ai_row_count", "factor_sum", "volume_sum", "sentiment_wavg", "factor_zscore"]
AI_NARRATIVE_ALIAS = "ai_narrative_factor"
AI_NARRATIVE_FEATURES = AI_NARRATIVE_COLUMNS + [AI_NARRATIVE_ALIAS]


def _load_market_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df.columns = [str(c).strip() for c in df.columns]
    df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True).copy()
    if "returns" not in df.columns:
        raise ValueError("Input data must contain a 'returns' column")
    return df


def _load_ai_narrative_daily(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None:
        path = AI_NARRATIVE_DEFAULT_PATH
    if not Path(path).exists():
        return None

    daily = pd.read_csv(path, parse_dates=["date"]) 
    daily.columns = [str(c).strip() for c in daily.columns]
    if "date" not in daily.columns:
        raise ValueError(f"AI narrative file must contain a date column: {path}")

    keep_cols = ["date"] + [c for c in AI_NARRATIVE_COLUMNS if c in daily.columns]
    daily = daily[keep_cols].copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()
    daily = daily.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    return daily


def _merge_ai_narrative_features(df: pd.DataFrame, narrative_path: Optional[Path]) -> pd.DataFrame:
    narrative = _load_ai_narrative_daily(narrative_path)
    if narrative is None or narrative.empty or "date" not in df.columns:
        out = df.copy()
    else:
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
        merged = out.merge(narrative, on="date", how="left", suffixes=("", "__narr"))
        for col in AI_NARRATIVE_COLUMNS:
            narr_col = f"{col}__narr"
            if narr_col in merged.columns:
                if col in merged.columns:
                    merged[col] = merged[col].combine_first(merged[narr_col])
                else:
                    merged[col] = merged[narr_col]
                merged = merged.drop(columns=[narr_col])
        out = merged.sort_values("date").reset_index(drop=True)

    for col in AI_NARRATIVE_COLUMNS:
        if col not in out.columns:
            out[col] = 0.0
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    if AI_NARRATIVE_ALIAS not in out.columns:
        if "factor_zscore" in out.columns:
            out[AI_NARRATIVE_ALIAS] = out["factor_zscore"]
        elif "sentiment_wavg" in out.columns:
            out[AI_NARRATIVE_ALIAS] = out["sentiment_wavg"]
        else:
            out[AI_NARRATIVE_ALIAS] = 0.0

    out[AI_NARRATIVE_ALIAS] = pd.to_numeric(out[AI_NARRATIVE_ALIAS], errors="coerce").fillna(0.0)

    return out


def _download_yfinance_bars(symbol: str, period: str, interval: str) -> pd.DataFrame:
    if not HAS_YFINANCE:
        raise ImportError("yfinance is required for --source yfinance; install it with pip install yfinance")

    raw = yf.download(
        tickers=symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        raise ValueError(f"No data returned for symbol={symbol}, period={period}, interval={interval}")

    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.xs(symbol, axis=1, level=1, drop_level=True)

    raw = raw.reset_index()
    raw.columns = [str(c).strip().lower().replace(" ", "_") for c in raw.columns]
    if "date" not in raw.columns:
        if "datetime" in raw.columns:
            raw = raw.rename(columns={"datetime": "date"})
        else:
            raw = raw.rename(columns={raw.columns[0]: "date"})

    raw["date"] = pd.to_datetime(raw["date"], utc=False).dt.tz_localize(None)
    keep_cols = [c for c in ["date", "open", "high", "low", "close", "adj_close", "volume"] if c in raw.columns]
    return raw[keep_cols].copy()


def _build_ohlcv_features(df: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"OHLCV data is missing required columns: {sorted(missing)}")

    out = df.copy().sort_values("date").reset_index(drop=True)
    open_ = pd.to_numeric(out["open"], errors="coerce")
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    close = pd.to_numeric(out["close"], errors="coerce")
    volume = pd.to_numeric(out["volume"], errors="coerce")

    prev_close = close.shift(1)
    out["returns"] = close.pct_change()
    out["log_return"] = np.log(close.replace(0, np.nan)).diff()
    out["oc_return"] = (close - open_) / open_.replace(0, np.nan)
    out["hl_spread"] = (high - low) / close.replace(0, np.nan)
    out["body_ratio"] = (close - open_) / close.replace(0, np.nan)
    out["upper_wick_ratio"] = (high - np.maximum(open_, close)) / close.replace(0, np.nan)
    out["lower_wick_ratio"] = (np.minimum(open_, close) - low) / close.replace(0, np.nan)
    out["volume_chg"] = volume.pct_change()
    out["volume_z_20"] = (volume - volume.rolling(20).mean()) / (volume.rolling(20).std() + 1e-12)

    for window in [3, 5, 10, 20, 60]:
        out[f"ret_mean_{window}"] = out["returns"].rolling(window).mean()
        out[f"ret_std_{window}"] = out["returns"].rolling(window).std()
        out[f"close_ma_gap_{window}"] = close / close.rolling(window).mean() - 1.0
        out[f"range_ma_gap_{window}"] = (high - low) / (high - low).rolling(window).mean() - 1.0

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    out["ema_12"] = ema12
    out["ema_26"] = ema26
    out["macd"] = macd
    out["macd_signal"] = signal
    out["macd_hist"] = macd - signal

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    out["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    tr_components = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    out["atr_14"] = tr_components.max(axis=1).rolling(14).mean() / close.replace(0, np.nan)
    out["close_rank_20"] = close.rolling(20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)

    return out


def _load_source_frame(args) -> pd.DataFrame:
    if args.source == "csv":
        return _merge_ai_narrative_features(_load_market_data(Path(args.data)), Path(args.narrative_data))

    bars = _download_yfinance_bars(args.symbol, args.period, args.interval)
    return _merge_ai_narrative_features(_build_ohlcv_features(bars), Path(args.narrative_data))


def _prepare_frames(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.DataFrame, pd.Timestamp]:
    df = df.assign(
        next_return=df["returns"].shift(-1),
        next_date=df["date"].shift(-1),
    ).copy()

    numeric = df.select_dtypes(include=[np.number]).copy()
    feature_cols = [c for c in numeric.columns if c not in {"next_return", "y_up"}]
    X_all = numeric[feature_cols].dropna(axis=1, how="all")
    X_all.index = df["date"]

    labeled_mask = df["next_return"].notna().to_numpy()
    X_labeled = X_all.iloc[labeled_mask].copy()
    y_labeled = (df.loc[labeled_mask, "next_return"] > 0).astype(int)
    next_ret_labeled = df.loc[labeled_mask, "next_return"].astype(float)
    next_date_labeled = df.loc[labeled_mask, "next_date"].copy()

    X_live = X_all.iloc[[-1]].copy()
    asof_date = pd.to_datetime(df["date"].iloc[-1])
    return X_labeled, y_labeled, next_ret_labeled, next_date_labeled, X_live, asof_date


def _signal_text(positioning: str, position_value: float) -> str:
    if positioning == "long_short":
        return "LONG" if position_value > 0 else "SHORT"
    return "LONG" if position_value > 0 else "FLAT"


def _model_configs(seed: int) -> Dict[str, Dict]:
    mlp_hidden_map = {
        "32": (32,),
        "64": (64,),
        "64_32": (64, 32),
        "128_64": (128, 64),
        "128_64_32": (128, 64, 32),
    }

    def _rf_factory(params: Dict) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            random_state=seed,
            n_jobs=-1,
        )

    def _mlp_factory(params: Dict) -> MLPClassifier:
        return MLPClassifier(
            hidden_layer_sizes=mlp_hidden_map[params["hidden_layer_key"]],
            alpha=params["alpha"],
            activation=params["activation"],
            learning_rate_init=params["learning_rate_init"],
            batch_size=params["batch_size"],
            random_state=seed,
            max_iter=800,
            early_stopping=False,
            n_iter_no_change=12,
        )

    def _svm_factory(params: Dict) -> SVC:
        return SVC(
            kernel=params["kernel"],
            C=params["C"],
            gamma=params.get("gamma", "scale"),
            probability=True,
            random_state=seed,
        )

    def _gbdt_factory(params: Dict) -> GradientBoostingClassifier:
        return GradientBoostingClassifier(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            subsample=params["subsample"],
            random_state=seed,
        )

    def _bagging_factory(params: Dict) -> BaggingClassifier:
        tree = DecisionTreeClassifier(
            max_depth=params["tree_max_depth"],
            min_samples_leaf=params["tree_min_samples_leaf"],
            random_state=seed,
        )
        return BaggingClassifier(
            estimator=tree,
            n_estimators=params["n_estimators"],
            max_samples=params["max_samples"],
            max_features=params["max_features"],
            bootstrap=True,
            random_state=seed,
            n_jobs=-1,
        )

    def _stacking_factory(params: Dict):
        estimators = [
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=params["rf_n_estimators"],
                    max_depth=params["rf_max_depth"],
                    min_samples_leaf=params["rf_min_samples_leaf"],
                    random_state=seed,
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
                    random_state=seed,
                ),
            ),
        ]
        final_estimator = LogisticRegression(C=params["meta_c"], max_iter=1000, solver="lbfgs")
        return TimeSeriesStackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            passthrough=params["passthrough"],
            n_splits=3,
        )

    return {
        "rf": {
            "factory": _rf_factory,
            "grid": [
                {"n_estimators": 500, "max_depth": 4, "min_samples_leaf": 50},
                {"n_estimators": 800, "max_depth": 6, "min_samples_leaf": 30},
                {"n_estimators": 800, "max_depth": None, "min_samples_leaf": 20},
            ],
            "mode": "grid",
        },
        "mlp": {
            "factory": _mlp_factory,
            "space": {
                "hidden_layer_key": {"type": "categorical", "choices": ["32", "64", "64_32", "128_64", "128_64_32"]},
                "alpha": {"type": "float", "low": 1e-6, "high": 1e-2, "log": True},
                "learning_rate_init": {"type": "float", "low": 1e-4, "high": 5e-2, "log": True},
                "batch_size": {"type": "categorical", "choices": [32, 64, 128, 256]},
                "activation": {"type": "categorical", "choices": ["relu", "tanh"]},
            },
            "mode": "bayes",
        },
        "svm": {
            "factory": _svm_factory,
            "grid": [
                {"kernel": "rbf", "C": 1.0, "gamma": "scale"},
                {"kernel": "rbf", "C": 3.0, "gamma": "scale"},
                {"kernel": "linear", "C": 1.0},
            ],
            "mode": "grid",
        },
        "gbdt": {
            "factory": _gbdt_factory,
            "space": {
                "n_estimators": {"type": "int", "low": 150, "high": 600, "step": 50},
                "learning_rate": {"type": "float", "low": 0.01, "high": 0.2, "log": True},
                "max_depth": {"type": "int", "low": 2, "high": 4, "step": 1},
                "subsample": {"type": "float", "low": 0.6, "high": 1.0},
            },
            "mode": "bayes",
        },
        "bagging": {
            "factory": _bagging_factory,
            "space": {
                "n_estimators": {"type": "int", "low": 100, "high": 500, "step": 50},
                "max_samples": {"type": "float", "low": 0.5, "high": 1.0},
                "max_features": {"type": "float", "low": 0.4, "high": 1.0},
                "tree_max_depth": {"type": "categorical", "choices": [None, 3, 5, 8]},
                "tree_min_samples_leaf": {"type": "int", "low": 5, "high": 60, "step": 5},
            },
            "mode": "bayes",
        },
        "stacking": {
            "factory": _stacking_factory,
            "space": {
                "rf_n_estimators": {"type": "int", "low": 200, "high": 600, "step": 100},
                "rf_max_depth": {"type": "categorical", "choices": [None, 4, 6]},
                "rf_min_samples_leaf": {"type": "int", "low": 10, "high": 50, "step": 10},
                "gbdt_n_estimators": {"type": "int", "low": 150, "high": 400, "step": 50},
                "gbdt_learning_rate": {"type": "float", "low": 0.01, "high": 0.15, "log": True},
                "gbdt_max_depth": {"type": "int", "low": 2, "high": 3, "step": 1},
                "gbdt_subsample": {"type": "float", "low": 0.6, "high": 1.0},
                "meta_c": {"type": "float", "low": 0.1, "high": 10.0, "log": True},
                "passthrough": {"type": "categorical", "choices": [False, True]},
            },
            "mode": "bayes",
        },
    }


def _fit_signal_candidate(
    *,
    model_name: str,
    model_factory,
    selector: str,
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    next_ret_dev: pd.Series,
    next_date_dev: pd.Series,
    X_live: pd.DataFrame,
    args,
    threshold_grid: List[float],
    quantile_grid: List[float],
    cost_per_unit: float,
) -> Dict:
    if selector == "grid":
        best_params, best_thr, best_q, wf_stats = _walk_forward_select(
            name=model_name,
            model_factory=model_factory,
            param_grid=args.param_grid,
            X_dev=X_dev,
            y_dev=y_dev,
            next_ret_dev=next_ret_dev,
            next_date_dev=next_date_dev,
            cv_splits=args.walkforward_splits,
            positioning=args.positioning,
            threshold_mode=args.threshold_mode,
            threshold_grid=[args.threshold] if args.threshold_mode == "fixed" else threshold_grid,
            quantile_grid=quantile_grid,
            selection_metric=args.selection_metric,
            cost_per_unit=cost_per_unit,
            seed=args.seed,
        )
    else:
        best_params, best_thr, best_q, wf_stats, _ = _optimize_params_walk_forward(
            name=model_name,
            model_factory=model_factory,
            search_space=args.search_space,
            n_trials=args.bayes_trials,
            startup_trials=args.bayes_startup_trials,
            seed=args.seed,
            X_dev=X_dev,
            y_dev=y_dev,
            next_ret_dev=next_ret_dev,
            next_date_dev=next_date_dev,
            cv_splits=args.walkforward_splits,
            positioning=args.positioning,
            threshold_mode=args.threshold_mode,
            threshold_grid=[args.threshold] if args.threshold_mode == "fixed" else threshold_grid,
            quantile_grid=quantile_grid,
            selection_metric=args.selection_metric,
            cost_per_unit=cost_per_unit,
        )

    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("clf", model_factory(best_params)),
        ]
    )
    pipe.fit(X_dev, y_dev)

    if hasattr(pipe, "predict_proba"):
        p_live = pipe.predict_proba(X_live)[:, 1]
    else:
        p_live = pipe.predict(X_live).astype(float)

    pos_live = _positions_from_proba(
        p_live,
        positioning=args.positioning,
        threshold=best_thr,
        quantile=best_q,
    )
    position_value = float(pos_live[0])
    prob_up = float(p_live[0])

    return {
        "model": model_name,
        "params": best_params,
        "threshold": best_thr,
        "quantile": best_q,
        "val_ann": float(wf_stats["ann"]),
        "val_cum": float(wf_stats["cum"]),
        "val_sharpe": float(wf_stats["sharpe"]),
        "prob_up": prob_up,
        "position": position_value,
        "signal": _signal_text(args.positioning, position_value),
        "pipeline": pipe,
    }


def _build_training_context(args, out_dir: Path) -> Dict:
    df = _load_source_frame(args)
    X_all, y_all, next_ret_all, next_date_all, X_live, asof_date = _prepare_frames(df)
    if len(X_all) < 50:
        raise ValueError("Not enough labeled rows to build a signal model")

    n = len(X_all)
    tr_idx, va_idx, _ = _time_split_index(n, args.train_frac, args.val_frac)
    X_train, X_val = X_all.iloc[tr_idx], X_all.iloc[va_idx]
    y_train, y_val = y_all.iloc[tr_idx], y_all.iloc[va_idx]
    nr_train, nr_val = next_ret_all.iloc[tr_idx], next_ret_all.iloc[va_idx]
    nd_train, nd_val = next_date_all.iloc[tr_idx], next_date_all.iloc[va_idx]

    X_dev = pd.concat([X_train, X_val], axis=0)
    y_dev = pd.concat([y_train, y_val], axis=0)
    nr_dev = pd.concat([nr_train, nr_val], axis=0)
    nd_dev = pd.concat([nd_train, nd_val], axis=0)

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
        lasso_plot_path=None,
    )
    selected_ridge = select_features_ridge(
        X_train,
        nr_train,
        X_val,
        nr_val,
        max_features=args.ridge_max_features,
        min_selected_features=args.ridge_min_features,
        min_coef_ratio=args.ridge_min_coef_ratio,
        cv_splits=args.ridge_cv_splits,
        coef_detail_path=None,
        random_state=args.seed,
    )
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
        detail_path=None,
        random_state=args.seed,
    )

    selected_map = {
        "lasso": selected_lasso,
        "slasso": selected_slasso,
        "ridge": selected_ridge,
    }
    selected = selected_map[args.feature_selector]
    if AI_NARRATIVE_ALIAS in X_train.columns and AI_NARRATIVE_ALIAS not in selected:
        selected = list(dict.fromkeys([*selected, AI_NARRATIVE_ALIAS]))
    if not selected:
        raise RuntimeError(f"{args.feature_selector.upper()} selected 0 features")

    (out_dir / f"{args.feature_selector}_selected_features.txt").write_text("\n".join(selected), encoding="utf-8")
    (out_dir / "lasso_selected_features.txt").write_text("\n".join(selected_lasso), encoding="utf-8")
    (out_dir / "slasso_selected_features.txt").write_text("\n".join(selected_slasso), encoding="utf-8")
    (out_dir / "ridge_selected_features.txt").write_text("\n".join(selected_ridge), encoding="utf-8")

    return {
        "asof_date": asof_date,
        "selected": selected,
        "X_dev": X_dev[selected],
        "y_dev": y_dev,
        "nr_dev": nr_dev,
        "nd_dev": nd_dev,
        "X_live": X_live[selected],
    }


def _fit_candidates_on_context(args, ctx: Dict) -> List[Dict]:
    threshold_grid = _parse_float_grid(args.threshold_grid) or [args.threshold]
    quantile_grid = _parse_float_grid(args.quantile_grid) or [0.8]
    cost_per_unit = float(args.cost_bps) / 10000.0

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    configs = _model_configs(args.seed)

    candidate_rows: List[Dict] = []
    for model_name in model_names:
        if model_name not in configs:
            raise ValueError(f"Unknown model: {model_name}")
        cfg = configs[model_name]
        local_args = argparse.Namespace(**vars(args))
        if cfg["mode"] == "grid":
            local_args.param_grid = cfg["grid"]
            result = _fit_signal_candidate(
                model_name=model_name,
                model_factory=cfg["factory"],
                selector="grid",
                X_dev=ctx["X_dev"],
                y_dev=ctx["y_dev"],
                next_ret_dev=ctx["nr_dev"],
                next_date_dev=ctx["nd_dev"],
                X_live=ctx["X_live"],
                args=local_args,
                threshold_grid=threshold_grid,
                quantile_grid=quantile_grid,
                cost_per_unit=cost_per_unit,
            )
        else:
            local_args.search_space = cfg["space"]
            result = _fit_signal_candidate(
                model_name=model_name,
                model_factory=cfg["factory"],
                selector="bayes",
                X_dev=ctx["X_dev"],
                y_dev=ctx["y_dev"],
                next_ret_dev=ctx["nr_dev"],
                next_date_dev=ctx["nd_dev"],
                X_live=ctx["X_live"],
                args=local_args,
                threshold_grid=threshold_grid,
                quantile_grid=quantile_grid,
                cost_per_unit=cost_per_unit,
            )
        candidate_rows.append(result)
    return candidate_rows


def _artifact_paths(args, out_dir: Path) -> Tuple[Path, Path]:
    artifact_dir = Path(args.artifact_dir) if str(args.artifact_dir).strip() else (out_dir / "artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = _safe_path(artifact_dir / f"{args.artifact_prefix}_pipeline.joblib")
    metadata_path = _safe_path(artifact_dir / f"{args.artifact_prefix}_metadata.json")
    return model_path, metadata_path


def _save_training_artifacts(args, out_dir: Path, best: Dict, selected: List[str], asof_date: pd.Timestamp, summary_path: Path, signal_path: Path) -> Dict:
    model_path, metadata_path = _artifact_paths(args, out_dir)
    joblib.dump(best["pipeline"], model_path)

    metadata = {
        "trained_asof_date": str(asof_date),
        "feature_selector": args.feature_selector,
        "selected_features": selected,
        "best_model": best["model"],
        "best_params": best["params"],
        "threshold": best["threshold"],
        "quantile": best["quantile"],
        "positioning": args.positioning,
        "source": args.source,
        "symbol": args.symbol,
        "period": args.period,
        "interval": args.interval,
        "summary_csv": str(summary_path),
        "signal_csv": str(signal_path),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")
    return {
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
    }


def _run_train_cycle(args, out_dir: Path, log_path: Path) -> Dict:
    ctx = _build_training_context(args, out_dir)
    candidate_rows = _fit_candidates_on_context(args, ctx)

    summary = pd.DataFrame(candidate_rows).sort_values(["val_ann", "val_sharpe"], ascending=[False, False]).reset_index(drop=True)
    summary_path = _safe_path(out_dir / "signal_model_summary.csv")
    summary.drop(columns=["pipeline"]).to_csv(summary_path, index=False, encoding="utf-8-sig")

    best = summary.iloc[0].to_dict()
    signal_row = pd.DataFrame(
        [
            {
                "asof_date": ctx["asof_date"],
                "signal_for": "next_bar",
                "model": best["model"],
                "signal": best["signal"],
                "position": best["position"],
                "prob_up": best["prob_up"],
                "threshold": best["threshold"],
                "quantile": best["quantile"],
                "selected_feature_count": len(ctx["selected"]),
            }
        ]
    )
    signal_path = _safe_path(out_dir / "latest_signal.csv")
    signal_row.to_csv(signal_path, index=False, encoding="utf-8-sig")

    artifacts = _save_training_artifacts(
        args,
        out_dir,
        best,
        ctx["selected"],
        ctx["asof_date"],
        summary_path,
        signal_path,
    )

    report = {
        "mode": "train",
        "asof_date": str(ctx["asof_date"]),
        "best_model": best["model"],
        "signal": best["signal"],
        "position": best["position"],
        "prob_up": best["prob_up"],
        "threshold": best["threshold"],
        "quantile": best["quantile"],
        "selected_feature_count": len(ctx["selected"]),
        "selected_features": ctx["selected"],
        "summary_csv": str(summary_path),
        "signal_csv": str(signal_path),
        "log_file": str(log_path),
        "source": args.source,
        **artifacts,
    }
    report_path = _safe_path(out_dir / "signal_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")

    LOGGER.info("Train saved model=%s, metadata=%s", artifacts["model_path"], artifacts["metadata_path"])
    LOGGER.info("Best model=%s signal=%s prob_up=%.4f position=%.1f", best["model"], best["signal"], float(best["prob_up"]), float(best["position"]))
    LOGGER.info("Saved summary: %s", summary_path)
    LOGGER.info("Saved signal: %s", signal_path)
    LOGGER.info("Saved report: %s", report_path)
    return report


def _run_infer_cycle(args, out_dir: Path, log_path: Path) -> Dict:
    model_path, metadata_path = _artifact_paths(args, out_dir)
    if not Path(model_path).exists() or not Path(metadata_path).exists():
        raise FileNotFoundError(
            f"Inference artifacts not found. expected_model={model_path}, expected_metadata={metadata_path}. Run --mode train first."
        )

    pipe = joblib.load(model_path)
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    selected = list(metadata.get("selected_features", []))
    if not selected:
        raise ValueError("Metadata has no selected_features")

    df = _load_source_frame(args)
    _, _, _, _, X_live, asof_date = _prepare_frames(df)
    missing = [c for c in selected if c not in X_live.columns]
    if missing:
        LOGGER.warning("Live frame missing %d trained features; they will be filled with NaN", len(missing))
    X_live_s = X_live.reindex(columns=selected)

    if hasattr(pipe, "predict_proba"):
        p_live = pipe.predict_proba(X_live_s)[:, 1]
    else:
        p_live = pipe.predict(X_live_s).astype(float)

    threshold = metadata.get("threshold")
    quantile = metadata.get("quantile")
    positioning = str(metadata.get("positioning", args.positioning))
    pos_live = _positions_from_proba(
        p_live,
        positioning=positioning,
        threshold=threshold,
        quantile=quantile,
    )

    signal = _signal_text(positioning, float(pos_live[0]))
    signal_row = pd.DataFrame(
        [
            {
                "asof_date": asof_date,
                "signal_for": "next_bar",
                "model": metadata.get("best_model", "loaded_model"),
                "signal": signal,
                "position": float(pos_live[0]),
                "prob_up": float(p_live[0]),
                "threshold": threshold,
                "quantile": quantile,
                "selected_feature_count": len(selected),
            }
        ]
    )
    signal_path = _safe_path(out_dir / "latest_signal.csv")
    signal_row.to_csv(signal_path, index=False, encoding="utf-8-sig")

    report = {
        "mode": "infer",
        "asof_date": str(asof_date),
        "best_model": metadata.get("best_model", "loaded_model"),
        "signal": signal,
        "position": float(pos_live[0]),
        "prob_up": float(p_live[0]),
        "threshold": threshold,
        "quantile": quantile,
        "selected_feature_count": len(selected),
        "selected_features": selected,
        "signal_csv": str(signal_path),
        "log_file": str(log_path),
        "source": args.source,
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
    }
    report_path = _safe_path(out_dir / "signal_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")

    LOGGER.info("Infer loaded model=%s", model_path)
    LOGGER.info("Signal=%s prob_up=%.4f position=%.1f", signal, float(p_live[0]), float(pos_live[0]))
    LOGGER.info("Saved signal: %s", signal_path)
    LOGGER.info("Saved report: %s", report_path)
    return report


def _run_signal_cycle(args, out_dir: Path, log_path: Path) -> Dict:
    ctx = _build_training_context(args, out_dir)
    candidate_rows = _fit_candidates_on_context(args, ctx)

    summary = pd.DataFrame(candidate_rows).sort_values(["val_ann", "val_sharpe"], ascending=[False, False]).reset_index(drop=True)
    summary_path = _safe_path(out_dir / "signal_model_summary.csv")
    summary.drop(columns=["pipeline"]).to_csv(summary_path, index=False, encoding="utf-8-sig")

    best = summary.iloc[0].to_dict()
    signal_row = pd.DataFrame(
        [
            {
                "asof_date": ctx["asof_date"],
                "signal_for": "next_bar",
                "model": best["model"],
                "signal": best["signal"],
                "position": best["position"],
                "prob_up": best["prob_up"],
                "threshold": best["threshold"],
                "quantile": best["quantile"],
                "selected_feature_count": len(ctx["selected"]),
            }
        ]
    )
    signal_path = _safe_path(out_dir / "latest_signal.csv")
    signal_row.to_csv(signal_path, index=False, encoding="utf-8-sig")

    report = {
        "mode": "retrain",
        "asof_date": str(ctx["asof_date"]),
        "best_model": best["model"],
        "signal": best["signal"],
        "position": best["position"],
        "prob_up": best["prob_up"],
        "threshold": best["threshold"],
        "quantile": best["quantile"],
        "selected_feature_count": len(ctx["selected"]),
        "selected_features": ctx["selected"],
        "summary_csv": str(summary_path),
        "signal_csv": str(signal_path),
        "log_file": str(log_path),
        "source": args.source,
    }
    report_path = _safe_path(out_dir / "signal_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")

    LOGGER.info("Best model=%s signal=%s prob_up=%.4f position=%.1f", best["model"], best["signal"], float(best["prob_up"]), float(best["position"]))
    LOGGER.info("Saved summary: %s", summary_path)
    LOGGER.info("Saved signal: %s", signal_path)
    LOGGER.info("Saved report: %s", report_path)

    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="TSLA read-only signal version 4: load market data and output signal only")
    ap.add_argument("--source", choices=["csv", "yfinance"], default="csv", help="Market data source")
    ap.add_argument("--mode", choices=["retrain", "train", "infer"], default="retrain", help="retrain=fit every run, train=fit and save artifacts, infer=load artifacts only")
    ap.add_argument("--data", default="data/TSLA_Merged_Final.csv")
    ap.add_argument("--narrative-data", default="data/tsla_ai_narrative_daily_from_rpa.csv")
    ap.add_argument("--symbol", default="TSLA", help="Ticker symbol used when --source=yfinance")
    ap.add_argument("--period", default="2y", help="History window for --source=yfinance")
    ap.add_argument("--interval", default="1d", help="Bar interval for --source=yfinance")
    ap.add_argument("--poll-seconds", type=int, default=0, help="If > 0, rerun signal generation every N seconds")
    ap.add_argument("--once", action="store_true", help="Run one cycle and exit even if --poll-seconds is set")
    ap.add_argument("--out", default="outputs/signal")
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--threshold-mode", choices=["fixed", "auto"], default="auto")
    ap.add_argument("--threshold-grid", default="0.4,0.45,0.5,0.55,0.6")
    ap.add_argument("--quantile-grid", default="0.6,0.7,0.8,0.9")
    ap.add_argument("--walkforward-splits", type=int, default=5)
    ap.add_argument("--selection-metric", choices=["cum", "sharpe", "ann"], default="ann")
    ap.add_argument("--bayes-trials", type=int, default=12)
    ap.add_argument("--bayes-startup-trials", type=int, default=4)
    ap.add_argument("--positioning", choices=["long_short", "long_flat", "long_flat_quantile"], default="long_flat")
    ap.add_argument("--cost-bps", type=float, default=0.0)
    ap.add_argument("--max-features", type=int, default=120)
    ap.add_argument("--lasso-min-features", type=int, default=30)
    ap.add_argument("--lasso-cv-splits", type=int, default=5)
    ap.add_argument("--lasso-n-alphas", type=int, default=60)
    ap.add_argument("--ridge-cv-splits", type=int, default=5)
    ap.add_argument("--ridge-max-features", type=int, default=60)
    ap.add_argument("--ridge-min-features", type=int, default=20)
    ap.add_argument("--ridge-min-coef-ratio", type=float, default=0.10)
    ap.add_argument("--feature-selector", choices=["lasso", "slasso", "ridge"], default="lasso")
    ap.add_argument("--slasso-min-features", type=int, default=30)
    ap.add_argument("--slasso-cv-splits", type=int, default=5)
    ap.add_argument("--slasso-n-alphas", type=int, default=60)
    ap.add_argument("--slasso-bootstrap", type=int, default=80)
    ap.add_argument("--slasso-sample-frac", type=float, default=0.75)
    ap.add_argument("--slasso-threshold", type=float, default=0.60)
    ap.add_argument("--models", default="rf,mlp,svm,gbdt,bagging,stacking", help="Comma-separated model candidates")
    ap.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    ap.add_argument("--log-file", default="signal_v4.log")
    ap.add_argument("--artifact-dir", default="", help="Directory to save/load train artifacts. default: <out>/artifacts")
    ap.add_argument("--artifact-prefix", default="signal_model", help="Prefix for saved model/metadata artifact files")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.source == "yfinance" and not HAS_YFINANCE:
        raise ImportError("yfinance is required for --source yfinance")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = _setup_logging(out_dir, level=args.log_level, log_file=args.log_file)
    LOGGER.info("Run started with args: %s", vars(args))

    def _run_once() -> Dict:
        if args.mode == "retrain":
            return _run_signal_cycle(args, out_dir, log_path)
        if args.mode == "train":
            return _run_train_cycle(args, out_dir, log_path)
        return _run_infer_cycle(args, out_dir, log_path)

    if args.once or args.poll_seconds <= 0:
        _run_once()
        return

    LOGGER.info("Polling enabled: source=%s interval=%s poll_seconds=%d", args.source, args.interval, args.poll_seconds)
    while True:
        try:
            _run_once()
        except Exception:
            LOGGER.exception("Signal cycle failed")
        time.sleep(max(1, int(args.poll_seconds)))


if __name__ == "__main__":
    main()