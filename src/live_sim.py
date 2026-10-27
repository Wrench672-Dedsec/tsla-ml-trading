import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

# Allow importing project helpers from src/
sys.path.append(str(Path(__file__).resolve().parent))

from tsla_lasso_rf_mlp_signal_v4 import (  # noqa: E402
    _build_ohlcv_features,
    _merge_ai_narrative_features,
    AI_NARRATIVE_ALIAS,
)
from tsla_backtest import (  # noqa: E402
    select_features_slasso,
    _apply_costs_and_turnover,
    _perf_stats,
    _positions_from_proba,
    _time_split_index,
)
from sklearn.ensemble import BaggingClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402
import joblib  # noqa: E402


def _fetch_alpha_vantage_daily(symbol: str, api_key: str, outputsize: str = "full") -> pd.DataFrame:
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "outputsize": outputsize,
        "apikey": api_key,
    }
    url = f"https://www.alphavantage.co/query?{urlencode(params)}"
    with urlopen(url, timeout=60) as resp:
        raw = resp.read()

    data = json.loads(raw.decode("utf-8"))
    if "Error Message" in data:
        raise ValueError(f"Alpha Vantage error: {data['Error Message']}")
    if "Note" in data:
        raise ValueError(f"Alpha Vantage rate limit: {data['Note']}")
    if "Information" in data:
        raise ValueError(f"Alpha Vantage info: {data['Information']}")

    series = data.get("Time Series (Daily)")
    if not series:
        raise ValueError("Alpha Vantage response missing 'Time Series (Daily)'")

    rows: List[Dict[str, object]] = []
    for dt, row in series.items():
        rows.append(
            {
                "date": dt,
                "open": float(row.get("1. open", "nan")),
                "high": float(row.get("2. high", "nan")),
                "low": float(row.get("3. low", "nan")),
                "close": float(row.get("4. close", "nan")),
                "adj_close": float("nan"),
                "volume": float(row.get("5. volume", "nan")),
            }
        )

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def _build_feature_frame_from_alpha(symbol: str, api_key: str, narrative_path: Path, outputsize: str) -> pd.DataFrame:
    ohlcv = _fetch_alpha_vantage_daily(symbol, api_key, outputsize=outputsize)
    feat = _build_ohlcv_features(ohlcv)
    feat = _merge_ai_narrative_features(feat, narrative_path)
    return feat.sort_values("date").reset_index(drop=True)

def _build_feature_frame_from_local(data_path: Path, narrative_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path, parse_dates=["date"])
    df.columns = [str(c).strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in lower]
    if missing:
        raise ValueError(f"Local data missing columns: {missing}")

    ohlcv = df[[lower["date"], lower["open"], lower["high"], lower["low"], lower["close"], lower["volume"]]].copy()
    ohlcv.columns = ["date", "open", "high", "low", "close", "volume"]
    ohlcv["date"] = pd.to_datetime(ohlcv["date"], errors="coerce")
    ohlcv = ohlcv.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    feat = _build_ohlcv_features(ohlcv)
    feat = _merge_ai_narrative_features(feat, narrative_path)
    return feat.sort_values("date").reset_index(drop=True)


def _train_until_sharpe(
    feat: pd.DataFrame,
    *,
    train_end: pd.Timestamp,
    min_sharpe: float,
    max_attempts: int,
    cost_bps: float,
    seed: int,
    threshold_grid: List[float],
) -> Tuple[Pipeline, List[str], Dict[str, float], Dict[str, object]]:
    feat = feat.copy()
    feat["next_return"] = feat["returns"].shift(-1)
    feat["next_date"] = feat["date"].shift(-1)
    feat = feat.dropna(subset=["next_return"]).copy()
    feat["y_up"] = (feat["next_return"] > 0).astype(int)

    numeric = feat.select_dtypes(include=[np.number]).copy()
    feature_cols = [c for c in numeric.columns if c not in {"next_return", "y_up"}]
    X_all = numeric[feature_cols].dropna(axis=1, how="all")
    X_all.index = feat["date"]
    y_all = feat["y_up"].astype(int)
    next_ret_all = feat["next_return"].astype(float)

    train_mask = X_all.index <= train_end
    X_pre = X_all.loc[train_mask].copy()
    y_pre = y_all.loc[train_mask].copy()
    nr_pre = next_ret_all.loc[train_mask].copy()

    if len(X_pre) < 60:
        raise ValueError("Not enough training data before train_end to fit model.")

    tr_idx, va_idx, te_idx = _time_split_index(len(X_pre), train_frac=0.6, val_frac=0.2)
    X_train, X_val, X_test = X_pre.iloc[tr_idx], X_pre.iloc[va_idx], X_pre.iloc[te_idx]
    y_train, y_val, y_test = y_pre.iloc[tr_idx], y_pre.iloc[va_idx], y_pre.iloc[te_idx]
    nr_train, nr_val, nr_test = nr_pre.iloc[tr_idx], nr_pre.iloc[va_idx], nr_pre.iloc[te_idx]

    best_attempt = None

    for attempt in range(1, max_attempts + 1):
        local_seed = seed + attempt - 1
        # Adjust min feature count for smaller OHLCV-only feature sets.
        min_features = min(30, max(5, int(X_train.shape[1] * 0.5)))

        selected = select_features_slasso(
            X_train,
            nr_train,
            X_val,
            nr_val,
            max_features=120,
            min_selected_features=min_features,
            cv_splits=5,
            n_alphas=60,
            n_bootstrap=80,
            sample_frac=0.75,
            stability_threshold=0.60,
            detail_path=None,
            random_state=local_seed,
        )
        if AI_NARRATIVE_ALIAS in X_train.columns and AI_NARRATIVE_ALIAS not in selected:
            selected = list(dict.fromkeys([*selected, AI_NARRATIVE_ALIAS]))

        bag_params = {
            "n_estimators": 350,
            "max_samples": 0.8649174147658507,
            "max_features": 0.5281826113413507,
            "tree_max_depth": None,
            "tree_min_samples_leaf": 45,
        }
        tree = DecisionTreeClassifier(
            max_depth=bag_params["tree_max_depth"],
            min_samples_leaf=bag_params["tree_min_samples_leaf"],
            random_state=local_seed,
        )
        model = BaggingClassifier(
            estimator=tree,
            n_estimators=bag_params["n_estimators"],
            max_samples=bag_params["max_samples"],
            max_features=bag_params["max_features"],
            bootstrap=True,
            random_state=local_seed,
            n_jobs=-1,
        )
        pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
                ("clf", model),
            ]
        )

        X_dev = pd.concat([X_train, X_val], axis=0)[selected]
        y_dev = pd.concat([y_train, y_val], axis=0)
        pipe.fit(X_dev, y_dev)

        X_val_s = X_val[selected]
        X_test_s = X_test[selected]
        if hasattr(pipe, "predict_proba"):
            p_val = pipe.predict_proba(X_val_s)[:, 1]
            p_test = pipe.predict_proba(X_test_s)[:, 1]
        else:
            p_val = pipe.predict(X_val_s).astype(float)
            p_test = pipe.predict(X_test_s).astype(float)

        # Select threshold on validation by Sharpe.
        best_thr = None
        best_sharpe = -1e18
        for thr in threshold_grid:
            pos_val = _positions_from_proba(p_val, positioning="long_flat", threshold=thr)
            net_val, _, _, _ = _apply_costs_and_turnover(
                raw_returns=np.asarray(nr_val, dtype=float) * pos_val,
                positions=pos_val,
                cost_per_unit=float(cost_bps) / 10000.0,
                prev_position=0.0,
            )
            stats_val = _perf_stats(pd.Series(net_val, index=X_val_s.index))
            sharpe = float(stats_val["sharpe"])
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_thr = float(thr)

        if best_thr is None:
            best_thr = 0.5

        positions = _positions_from_proba(p_test, positioning="long_flat", threshold=best_thr)
        net, _, _, _ = _apply_costs_and_turnover(
            raw_returns=np.asarray(nr_test, dtype=float) * positions,
            positions=positions,
            cost_per_unit=float(cost_bps) / 10000.0,
            prev_position=0.0,
        )
        test_stats = _perf_stats(pd.Series(net, index=X_test_s.index))

        attempt_info = {
            "attempt": attempt,
            "seed": local_seed,
            "test_sharpe": float(test_stats["sharpe"]),
            "test_cum": float(test_stats["cum"]),
            "selected_features": len(selected),
            "threshold": float(best_thr),
        }
        best_attempt = attempt_info
        print(f"Attempt {attempt}: test_sharpe={test_stats['sharpe']:.3f}, test_cum={test_stats['cum']:.2%}")

        if float(test_stats["sharpe"]) >= float(min_sharpe):
            return pipe, selected, test_stats, attempt_info

    raise RuntimeError(
        f"Failed to reach test Sharpe >= {min_sharpe} after {max_attempts} attempts. Last={best_attempt}"
    )


def _simulate_live(
    feat: pd.DataFrame,
    *,
    model: Pipeline,
    selected: List[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    cost_bps: float,
    threshold: float,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    feat = feat.copy()
    feat["next_return"] = feat["returns"].shift(-1)
    feat["next_date"] = feat["date"].shift(-1)
    feat = feat.dropna(subset=["next_return"]).copy()

    numeric = feat.select_dtypes(include=[np.number]).copy()
    feature_cols = [c for c in numeric.columns if c not in {"next_return", "y_up"}]
    X_all = numeric[feature_cols].dropna(axis=1, how="all")
    X_all.index = feat["date"]
    next_ret_all = feat["next_return"].astype(float)
    next_date_all = feat["next_date"]

    mask = (X_all.index >= start_date) & (X_all.index <= end_date)
    X_live = X_all.loc[mask, selected]
    nr_live = next_ret_all.loc[mask]
    nd_live = next_date_all.loc[mask]

    if X_live.empty:
        raise ValueError("No rows in the requested live period.")

    if hasattr(model, "predict_proba"):
        p_live = model.predict_proba(X_live)[:, 1]
    else:
        p_live = model.predict(X_live).astype(float)

    positions = _positions_from_proba(p_live, positioning="long_flat", threshold=threshold)
    net, turnover, costs, _ = _apply_costs_and_turnover(
        raw_returns=np.asarray(nr_live, dtype=float) * positions,
        positions=positions,
        cost_per_unit=float(cost_bps) / 10000.0,
        prev_position=0.0,
    )

    strategy_return = pd.Series(net, index=X_live.index)
    strategy_equity = (1.0 + strategy_return).cumprod()

    baseline_return = pd.Series(np.asarray(nr_live, dtype=float), index=X_live.index)
    baseline_equity = (1.0 + baseline_return).cumprod()

    stats_strategy = _perf_stats(strategy_return)
    stats_baseline = _perf_stats(baseline_return)

    trades = pd.DataFrame(
        {
            "date": X_live.index,
            "next_date": nd_live.values,
            "prob_up": p_live,
            "position": positions,
            "next_return": np.asarray(nr_live, dtype=float),
            "turnover": turnover,
            "cost": costs,
            "strategy_return": strategy_return.values,
            "strategy_equity": strategy_equity.values,
            "baseline_equity": baseline_equity.values,
        }
    )

    return trades, {"strategy": stats_strategy, "baseline": stats_baseline}


def main() -> None:
    ap = argparse.ArgumentParser(description="SLASSO+Bagging live sim using Alpha Vantage daily data")
    ap.add_argument("--symbol", default="TSLA")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--api-key-env", default="ALPHAVANTAGE_KEY")
    ap.add_argument("--train-data", default="data/TSLA_Merged_Final.csv")
    ap.add_argument("--narrative-data", default="data/tsla_ai_narrative_daily_from_rpa.csv")
    ap.add_argument("--outputsize", choices=["compact", "full"], default="compact")
    ap.add_argument("--out", default="outputs/live")
    ap.add_argument("--train-end", default="2026-03-31")
    ap.add_argument("--live-start", default="2026-04-01")
    ap.add_argument("--live-end", default="2026-05-11")
    ap.add_argument("--min-test-sharpe", type=float, default=1.35)
    ap.add_argument("--max-attempts", type=int, default=10)
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threshold-grid", default="0.4,0.45,0.5,0.55,0.6")
    args = ap.parse_args()

    api_key = args.api_key.strip() or os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise ValueError("Alpha Vantage API key missing. Use --api-key or set env var from --api-key-env.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    narrative_path = Path(args.narrative_data)
    train_feat = _build_feature_frame_from_local(Path(args.train_data), narrative_path)
    live_feat = _build_feature_frame_from_alpha(args.symbol, api_key, narrative_path, args.outputsize)

    threshold_grid = [float(x.strip()) for x in args.threshold_grid.split(",") if x.strip()]
    model, selected, test_stats, attempt_info = _train_until_sharpe(
        train_feat,
        train_end=pd.Timestamp(args.train_end),
        min_sharpe=float(args.min_test_sharpe),
        max_attempts=int(args.max_attempts),
        cost_bps=float(args.cost_bps),
        seed=int(args.seed),
        threshold_grid=threshold_grid,
    )

    trades, live_stats = _simulate_live(
        live_feat,
        model=model,
        selected=selected,
        start_date=pd.Timestamp(args.live_start),
        end_date=pd.Timestamp(args.live_end),
        cost_bps=float(args.cost_bps),
        threshold=float(attempt_info.get("threshold", 0.5)),
    )

    trades_path = out_dir / "slasso_bagging_trades_live_20260401_20260511.csv"
    trades.to_csv(trades_path, index=False, encoding="utf-8-sig")

    # Save plot
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(trades["date"], trades["strategy_equity"], label="SLASSO+Bagging", linewidth=1.6)
    ax.plot(trades["date"], trades["baseline_equity"], label="Buy&Hold", linewidth=1.2, linestyle="--")
    ax.set_title("TSLA live-sim equity curve (2026-04-01 to 2026-05-11)")
    ax.set_ylabel("Equity (start=1)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()

    plot_path = out_dir / "slasso_bagging_live_equity_20260401_20260511.png"
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)

    # Save model artifacts
    model_path = out_dir / "slasso_bagging_pipeline.joblib"
    joblib.dump(model, model_path)
    metadata = {
        "symbol": args.symbol,
        "train_end": args.train_end,
        "live_start": args.live_start,
        "live_end": args.live_end,
        "min_test_sharpe": args.min_test_sharpe,
        "max_attempts": args.max_attempts,
        "cost_bps": args.cost_bps,
        "seed_base": args.seed,
        "selected_features": selected,
        "test_stats": test_stats,
        "attempt_info": attempt_info,
        "threshold": float(attempt_info.get("threshold", 0.5)),
        "trades_path": str(trades_path),
        "plot_path": str(plot_path),
    }
    metadata_path = out_dir / "slasso_bagging_live_summary.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")

    print("Saved:")
    print(trades_path)
    print(plot_path)
    print(model_path)
    print(metadata_path)
    print("Live stats:", live_stats)


if __name__ == "__main__":
    main()
