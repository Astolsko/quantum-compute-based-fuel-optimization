"""
train_model.py  --  Ship fuel-rate prediction models (report section 6 + eval).

We train the report's "baseline -> ensemble -> boosting" progression, then the
models added to answer what that progression left open:

    1. Linear Regression   (baseline; fast, but underfits the cubic speed law)
    2. Random Forest       (captures non-linearity; strong classical baseline)
    3. Gradient Boosting   (HistGradientBoosting; state-of-the-art on tabular)
    4. Gradient Boosting (log, monotone)
                           same booster, trained on ln(rate) because the noise is
                           multiplicative, with physics as monotonic constraints
    5. XGBoost / 6. LightGBM   (the other two standard boosters; skipped if the
                           packages are not installed)
    7. Physics + Boosting  (core/physics_model.py: Admiralty-law form + boosting on
                           the residual -- the only one with a SMOOTH speed curve)

Target: fuel-burn RATE in tonnes/day. The model is domain-agnostic -- it simply
reads the feature schema from config.py, so the same code trains on ship data.

Each model is wrapped in a scikit-learn Pipeline that one-hot-encodes the
categorical columns, so the whole thing is a single fit/predict object. We score
every model with R2, MAE, RMSE and MAPE on a held-out test set, plus 5-fold
cross-validation R2 for robustness. Given the true physics (synthetic data only)
we also score DECISION REGRET -- how much fuel the optimizer wastes by trusting
each model -- which is the number that matters downstream. The best model (by
test R2) is saved to models/ so the optimizer and (later) the dashboard reuse it.

Run:
    python train_model.py
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits

import config
import generate_data
import optimizer
from core.physics_model import PhysicsResidualRegressor

try:                                    # Phase 2 optional extras
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None
try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    """Load the synthetic dataset, generating it first if it doesn't exist."""
    if not config.DATA_CSV.exists():
        print("No dataset found -- generating a default one (10,000 rows)...")
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        generate_data.generate().to_csv(config.DATA_CSV, index=False)
    return pd.read_csv(config.DATA_CSV)


def _one_hot_encoder() -> OneHotEncoder:
    """OneHotEncoder that returns dense output, across sklearn versions."""
    try:  # sklearn >= 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # older sklearn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_pipeline(estimator) -> Pipeline:
    """Wrap an estimator with the shared preprocessing (one-hot categoricals).
    The encoded matrix keeps its column names, so constraints can be given by
    feature name (config.MONOTONE) however many one-hot columns there are."""
    preprocess = ColumnTransformer([
        ("num", "passthrough", config.NUMERIC_FEATURES),
        ("cat", _one_hot_encoder(), config.CATEGORICAL),
    ], verbose_feature_names_out=False).set_output(transform="pandas")
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def _log_target(estimator) -> TransformedTargetRegressor:
    """Fit on ln(fuel rate): the noise is multiplicative, so that is the scale
    where errors are even-sized (and the scale MAPE measures)."""
    return TransformedTargetRegressor(regressor=estimator, func=np.log,
                                      inverse_func=np.exp)


def _metrics(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse,
        "MAPE%": mape,
    }


def candidate_models() -> dict:
    """name -> unfitted model taking the raw feature DataFrame."""
    estimators = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, n_jobs=config.N_JOBS, random_state=RANDOM_STATE
        ),
        "Gradient Boosting": HistGradientBoostingRegressor(
            random_state=RANDOM_STATE
        ),
        "Gradient Boosting (log, monotone)": _log_target(
            HistGradientBoostingRegressor(monotonic_cst=config.MONOTONE,
                                          random_state=RANDOM_STATE)
        ),
    }
    if XGBRegressor is not None:
        estimators["XGBoost"] = XGBRegressor(
            n_jobs=config.N_JOBS, random_state=RANDOM_STATE
        )
    if LGBMRegressor is not None:
        estimators["LightGBM"] = LGBMRegressor(
            n_jobs=config.N_JOBS, random_state=RANDOM_STATE, verbose=-1
        )
    models = {name: build_pipeline(est) for name, est in estimators.items()}
    # The grey box reads named columns itself; only its residual booster needs
    # the encoding. Early stopping keeps that booster quiet where the physical
    # form already fits.
    models["Physics + Boosting"] = PhysicsResidualRegressor(build_pipeline(
        HistGradientBoostingRegressor(early_stopping=True,
                                      random_state=RANDOM_STATE)
    ))
    return models


def train_all(df: pd.DataFrame, truth=None):
    """Train + evaluate all candidate models. Returns (results_df, name, pipe).
    With `truth` (the true physics; synthetic data only) each model also gets
    its decision regret over optimizer.regret_panel."""
    X = df[config.FEATURE_COLUMNS]
    y = df[config.TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    rows, fitted = [], {}
    # Boosters thread through OpenMP regardless of n_jobs; cap those pools too.
    with threadpool_limits(limits=config.N_JOBS if config.N_JOBS > 0 else None):
        for name, pipe in candidate_models().items():
            # 5-fold cross-validated R2 (robustness check on the full dataset).
            cv = cross_val_score(pipe, X, y, cv=5, scoring="r2")

            # Held-out test performance.
            pipe.fit(X_train, y_train)
            m = _metrics(y_test, pipe.predict(X_test))

            fitted[name] = pipe
            row = {
                "Model": name,
                "CV_R2_mean": round(cv.mean(), 4),
                "CV_R2_std": round(cv.std(), 4),
                "Test_R2": round(m["R2"], 4),
                "Test_MAE": round(m["MAE"], 3),
                "Test_RMSE": round(m["RMSE"], 3),
                "Test_MAPE%": round(m["MAPE%"], 2),
            }
            if truth is not None:
                regret = optimizer.regret_panel(pipe, truth)
                row["Regret_mean%"] = round(float(regret.mean()), 2)
                row["Regret_max%"] = round(float(regret.max()), 2)
            rows.append(row)

    results = pd.DataFrame(rows).sort_values("Test_R2", ascending=False)
    best_name = results.iloc[0]["Model"]
    return results, best_name, fitted[best_name]


def physics_note(pipe) -> str:
    """One line of fitted physical coefficients, if the model has any."""
    coef = getattr(pipe, "coefficients", None)
    if coef is None:
        return ""
    return "Fitted physics: " + ", ".join(f"{k}={v:.4g}" for k, v in coef().items())


def save(best_name: str, best_pipe: Pipeline, df: pd.DataFrame) -> None:
    """Persist the best pipeline + metadata for downstream use."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipe, config.MODEL_PATH)

    meta = {
        "best_model": best_name,
        # Pickles do not survive a scikit-learn upgrade; optimizer.load_model
        # retrains when this does not match the running version.
        "sklearn_version": sklearn.__version__,
        "feature_columns": config.FEATURE_COLUMNS,
        "numeric_features": config.NUMERIC_FEATURES,
        "categorical": config.CATEGORICAL,
        "target": config.TARGET,
        # Handy for the future dashboard: valid dropdown values.
        "categorical_options": {c: sorted(df[c].unique().tolist())
                                for c in config.CATEGORICAL},
    }
    joblib.dump(meta, config.META_PATH)


def main() -> None:
    df = load_data()
    print(f"Loaded {len(df):,} rows from {config.DATA_CSV.name}\n")

    # ponytail: assumes the CSV is the synthetic set (the only source today);
    # drop truth= when real voyage logs replace it -- there is no oracle then.
    results, best_name, best_pipe = train_all(df, truth=generate_data.TruthModel())

    print("Model comparison (sorted by test R2):")
    print(results.to_string(index=False))
    print(f"\nBest model: {best_name}")
    if physics_note(best_pipe):
        print(physics_note(best_pipe))

    save(best_name, best_pipe, df)
    print(f"Saved best model -> {config.MODEL_PATH}")
    print(f"Saved metadata   -> {config.META_PATH}")


if __name__ == "__main__":
    main()
