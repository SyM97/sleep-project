from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    TimeSeriesSplit,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "prediction"
REPORT_DIR = PROJECT_ROOT / "reports" / "prediction"

HORIZONS = [1, 7, 30]
TRAIN_RATIO = 0.80
RANDOM_STATE = 42


FEATURES = [
    "last_sleep_hours",
    "last_sleep_quality_pct",
    "last_sleep_efficiency_pct",
    "last_regularity_pct",
    "last_sleep_latency_minutes",
    "last_bedtime_minutes_adjusted",

    "days_since_last_record",

    "sleep_mean_3d",
    "sleep_mean_7d",
    "sleep_mean_14d",
    "sleep_mean_30d",
    "sleep_std_7d",
    "sleep_std_30d",
    "sleep_trend_7d_vs_30d",

    "quality_mean_7d",
    "quality_mean_30d",

    "efficiency_mean_7d",
    "efficiency_mean_30d",

    "regularity_mean_7d",
    "regularity_mean_30d",

    "latency_mean_7d",
    "latency_mean_30d",

    "bedtime_median_7d",
    "bedtime_median_30d",

    "records_7d",
    "records_14d",
    "records_30d",

    "coverage_7d",
    "coverage_30d",

    "target_is_weekend",
    "target_dow_sin",
    "target_dow_cos",
    "target_month_sin",
    "target_month_cos",
]


def load_dataset(horizon):

    file_path = (
        DATA_DIR
        / f"prediction_dataset_h{horizon}.csv"
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"No se encontró: {file_path}"
        )

    df = pd.read_csv(
        file_path,
        sep=";",
    )

    df["origin_date"] = pd.to_datetime(
        df["origin_date"]
    )

    df["target_date"] = pd.to_datetime(
        df["target_date"]
    )

    df = (
        df
        .sort_values("origin_date")
        .reset_index(drop=True)
    )

    return df


def purged_final_split(df):

    split_index = int(
        len(df) * TRAIN_RATIO
    )

    raw_train = (
        df
        .iloc[:split_index]
        .copy()
    )

    test = (
        df
        .iloc[split_index:]
        .copy()
    )

    test_start = (
        test["origin_date"].min()
    )

    # IMPORTANTE:
    # eliminamos cualquier fila cuyo resultado
    # futuro aún no estaría disponible cuando
    # comienza el test.
    train = raw_train[
        raw_train["target_date"]
        < test_start
    ].copy()

    train = (
        train
        .reset_index(drop=True)
    )

    test = (
        test
        .reset_index(drop=True)
    )

    purged_rows = (
        len(raw_train)
        - len(train)
    )

    return (
        train,
        test,
        purged_rows,
    )


def create_purged_cv_splits(
    train,
    n_splits=5,
):

    normal_splitter = TimeSeriesSplit(
        n_splits=n_splits
    )

    purged_splits = []

    for (
        train_indices,
        validation_indices,
    ) in normal_splitter.split(train):

        validation_start = (
            train
            .iloc[validation_indices]
            ["origin_date"]
            .min()
        )

        candidate_train = (
            train
            .iloc[train_indices]
        )

        available_mask = (
            candidate_train[
                "target_date"
            ]
            < validation_start
        )

        purged_train_indices = (
            np.asarray(train_indices)[
                available_mask.to_numpy()
            ]
        )

        if len(
            purged_train_indices
        ) < 20:
            continue

        purged_splits.append(
            (
                purged_train_indices,
                np.asarray(
                    validation_indices
                ),
            )
        )

    if len(purged_splits) < 2:
        raise ValueError(
            "No hay suficientes folds "
            "temporales después de aplicar "
            "la purga."
        )

    return purged_splits


def create_models():

    ridge = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                Ridge(),
            ),
        ]
    )

    random_forest = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    gradient_boosting = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_iter=250,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    return {
        "ridge": (
            ridge,
            {
                "model__alpha": [
                    0.1,
                    1.0,
                    10.0,
                    100.0,
                ],
            },
        ),

        "random_forest": (
            random_forest,
            {
                "model__max_depth": [
                    3,
                    5,
                    None,
                ],
                "model__min_samples_leaf": [
                    3,
                    5,
                    10,
                ],
                "model__max_features": [
                    0.5,
                    1.0,
                ],
            },
        ),

        "hist_gradient_boosting": (
            gradient_boosting,
            {
                "model__learning_rate": [
                    0.05,
                    0.1,
                ],
                "model__max_leaf_nodes": [
                    7,
                    15,
                ],
                "model__min_samples_leaf": [
                    10,
                    20,
                ],
                "model__l2_regularization": [
                    1.0,
                    10.0,
                ],
            },
        ),
    }


def metrics(
    y_true,
    y_pred,
):

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    errors = np.abs(
        y_true - y_pred
    )

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    return {
        "mae_minutes":
            round(
                float(mae * 60),
                2,
            ),

        "rmse_minutes":
            round(
                float(rmse * 60),
                2,
            ),

        "median_error_minutes":
            round(
                float(
                    np.median(errors)
                    * 60
                ),
                2,
            ),

        "within_30_minutes_pct":
            round(
                float(
                    np.mean(
                        errors <= 0.5
                    )
                    * 100
                ),
                2,
            ),

        "within_60_minutes_pct":
            round(
                float(
                    np.mean(
                        errors <= 1.0
                    )
                    * 100
                ),
                2,
            ),

        "r2":
            round(
                float(
                    r2_score(
                        y_true,
                        y_pred,
                    )
                ),
                4,
            ),
    }


def evaluate_horizon(horizon):

    print()
    print("=" * 76)
    print(
        f"VALIDACIÓN PURGADA — +{horizon} DÍA(S)"
    )
    print("=" * 76)

    df = load_dataset(
        horizon
    )

    (
        train,
        test,
        purged_rows,
    ) = purged_final_split(
        df
    )

    print(
        f"Total: {len(df)}"
    )

    print(
        f"Entrenamiento tras purga: "
        f"{len(train)}"
    )

    print(
        f"Filas eliminadas por purga: "
        f"{purged_rows}"
    )

    print(
        f"Test: {len(test)}"
    )

    print(
        "Último target conocido "
        "en entrenamiento:"
    )

    print(
        f"  {train['target_date'].max().date()}"
    )

    print(
        "Primer origen del test:"
    )

    print(
        f"  {test['origin_date'].min().date()}"
    )

    X_train = train[FEATURES]

    y_train = (
        train[
            "target_sleep_hours"
        ]
    )

    X_test = test[FEATURES]

    y_test = (
        test[
            "target_sleep_hours"
        ]
    )

    # -------------------------------------------------
    # BASELINE
    # -------------------------------------------------

    historical_mean = (
        y_train.mean()
    )

    baseline_predictions = (
        np.repeat(
            historical_mean,
            len(test),
        )
    )

    baseline_metrics = metrics(
        y_test,
        baseline_predictions,
    )

    baseline_mae = (
        baseline_metrics[
            "mae_minutes"
        ]
    )

    print()
    print(
        "BASELINE: historical_mean"
    )

    print(
        f"  Predicción constante: "
        f"{historical_mean:.3f} h"
    )

    print(
        f"  MAE: "
        f"{baseline_mae:.2f} min"
    )

    print(
        f"  RMSE: "
        f"{baseline_metrics['rmse_minutes']:.2f} min"
    )

    # -------------------------------------------------
    # CV PURGADA
    # -------------------------------------------------

    cv_splits = (
        create_purged_cv_splits(
            train,
            n_splits=5,
        )
    )

    model_results = []

    for (
        model_name,
        (
            estimator,
            param_grid,
        ),
    ) in create_models().items():

        print()
        print("-" * 76)
        print(
            f"Modelo: {model_name}"
        )

        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring=(
                "neg_mean_absolute_error"
            ),
            cv=cv_splits,
            n_jobs=-1,
            refit=True,
        )

        search.fit(
            X_train,
            y_train,
        )

        predictions = search.predict(
            X_test
        )

        result_metrics = metrics(
            y_test,
            predictions,
        )

        cv_mae_minutes = (
            -search.best_score_
            * 60
        )

        difference = (
            result_metrics[
                "mae_minutes"
            ]
            - baseline_mae
        )

        improvement_pct = (
            (
                baseline_mae
                - result_metrics[
                    "mae_minutes"
                ]
            )
            / baseline_mae
            * 100
        )

        result = {
            "horizon_days":
                horizon,

            "model":
                model_name,

            "train_rows":
                int(len(train)),

            "test_rows":
                int(len(test)),

            "purged_rows":
                int(purged_rows),

            "cv_mae_minutes":
                round(
                    float(
                        cv_mae_minutes
                    ),
                    2,
                ),

            "baseline_mae_minutes":
                baseline_mae,

            **result_metrics,

            "difference_vs_baseline_minutes":
                round(
                    float(difference),
                    2,
                ),

            "improvement_vs_baseline_pct":
                round(
                    float(
                        improvement_pct
                    ),
                    2,
                ),

            "beats_baseline":
                bool(
                    result_metrics[
                        "mae_minutes"
                    ]
                    < baseline_mae
                ),

            "best_params":
                search.best_params_,
        }

        model_results.append(
            result
        )

        print(
            f"  CV MAE: "
            f"{cv_mae_minutes:.2f} min"
        )

        print(
            f"  TEST MAE: "
            f"{result_metrics['mae_minutes']:.2f} min"
        )

        print(
            f"  Diferencia vs baseline: "
            f"{difference:+.2f} min"
        )

        print(
            f"  Dentro ±60 min: "
            f"{result_metrics['within_60_minutes_pct']:.2f}%"
        )

        print(
            f"  R²: "
            f"{result_metrics['r2']:.4f}"
        )

        print(
            f"  Parámetros: "
            f"{search.best_params_}"
        )

    return {
        "horizon_days":
            horizon,

        "baseline": {
            "model":
                "historical_mean",

            "train_mean_hours":
                round(
                    float(
                        historical_mean
                    ),
                    4,
                ),

            **baseline_metrics,
        },

        "models":
            model_results,
    }


def main():

    print("=" * 76)
    print(
        "VALIDACIÓN TEMPORAL PURGADA"
    )
    print("=" * 76)

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "evaluation":
            "purged chronological holdout + "
            "purged TimeSeriesSplit",

        "horizons": {},
    }

    rows = []

    for horizon in HORIZONS:

        horizon_result = (
            evaluate_horizon(
                horizon
            )
        )

        report[
            "horizons"
        ][str(horizon)] = (
            horizon_result
        )

        for model_result in (
            horizon_result[
                "models"
            ]
        ):
            rows.append(
                model_result
            )

    csv_file = (
        REPORT_DIR
        / "purged_model_results.csv"
    )

    pd.DataFrame(
        rows
    ).to_csv(
        csv_file,
        sep=";",
        index=False,
    )

    json_file = (
        REPORT_DIR
        / "purged_model_results.json"
    )

    with open(
        json_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 76)
    print(
        "VALIDACIÓN FINALIZADA"
    )
    print("=" * 76)

    print(
        csv_file
    )

    print(
        json_file
    )

    print()
    print(
        "Todavía no se ha seleccionado "
        "ningún modelo de producción."
    )


if __name__ == "__main__":
    main()