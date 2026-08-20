from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import TransformedTargetRegressor
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
MODEL_DIR = PROJECT_ROOT / "models" / "prediction"

BASELINE_FILE = (
    REPORT_DIR
    / "baseline_results.json"
)

HORIZONS = [1, 7, 30]
TRAIN_RATIO = 0.80
RANDOM_STATE = 42


FEATURES = [
    # Último registro conocido
    "last_sleep_hours",
    "last_sleep_quality_pct",
    "last_sleep_efficiency_pct",
    "last_regularity_pct",
    "last_sleep_latency_minutes",
    "last_bedtime_minutes_adjusted",

    # Recencia
    "days_since_last_record",

    # Duración reciente
    "sleep_mean_3d",
    "sleep_mean_7d",
    "sleep_mean_14d",
    "sleep_mean_30d",
    "sleep_std_7d",
    "sleep_std_30d",
    "sleep_trend_7d_vs_30d",

    # Calidad
    "quality_mean_7d",
    "quality_mean_30d",

    # Eficiencia
    "efficiency_mean_7d",
    "efficiency_mean_30d",

    # Regularidad
    "regularity_mean_7d",
    "regularity_mean_30d",

    # Latencia
    "latency_mean_7d",
    "latency_mean_30d",

    # Horario
    "bedtime_median_7d",
    "bedtime_median_30d",

    # Cobertura
    "records_7d",
    "records_14d",
    "records_30d",
    "coverage_7d",
    "coverage_30d",

    # Calendario futuro conocido
    "target_is_weekend",
    "target_dow_sin",
    "target_dow_cos",
    "target_month_sin",
    "target_month_cos",
]


def load_baselines():
    if not BASELINE_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró: {BASELINE_FILE}"
        )

    with open(
        BASELINE_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


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

    missing_features = [
        col
        for col in FEATURES
        if col not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "Faltan columnas predictoras: "
            + ", ".join(missing_features)
        )

    return df


def chronological_split(df):
    split_index = int(
        len(df) * TRAIN_RATIO
    )

    train = (
        df
        .iloc[:split_index]
        .copy()
    )

    test = (
        df
        .iloc[split_index:]
        .copy()
    )

    return train, test


def evaluate_predictions(y_true, y_pred):
    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    absolute_errors = np.abs(
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

    median_ae = np.median(
        absolute_errors
    )

    return {
        "mae_hours":
            round(float(mae), 4),

        "mae_minutes":
            round(float(mae * 60), 1),

        "rmse_hours":
            round(float(rmse), 4),

        "rmse_minutes":
            round(float(rmse * 60), 1),

        "median_absolute_error_minutes":
            round(
                float(
                    median_ae * 60
                ),
                1,
            ),

        "within_30_minutes_pct":
            round(
                float(
                    np.mean(
                        absolute_errors <= 0.5
                    ) * 100
                ),
                2,
            ),

        "within_60_minutes_pct":
            round(
                float(
                    np.mean(
                        absolute_errors <= 1.0
                    ) * 100
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


def create_model_searches():

    # ---------------------------------------------------------
    # Ridge
    # ---------------------------------------------------------

    ridge_pipeline = Pipeline(
        steps=[
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

    ridge_grid = {
        "model__alpha": [
            0.1,
            1.0,
            10.0,
            100.0,
        ]
    }

    # ---------------------------------------------------------
    # Random Forest
    # ---------------------------------------------------------

    rf_pipeline = Pipeline(
        steps=[
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

    rf_grid = {
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
    }

    # ---------------------------------------------------------
    # HistGradientBoosting
    # ---------------------------------------------------------

    hgb_pipeline = Pipeline(
        steps=[
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

    hgb_grid = {
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
    }

    return {
        "ridge": (
            ridge_pipeline,
            ridge_grid,
        ),

        "random_forest": (
            rf_pipeline,
            rf_grid,
        ),

        "hist_gradient_boosting": (
            hgb_pipeline,
            hgb_grid,
        ),
    }


def evaluate_horizon(
    horizon,
    baseline_report,
):

    print()
    print("=" * 76)
    print(
        f"MODELOS ML — HORIZONTE +{horizon} DÍA(S)"
    )
    print("=" * 76)

    df = load_dataset(
        horizon
    )

    train, test = (
        chronological_split(df)
    )

    X_train = train[FEATURES]
    y_train = train[
        "target_sleep_hours"
    ]

    X_test = test[FEATURES]
    y_test = test[
        "target_sleep_hours"
    ]

    print(
        f"Entrenamiento: {len(train)}"
    )

    print(
        f"Test final: {len(test)}"
    )

    print(
        "Periodo test:"
    )

    print(
        f"  {test['origin_date'].min().date()}"
        f" → "
        f"{test['origin_date'].max().date()}"
    )

    # ---------------------------------------------------------
    # Baseline
    # ---------------------------------------------------------

    baseline_info = (
        baseline_report[
            "horizons"
        ][str(horizon)]
    )

    baseline_name = (
        baseline_info[
            "best_baseline"
        ]
    )

    baseline_mae_minutes = float(
        baseline_info[
            "best_mae_minutes"
        ]
    )

    print()
    print(
        f"Mejor baseline: {baseline_name}"
    )

    print(
        f"MAE baseline: "
        f"{baseline_mae_minutes:.1f} min"
    )

    # ---------------------------------------------------------
    # Validación temporal SOLO dentro del entrenamiento
    # ---------------------------------------------------------

    time_cv = TimeSeriesSplit(
        n_splits=5
    )

    searches = (
        create_model_searches()
    )

    model_results = []

    fitted_searches = {}

    for (
        model_name,
        (
            pipeline,
            param_grid,
        ),
    ) in searches.items():

        print()
        print("-" * 76)

        print(
            f"Entrenando: {model_name}"
        )

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=(
                "neg_mean_absolute_error"
            ),
            cv=time_cv,
            n_jobs=-1,
            refit=True,
        )

        search.fit(
            X_train,
            y_train,
        )

        fitted_searches[
            model_name
        ] = search

        y_pred = search.predict(
            X_test
        )

        metrics = (
            evaluate_predictions(
                y_test,
                y_pred,
            )
        )

        cv_mae_hours = (
            -search.best_score_
        )

        cv_mae_minutes = (
            cv_mae_hours * 60
        )

        improvement_pct = (
            (
                baseline_mae_minutes
                - metrics["mae_minutes"]
            )
            / baseline_mae_minutes
            * 100
        )

        result = {
            "horizon_days":
                horizon,

            "model":
                model_name,

            "cv_mae_minutes":
                round(
                    float(
                        cv_mae_minutes
                    ),
                    1,
                ),

            "test_mae_minutes":
                metrics[
                    "mae_minutes"
                ],

            "test_rmse_minutes":
                metrics[
                    "rmse_minutes"
                ],

            "median_absolute_error_minutes":
                metrics[
                    "median_absolute_error_minutes"
                ],

            "within_30_minutes_pct":
                metrics[
                    "within_30_minutes_pct"
                ],

            "within_60_minutes_pct":
                metrics[
                    "within_60_minutes_pct"
                ],

            "r2":
                metrics["r2"],

            "baseline":
                baseline_name,

            "baseline_mae_minutes":
                baseline_mae_minutes,

            "improvement_vs_baseline_pct":
                round(
                    float(
                        improvement_pct
                    ),
                    2,
                ),

            "beats_baseline":
                bool(
                    metrics[
                        "mae_minutes"
                    ]
                    < baseline_mae_minutes
                ),

            "best_params":
                search.best_params_,
        }

        model_results.append(
            result
        )

        print(
            f"CV MAE: "
            f"{cv_mae_minutes:.1f} min"
        )

        print(
            f"TEST MAE: "
            f"{metrics['mae_minutes']:.1f} min"
        )

        print(
            f"TEST RMSE: "
            f"{metrics['rmse_minutes']:.1f} min"
        )

        print(
            f"Error mediano: "
            f"{metrics['median_absolute_error_minutes']:.1f} min"
        )

        print(
            f"Dentro de ±30 min: "
            f"{metrics['within_30_minutes_pct']:.1f}%"
        )

        print(
            f"Dentro de ±60 min: "
            f"{metrics['within_60_minutes_pct']:.1f}%"
        )

        print(
            f"R²: "
            f"{metrics['r2']:.3f}"
        )

        print(
            f"Mejora vs baseline: "
            f"{improvement_pct:+.2f}%"
        )

        print(
            "Mejores parámetros:"
        )

        print(
            f"  {search.best_params_}"
        )

    # ---------------------------------------------------------
    # Mejor ML según el TEST FINAL
    #
    # Esto es únicamente para evaluación.
    # Después comprobaremos robustez antes de decidir
    # el modelo definitivo.
    # ---------------------------------------------------------

    result_df = pd.DataFrame(
        model_results
    )

    best_row = (
        result_df
        .sort_values(
            "test_mae_minutes"
        )
        .iloc[0]
    )

    best_model_name = (
        best_row["model"]
    )

    best_search = (
        fitted_searches[
            best_model_name
        ]
    )

    model_file = (
        MODEL_DIR
        / f"candidate_sleep_model_h{horizon}.joblib"
    )

    joblib.dump(
        {
            "model":
                best_search.best_estimator_,

            "features":
                FEATURES,

            "horizon_days":
                horizon,

            "model_name":
                best_model_name,

            "best_params":
                best_search.best_params_,

            "train_end":
                str(
                    train[
                        "origin_date"
                    ].max().date()
                ),

            "test_start":
                str(
                    test[
                        "origin_date"
                    ].min().date()
                ),
        },
        model_file,
    )

    # ---------------------------------------------------------
    # Predicciones del mejor candidato
    # ---------------------------------------------------------

    best_predictions = (
        best_search
        .best_estimator_
        .predict(
            X_test
        )
    )

    historical_mean = (
        y_train.mean()
    )

    prediction_output = pd.DataFrame(
        {
            "origin_date":
                test[
                    "origin_date"
                ].dt.date,

            "target_date":
                test[
                    "target_date"
                ].dt.date,

            "actual_sleep_hours":
                y_test.values,

            "baseline_prediction_hours":
                historical_mean,

            "ml_prediction_hours":
                best_predictions,
        }
    )

    prediction_output[
        "baseline_absolute_error_minutes"
    ] = (
        np.abs(
            prediction_output[
                "actual_sleep_hours"
            ]
            - prediction_output[
                "baseline_prediction_hours"
            ]
        )
        * 60
    ).round(1)

    prediction_output[
        "ml_absolute_error_minutes"
    ] = (
        np.abs(
            prediction_output[
                "actual_sleep_hours"
            ]
            - prediction_output[
                "ml_prediction_hours"
            ]
        )
        * 60
    ).round(1)

    prediction_file = (
        REPORT_DIR
        / f"ml_test_predictions_h{horizon}.csv"
    )

    prediction_output.to_csv(
        prediction_file,
        sep=";",
        index=False,
    )

    print()
    print("-" * 76)

    print(
        "MEJOR CANDIDATO ML:"
    )

    print(
        f"  {best_model_name}"
    )

    print(
        f"  MAE test: "
        f"{best_row['test_mae_minutes']:.1f} min"
    )

    print(
        f"  Baseline: "
        f"{baseline_mae_minutes:.1f} min"
    )

    print(
        f"  Mejora: "
        f"{best_row['improvement_vs_baseline_pct']:+.2f}%"
    )

    print(
        f"  Modelo candidato guardado:"
    )

    print(
        f"  {model_file}"
    )

    return model_results


def main():

    print("=" * 76)
    print(
        "ENTRENAMIENTO Y EVALUACIÓN DE MODELOS DE SUEÑO"
    )
    print("=" * 76)

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    baseline_report = (
        load_baselines()
    )

    all_results = []

    for horizon in HORIZONS:

        horizon_results = (
            evaluate_horizon(
                horizon,
                baseline_report,
            )
        )

        all_results.extend(
            horizon_results
        )

    results_df = pd.DataFrame(
        all_results
    )

    csv_file = (
        REPORT_DIR
        / "ml_model_results.csv"
    )

    results_df.to_csv(
        csv_file,
        sep=";",
        index=False,
    )

    json_file = (
        REPORT_DIR
        / "ml_model_results.json"
    )

    with open(
        json_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "evaluation_strategy":
                    "80% chronological train + "
                    "5-fold TimeSeriesSplit tuning + "
                    "20% untouched final test",

                "features":
                    FEATURES,

                "results":
                    all_results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 76)
    print(
        "ENTRENAMIENTO TERMINADO"
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
        "IMPORTANTE:"
    )

    print(
        "Los modelos guardados siguen siendo candidatos."
    )

    print(
        "No deben usarse todavía para generar "
        "predicciones en producción."
    )


if __name__ == "__main__":
    main()