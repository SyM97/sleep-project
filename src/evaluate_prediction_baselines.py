from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "prediction"
REPORT_DIR = PROJECT_ROOT / "reports" / "prediction"

HORIZONS = [1, 7, 30]

TRAIN_RATIO = 0.80


def hours_to_minutes(hours):
    return hours * 60


def evaluate_predictions(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    absolute_errors = np.abs(y_true - y_pred)

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

    within_30_min = np.mean(
        absolute_errors <= 0.5
    )

    within_60_min = np.mean(
        absolute_errors <= 1.0
    )

    return {
        "mae_hours": round(float(mae), 4),
        "mae_minutes": round(
            float(hours_to_minutes(mae)),
            1,
        ),

        "rmse_hours": round(float(rmse), 4),
        "rmse_minutes": round(
            float(hours_to_minutes(rmse)),
            1,
        ),

        "median_absolute_error_hours":
            round(float(median_ae), 4),

        "median_absolute_error_minutes":
            round(
                float(
                    hours_to_minutes(
                        median_ae
                    )
                ),
                1,
            ),

        "within_30_minutes_pct":
            round(
                float(
                    within_30_min * 100
                ),
                2,
            ),

        "within_60_minutes_pct":
            round(
                float(
                    within_60_min * 100
                ),
                2,
            ),
    }


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


def build_baseline_predictions(
    train,
    test,
):

    train_target_mean = (
        train[
            "target_sleep_hours"
        ]
        .mean()
    )

    predictions = {}

    # ---------------------------------------------------------
    # Baseline 1:
    # media histórica del entrenamiento
    # ---------------------------------------------------------

    predictions[
        "historical_mean"
    ] = pd.Series(
        train_target_mean,
        index=test.index,
    )

    # ---------------------------------------------------------
    # Baseline 2:
    # última duración conocida
    # ---------------------------------------------------------

    predictions[
        "last_sleep"
    ] = (
        test["last_sleep_hours"]
        .fillna(train_target_mean)
    )

    # ---------------------------------------------------------
    # Baseline 3:
    # media últimos 7 días
    # ---------------------------------------------------------

    predictions[
        "rolling_mean_7d"
    ] = (
        test["sleep_mean_7d"]
        .fillna(
            test["sleep_mean_30d"]
        )
        .fillna(
            train_target_mean
        )
    )

    # ---------------------------------------------------------
    # Baseline 4:
    # media últimos 30 días
    # ---------------------------------------------------------

    predictions[
        "rolling_mean_30d"
    ] = (
        test["sleep_mean_30d"]
        .fillna(
            train_target_mean
        )
    )

    # ---------------------------------------------------------
    # Baseline 5:
    # media histórica según día de semana futuro
    #
    # IMPORTANTE:
    # se calcula SOLO usando entrenamiento.
    # ---------------------------------------------------------

    weekday_means = (
        train
        .groupby(
            "target_day_of_week"
        )[
            "target_sleep_hours"
        ]
        .mean()
    )

    predictions[
        "weekday_mean"
    ] = (
        test[
            "target_day_of_week"
        ]
        .map(
            weekday_means
        )
        .fillna(
            train_target_mean
        )
    )

    return predictions


def evaluate_horizon(horizon):

    print()
    print("=" * 72)
    print(
        f"HORIZONTE +{horizon} DÍA(S)"
    )
    print("=" * 72)

    df = load_dataset(
        horizon
    )

    train, test = (
        chronological_split(df)
    )

    print(
        f"Total ejemplos: {len(df)}"
    )

    print(
        f"Entrenamiento: {len(train)}"
    )

    print(
        f"Test: {len(test)}"
    )

    print(
        "\nPeriodo entrenamiento:"
    )

    print(
        f"  {train['origin_date'].min().date()}"
        f" → "
        f"{train['origin_date'].max().date()}"
    )

    print(
        "\nPeriodo test:"
    )

    print(
        f"  {test['origin_date'].min().date()}"
        f" → "
        f"{test['origin_date'].max().date()}"
    )

    predictions = (
        build_baseline_predictions(
            train,
            test,
        )
    )

    y_true = (
        test[
            "target_sleep_hours"
        ]
    )

    results = []

    print(
        "\nRESULTADOS"
    )

    print(
        "-" * 72
    )

    for (
        baseline_name,
        y_pred,
    ) in predictions.items():

        metrics = (
            evaluate_predictions(
                y_true,
                y_pred,
            )
        )

        row = {
            "horizon_days":
                horizon,

            "baseline":
                baseline_name,

            **metrics,
        }

        results.append(row)

        print(
            f"\n{baseline_name}"
        )

        print(
            f"  MAE: "
            f"{metrics['mae_hours']:.3f} h "
            f"({metrics['mae_minutes']:.1f} min)"
        )

        print(
            f"  RMSE: "
            f"{metrics['rmse_hours']:.3f} h "
            f"({metrics['rmse_minutes']:.1f} min)"
        )

        print(
            f"  Error mediano: "
            f"{metrics['median_absolute_error_minutes']:.1f} min"
        )

        print(
            f"  Dentro de ±30 min: "
            f"{metrics['within_30_minutes_pct']:.1f}%"
        )

        print(
            f"  Dentro de ±60 min: "
            f"{metrics['within_60_minutes_pct']:.1f}%"
        )

    results_df = pd.DataFrame(
        results
    )

    best_mae = (
        results_df
        .sort_values(
            "mae_hours"
        )
        .iloc[0]
    )

    print()
    print("-" * 72)

    print(
        "MEJOR BASELINE POR MAE:"
    )

    print(
        f"  {best_mae['baseline']}"
    )

    print(
        f"  Error medio absoluto: "
        f"{best_mae['mae_minutes']:.1f} minutos"
    )

    return {
        "horizon_days":
            horizon,

        "total_examples":
            int(len(df)),

        "train_examples":
            int(len(train)),

        "test_examples":
            int(len(test)),

        "train_start":
            str(
                train[
                    "origin_date"
                ].min().date()
            ),

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

        "test_end":
            str(
                test[
                    "origin_date"
                ].max().date()
            ),

        "best_baseline":
            str(
                best_mae[
                    "baseline"
                ]
            ),

        "best_mae_minutes":
            float(
                best_mae[
                    "mae_minutes"
                ]
            ),

        "results":
            results,
    }


def main():

    print("=" * 72)
    print(
        "EVALUACIÓN DE BASELINES DE PREDICCIÓN"
    )
    print("=" * 72)

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    complete_report = {
        "train_ratio":
            TRAIN_RATIO,

        "evaluation_strategy":
            "chronological_holdout",

        "horizons":
            {},
    }

    all_rows = []

    for horizon in HORIZONS:

        horizon_report = (
            evaluate_horizon(
                horizon
            )
        )

        complete_report[
            "horizons"
        ][str(horizon)] = (
            horizon_report
        )

        all_rows.extend(
            horizon_report[
                "results"
            ]
        )

    # ---------------------------------------------------------
    # CSV resumen
    # ---------------------------------------------------------

    results_df = pd.DataFrame(
        all_rows
    )

    csv_output = (
        REPORT_DIR
        / "baseline_results.csv"
    )

    results_df.to_csv(
        csv_output,
        sep=";",
        index=False,
    )

    # ---------------------------------------------------------
    # JSON completo
    # ---------------------------------------------------------

    json_output = (
        REPORT_DIR
        / "baseline_results.json"
    )

    with open(
        json_output,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            complete_report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 72)
    print("ARCHIVOS GENERADOS")
    print("=" * 72)

    print(
        csv_output
    )

    print(
        json_output
    )

    print()
    print(
        "Evaluación terminada."
    )


if __name__ == "__main__":
    main()