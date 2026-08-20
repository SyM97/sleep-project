from pathlib import Path
import json

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "prediction"
REPORT_DIR = PROJECT_ROOT / "reports" / "prediction"

HORIZONS = [1, 7, 30]

MIN_HISTORY = 60


def load_prediction_dataset(horizon):
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


def get_latest_data_date():
    history_file = (
        DATA_DIR
        / "prediction_daily_history.csv"
    )

    if not history_file.exists():
        raise FileNotFoundError(
            f"No se encontró: {history_file}"
        )

    history = pd.read_csv(
        history_file,
        sep=";",
    )

    history["calendar_date"] = pd.to_datetime(
        history["calendar_date"]
    )

    if history["has_data"].dtype == bool:
        has_data = history["has_data"]
    else:
        has_data = (
            history["has_data"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("true")
        )

    observed = history[
        has_data
    ].copy()

    if observed.empty:
        raise ValueError(
            "No existen días observados."
        )

    latest_date = (
        observed[
            "calendar_date"
        ]
        .max()
    )

    return (
        pd.Timestamp(
            latest_date
        )
        .normalize()
    )


def build_walk_forward_errors(df):
    """
    Simula predicciones históricas reales.

    Para cada fecha de origen:
    solamente utiliza resultados cuyo target_date
    era anterior a esa fecha.

    Esto evita utilizar información futura.
    """

    results = []

    for i in range(len(df)):

        row = df.iloc[i]

        origin_date = (
            row["origin_date"]
        )

        # Solo resultados que ya habrían ocurrido
        # antes de realizar esta predicción.
        available = df[
            df["target_date"]
            < origin_date
        ]

        if len(available) < MIN_HISTORY:
            continue

        prediction = (
            available[
                "target_sleep_hours"
            ]
            .mean()
        )

        actual = float(
            row[
                "target_sleep_hours"
            ]
        )

        error = (
            actual
            - prediction
        )

        absolute_error = abs(
            error
        )

        results.append(
            {
                "origin_date":
                    origin_date,

                "target_date":
                    row[
                        "target_date"
                    ],

                "prediction_hours":
                    float(
                        prediction
                    ),

                "actual_hours":
                    actual,

                "error_hours":
                    float(
                        error
                    ),

                "absolute_error_hours":
                    float(
                        absolute_error
                    ),

                "absolute_error_minutes":
                    float(
                        absolute_error
                        * 60
                    ),

                "training_examples":
                    int(
                        len(available)
                    ),
            }
        )

    result_df = pd.DataFrame(
        results
    )

    if result_df.empty:
        raise ValueError(
            "No existen suficientes predicciones "
            "walk-forward para calibrar intervalos."
        )

    return result_df


def calculate_uncertainty(calibration):
    """
    Calcula intervalos simétricos usando
    la distribución empírica del error absoluto.

    80 %:
        aproximadamente 8 de cada 10 errores
        históricos quedaron dentro de este rango.

    90 %:
        aproximadamente 9 de cada 10.
    """

    errors = (
        calibration[
            "absolute_error_hours"
        ]
        .dropna()
    )

    q80 = errors.quantile(
        0.80,
        interpolation="higher",
    )

    q90 = errors.quantile(
        0.90,
        interpolation="higher",
    )

    mae = errors.mean()

    median_error = (
        errors.median()
    )

    within_30 = (
        errors <= 0.5
    ).mean()

    within_60 = (
        errors <= 1.0
    ).mean()

    return {
        "calibration_examples":
            int(len(errors)),

        "mae_hours":
            float(mae),

        "mae_minutes":
            float(
                mae * 60
            ),

        "median_absolute_error_minutes":
            float(
                median_error * 60
            ),

        "interval_80_half_width_hours":
            float(q80),

        "interval_80_half_width_minutes":
            float(q80 * 60),

        "interval_90_half_width_hours":
            float(q90),

        "interval_90_half_width_minutes":
            float(q90 * 60),

        "within_30_minutes_pct":
            float(
                within_30 * 100
            ),

        "within_60_minutes_pct":
            float(
                within_60 * 100
            ),
    }


def build_current_forecast(
    horizon,
    as_of_date,
):

    # Normalizamos explícitamente la fecha de origen y convertimos
    # el horizonte a int para evitar operaciones con unidades
    # timedelta genéricas.
    as_of_date = (
        pd.Timestamp(
            as_of_date
        )
        .normalize()
    )

    horizon = int(horizon)

    df = load_prediction_dataset(
        horizon
    )

    # Solo resultados que ya conocemos
    # en la fecha actual del dataset.
    available = df[
        df["target_date"]
        <= as_of_date
    ].copy()

    if available.empty:
        raise ValueError(
            f"No hay historial utilizable "
            f"para horizonte {horizon}."
        )

    prediction = (
        available[
            "target_sleep_hours"
        ]
        .mean()
    )

    calibration = (
        build_walk_forward_errors(
            df
        )
    )

    uncertainty = (
        calculate_uncertainty(
            calibration
        )
    )

    half_80 = (
        uncertainty[
            "interval_80_half_width_hours"
        ]
    )

    half_90 = (
        uncertainty[
            "interval_90_half_width_hours"
        ]
    )

    target_date = (
        as_of_date
        + pd.Timedelta(
            horizon,
            unit="D",
        )
    )

    low_80 = max(
        0.0,
        prediction - half_80,
    )

    high_80 = (
        prediction + half_80
    )

    low_90 = max(
        0.0,
        prediction - half_90,
    )

    high_90 = (
        prediction + half_90
    )

    forecast = {
        "as_of_date":
            str(
                as_of_date.date()
            ),

        "target_date":
            str(
                target_date.date()
            ),

        "horizon_days":
            int(horizon),

        "prediction_method":
            "historical_mean",

        "uses_machine_learning":
            False,

        "training_examples":
            int(
                len(available)
            ),

        "predicted_sleep_hours":
            round(
                float(prediction),
                4,
            ),

        "predicted_sleep_minutes":
            round(
                float(
                    prediction * 60
                )
            ),

        "interval_80_low_hours":
            round(
                float(low_80),
                4,
            ),

        "interval_80_high_hours":
            round(
                float(high_80),
                4,
            ),

        "interval_90_low_hours":
            round(
                float(low_90),
                4,
            ),

        "interval_90_high_hours":
            round(
                float(high_90),
                4,
            ),

        "estimated_mae_minutes":
            round(
                uncertainty[
                    "mae_minutes"
                ],
                1,
            ),

        "median_absolute_error_minutes":
            round(
                uncertainty[
                    "median_absolute_error_minutes"
                ],
                1,
            ),

        "interval_80_half_width_minutes":
            round(
                uncertainty[
                    "interval_80_half_width_minutes"
                ],
                1,
            ),

        "interval_90_half_width_minutes":
            round(
                uncertainty[
                    "interval_90_half_width_minutes"
                ],
                1,
            ),

        "within_30_minutes_pct":
            round(
                uncertainty[
                    "within_30_minutes_pct"
                ],
                1,
            ),

        "within_60_minutes_pct":
            round(
                uncertainty[
                    "within_60_minutes_pct"
                ],
                1,
            ),

        "calibration_examples":
            uncertainty[
                "calibration_examples"
            ],
    }

    return (
        forecast,
        calibration,
    )


def decimal_hours_to_text(value):
    total_minutes = int(
        round(
            value * 60
        )
    )

    hours = (
        total_minutes // 60
    )

    minutes = (
        total_minutes % 60
    )

    return (
        f"{hours} h "
        f"{minutes:02d} min"
    )


def main():

    print("=" * 76)
    print(
        "PREDICTOR DE PRODUCCIÓN — DURACIÓN DEL SUEÑO"
    )
    print("=" * 76)

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    as_of_date = (
        get_latest_data_date()
    )

    print()
    print(
        "Última fecha con datos:"
    )

    print(
        f"  {as_of_date.date()}"
    )

    forecasts = []

    report = {
        "as_of_date":
            str(
                as_of_date.date()
            ),

        "method":
            "historical_mean",

        "machine_learning_used":
            False,

        "selection_reason":
            (
                "Los modelos de Machine Learning "
                "no mostraron una mejora robusta "
                "frente al baseline en validación "
                "temporal purgada."
            ),

        "interval_method":
            (
                "Empirical absolute errors from "
                "leakage-free walk-forward forecasts"
            ),

        "forecasts":
            [],
    }

    for horizon in HORIZONS:

        (
            forecast,
            calibration,
        ) = build_current_forecast(
            horizon,
            as_of_date,
        )

        forecasts.append(
            forecast
        )

        report[
            "forecasts"
        ].append(
            forecast
        )

        calibration_file = (
            REPORT_DIR
            / f"production_calibration_h{horizon}.csv"
        )

        calibration.to_csv(
            calibration_file,
            sep=";",
            index=False,
        )

        print()
        print("-" * 76)

        print(
            f"HORIZONTE +{horizon} DÍA(S)"
        )

        print(
            f"Fecha objetivo: "
            f"{forecast['target_date']}"
        )

        print(
            "Duración estimada:"
        )

        print(
            "  "
            + decimal_hours_to_text(
                forecast[
                    "predicted_sleep_hours"
                ]
            )
        )

        print(
            "Rango empírico 80 %:"
        )

        print(
            "  "
            + decimal_hours_to_text(
                forecast[
                    "interval_80_low_hours"
                ]
            )
            + " — "
            + decimal_hours_to_text(
                forecast[
                    "interval_80_high_hours"
                ]
            )
        )

        print(
            "Rango empírico 90 %:"
        )

        print(
            "  "
            + decimal_hours_to_text(
                forecast[
                    "interval_90_low_hours"
                ]
            )
            + " — "
            + decimal_hours_to_text(
                forecast[
                    "interval_90_high_hours"
                ]
            )
        )

        print(
            f"MAE walk-forward estimado: "
            f"{forecast['estimated_mae_minutes']:.1f} min"
        )

        print(
            f"Ejemplos de calibración: "
            f"{forecast['calibration_examples']}"
        )

    # ---------------------------------------------------------
    # CSV para futuro dashboard / Supabase
    # ---------------------------------------------------------

    forecast_df = pd.DataFrame(
        forecasts
    )

    forecast_csv = (
        DATA_DIR
        / "sleep_forecast_current.csv"
    )

    forecast_df.to_csv(
        forecast_csv,
        sep=";",
        index=False,
    )

    # ---------------------------------------------------------
    # JSON de informe
    # ---------------------------------------------------------

    report_json = (
        REPORT_DIR
        / "production_forecast_summary.json"
    )

    with open(
        report_json,
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
        "ARCHIVOS GENERADOS"
    )
    print("=" * 76)

    print(
        forecast_csv
    )

    print(
        report_json
    )

    print()
    print(
        "IMPORTANTE:"
    )

    print(
        "Estas cifras son estimaciones estadísticas "
        "personales, no predicciones médicas."
    )

    print(
        "La incertidumbre debe mostrarse siempre "
        "junto a la predicción puntual."
    )


if __name__ == "__main__":
    main()