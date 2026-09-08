from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURACIÃ“N
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "prediction"
    / "prediction_dataset_h1.csv"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "prediction"
)

SHORT_SLEEP_THRESHOLD_HOURS = 6.0

TEST_FRACTION = 0.20


# ============================================================
# UTILIDADES
# ============================================================

def safe_pct(value: float) -> float:
    return round(float(value) * 100, 2)


def calculate_log_loss(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    probabilities = np.clip(
        probabilities,
        1e-15,
        1 - 1e-15,
    )

    loss = -(
        y_true * np.log(probabilities)
        + (1 - y_true)
        * np.log(1 - probabilities)
    )

    return float(np.mean(loss))


# ============================================================
# CARGA
# ============================================================

def load_dataset() -> pd.DataFrame:

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"No se encontrÃ³:\n{INPUT_FILE}\n\n"
            "Ejecuta primero prepare_prediction_dataset.py "
            "o el pipeline completo."
        )

    df = pd.read_csv(
        INPUT_FILE,
        sep=";",
        low_memory=False,
    )

    required_columns = {
        "origin_date",
        "target_date",
        "target_sleep_hours",
    }

    missing = sorted(
        required_columns.difference(df.columns)
    )

    if missing:
        raise ValueError(
            "Faltan columnas necesarias:\n- "
            + "\n- ".join(missing)
        )

    df["origin_date"] = pd.to_datetime(
        df["origin_date"],
        errors="coerce",
    )

    df["target_date"] = pd.to_datetime(
        df["target_date"],
        errors="coerce",
    )

    df["target_sleep_hours"] = pd.to_numeric(
        df["target_sleep_hours"],
        errors="coerce",
    )

    return (
        df
        .sort_values("origin_date")
        .reset_index(drop=True)
    )


# ============================================================
# AUDITORÃA
# ============================================================

def main() -> int:

    print("=" * 76)
    print("AUDITORÃA â€” RIESGO DE SUEÃ‘O CORTO")
    print("=" * 76)

    df = load_dataset()

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_rows = len(df)

    target_nulls = int(
        df["target_sleep_hours"]
        .isna()
        .sum()
    )

    invalid_dates = int(
        (
            df["origin_date"].isna()
            | df["target_date"].isna()
        ).sum()
    )

    # --------------------------------------------------------
    # Solo se eliminan aquÃ­ observaciones que no pueden
    # utilizarse como ejemplo supervisado:
    # - sin target
    # - sin fecha de origen
    # - sin fecha objetivo
    #
    # NO se eliminan filas por tener nulos en features.
    # --------------------------------------------------------

    usable = df.loc[
        df["origin_date"].notna()
        & df["target_date"].notna()
        & df["target_sleep_hours"].notna()
    ].copy()

    if usable.empty:
        raise ValueError(
            "No hay ejemplos utilizables."
        )

    # Target binario:
    #
    # 1 = menos de 6 horas
    # 0 = 6 horas o mÃ¡s

    usable["short_sleep_target"] = (
        usable["target_sleep_hours"]
        < SHORT_SLEEP_THRESHOLD_HOURS
    ).astype(int)

    positive_count = int(
        usable["short_sleep_target"].sum()
    )

    negative_count = int(
        len(usable) - positive_count
    )

    positive_rate = float(
        usable["short_sleep_target"].mean()
    )

    # --------------------------------------------------------
    # SPLIT TEMPORAL 80 / 20
    # --------------------------------------------------------

    split_index = int(
        len(usable)
        * (1 - TEST_FRACTION)
    )

    if split_index <= 0 or split_index >= len(usable):
        raise ValueError(
            "No hay suficientes ejemplos para "
            "realizar el split temporal."
        )

    train = (
        usable
        .iloc[:split_index]
        .copy()
    )

    test = (
        usable
        .iloc[split_index:]
        .copy()
    )

    train_rate = float(
        train["short_sleep_target"].mean()
    )

    test_rate = float(
        test["short_sleep_target"].mean()
    )

    # --------------------------------------------------------
    # BASELINE PROBABILÃSTICO
    #
    # Predice siempre la prevalencia observada
    # solamente en el periodo de entrenamiento.
    # --------------------------------------------------------

    baseline_probability = train_rate

    y_test = (
        test["short_sleep_target"]
        .to_numpy(dtype=float)
    )

    baseline_predictions = np.full(
        len(test),
        baseline_probability,
        dtype=float,
    )

    baseline_brier = float(
        np.mean(
            (
                y_test
                - baseline_predictions
            ) ** 2
        )
    )

    baseline_log_loss = calculate_log_loss(
        y_test,
        baseline_predictions,
    )

    # --------------------------------------------------------
    # ÃšLTIMOS 7 RESULTADOS CONOCIDOS
    #
    # Son los Ãºltimos 7 ejemplos disponibles.
    # TodavÃ­a NO los llamamos "Ãºltimos 7 dÃ­as"
    # porque puede haber huecos de calendario.
    # --------------------------------------------------------

    recent_7 = usable.tail(7)

    recent_7_short_count = int(
        recent_7[
            "short_sleep_target"
        ].sum()
    )

    recent_7_rate = float(
        recent_7[
            "short_sleep_target"
        ].mean()
    )

    # --------------------------------------------------------
    # AUDITORÃA DE NULOS EN FEATURES
    # --------------------------------------------------------

    non_feature_columns = {
        "origin_date",
        "target_date",
        "target_sleep_hours",
        "short_sleep_target",
    }

    feature_columns = [
        column
        for column in usable.columns
        if column not in non_feature_columns
    ]

    missing_rows = []

    for column in feature_columns:

        null_count = int(
            usable[column]
            .isna()
            .sum()
        )

        null_pct = (
            null_count
            / len(usable)
            * 100
        )

        missing_rows.append(
            {
                "feature": column,
                "null_count": null_count,
                "null_pct": round(
                    float(null_pct),
                    2,
                ),
                "dtype": str(
                    usable[column].dtype
                ),
            }
        )

    missingness = (
        pd.DataFrame(missing_rows)
        .sort_values(
            [
                "null_pct",
                "feature",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    missingness_file = (
        REPORT_DIR
        / "short_sleep_feature_missingness.csv"
    )

    missingness.to_csv(
        missingness_file,
        sep=";",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # RESUMEN
    # --------------------------------------------------------

    summary = {
        "definition": {
            "target": (
                "1 if next-night sleep duration "
                "is below 6 hours, otherwise 0"
            ),
            "threshold_hours":
                SHORT_SLEEP_THRESHOLD_HOURS,
        },

        "dataset": {
            "input_file":
                str(INPUT_FILE),

            "rows_original":
                int(total_rows),

            "rows_usable":
                int(len(usable)),

            "target_nulls":
                target_nulls,

            "invalid_date_rows":
                invalid_dates,

            "first_origin_date":
                str(
                    usable[
                        "origin_date"
                    ]
                    .min()
                    .date()
                ),

            "last_origin_date":
                str(
                    usable[
                        "origin_date"
                    ]
                    .max()
                    .date()
                ),
        },

        "class_balance": {
            "short_sleep_count":
                positive_count,

            "normal_or_long_sleep_count":
                negative_count,

            "short_sleep_pct":
                safe_pct(
                    positive_rate
                ),
        },

        "temporal_split": {
            "train_examples":
                int(len(train)),

            "test_examples":
                int(len(test)),

            "train_start":
                str(
                    train[
                        "origin_date"
                    ]
                    .min()
                    .date()
                ),

            "train_end":
                str(
                    train[
                        "origin_date"
                    ]
                    .max()
                    .date()
                ),

            "test_start":
                str(
                    test[
                        "origin_date"
                    ]
                    .min()
                    .date()
                ),

            "test_end":
                str(
                    test[
                        "origin_date"
                    ]
                    .max()
                    .date()
                ),

            "train_short_sleep_pct":
                safe_pct(
                    train_rate
                ),

            "test_short_sleep_pct":
                safe_pct(
                    test_rate
                ),
        },

        "probability_baseline": {
            "probability":
                round(
                    baseline_probability,
                    4,
                ),

            "probability_pct":
                safe_pct(
                    baseline_probability
                ),

            "brier_score_test":
                round(
                    baseline_brier,
                    4,
                ),

            "log_loss_test":
                round(
                    baseline_log_loss,
                    4,
                ),
        },

        "recent_7_known_examples": {
            "short_sleep_count":
                recent_7_short_count,

            "short_sleep_pct":
                safe_pct(
                    recent_7_rate
                ),

            "first_target_date":
                str(
                    recent_7[
                        "target_date"
                    ]
                    .min()
                    .date()
                ),

            "last_target_date":
                str(
                    recent_7[
                        "target_date"
                    ]
                    .max()
                    .date()
                ),
        },

        "features": {
            "feature_count":
                int(
                    len(feature_columns)
                ),

            "features_with_nulls":
                int(
                    (
                        missingness[
                            "null_count"
                        ]
                        > 0
                    ).sum()
                ),
        },
    }

    summary_file = (
        REPORT_DIR
        / "short_sleep_target_audit.json"
    )

    with summary_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------
    # SALIDA TERMINAL
    # --------------------------------------------------------

    print()
    print("DEFINICIÃ“N DEL TARGET")
    print("-" * 76)

    print(
        "SueÃ±o corto:"
        f" < {SHORT_SLEEP_THRESHOLD_HOURS:g} horas"
    )

    print(
        "Target:"
        " 1 = sueÃ±o corto,"
        " 0 = 6 h o mÃ¡s"
    )

    print()
    print("DATASET")
    print("-" * 76)

    print(
        f"Filas originales:        "
        f"{total_rows}"
    )

    print(
        f"Ejemplos utilizables:    "
        f"{len(usable)}"
    )

    print(
        f"Targets nulos:           "
        f"{target_nulls}"
    )

    print(
        f"Fechas invÃ¡lidas:        "
        f"{invalid_dates}"
    )

    print()
    print("BALANCE DE CLASES")
    print("-" * 76)

    print(
        f"Noches < 6 h:            "
        f"{positive_count}"
    )

    print(
        f"Noches >= 6 h:           "
        f"{negative_count}"
    )

    print(
        f"Prevalencia < 6 h:       "
        f"{safe_pct(positive_rate):.2f} %"
    )

    print()
    print("SPLIT TEMPORAL")
    print("-" * 76)

    print(
        f"Train:                    "
        f"{len(train)} ejemplos"
    )

    print(
        f"Test:                     "
        f"{len(test)} ejemplos"
    )

    print(
        f"Train < 6 h:              "
        f"{safe_pct(train_rate):.2f} %"
    )

    print(
        f"Test < 6 h:               "
        f"{safe_pct(test_rate):.2f} %"
    )

    print()
    print("BASELINE PROBABILÃSTICO")
    print("-" * 76)

    print(
        f"Probabilidad baseline:    "
        f"{safe_pct(baseline_probability):.2f} %"
    )

    print(
        f"Brier Score test:         "
        f"{baseline_brier:.4f}"
    )

    print(
        f"Log Loss test:            "
        f"{baseline_log_loss:.4f}"
    )

    print()
    print("ÃšLTIMOS 7 EJEMPLOS CONOCIDOS")
    print("-" * 76)

    print(
        f"Noches < 6 h:            "
        f"{recent_7_short_count} / 7"
    )

    print(
        f"Porcentaje:               "
        f"{safe_pct(recent_7_rate):.2f} %"
    )

    print()
    print("NULOS EN FEATURES")
    print("-" * 76)

    features_with_nulls = (
        missingness.loc[
            missingness[
                "null_count"
            ]
            > 0
        ]
    )

    if features_with_nulls.empty:

        print(
            "No hay features con valores nulos."
        )

    else:

        print(
            features_with_nulls[
                [
                    "feature",
                    "null_count",
                    "null_pct",
                ]
            ]
            .to_string(
                index=False
            )
        )

    print()
    print("=" * 76)
    print("ARCHIVOS GENERADOS")
    print("=" * 76)

    print(summary_file)
    print(missingness_file)

    print()
    print(
        "No se ha eliminado ni imputado "
        "ningÃºn valor de las features."
    )

    print(
        "Esta etapa Ãºnicamente audita "
        "el nuevo problema de clasificaciÃ³n."
    )

    print("=" * 76)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
