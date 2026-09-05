from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURACIÓN
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "prediction"
    / "prediction_dataset_h1.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "prediction"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "prediction"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "sleep_type_prediction.csv"
)

OUTPUT_JSON = (
    REPORT_DIR
    / "sleep_type_prediction.json"
)


# ============================================================
# DEFINICIÓN DE CATEGORÍAS
# ============================================================

SHORT_MAX_EXCLUSIVE = 6.0
MEDIUM_MAX_INCLUSIVE = 8.0


def classify_sleep_type(hours: float) -> str:
    """
    Clasificación descriptiva utilizada por el proyecto.

    Corto:
        < 6 horas

    Medio:
        >= 6 y <= 8 horas

    Largo:
        > 8 horas
    """

    if hours < SHORT_MAX_EXCLUSIVE:
        return "Corto"

    if hours <= MEDIUM_MAX_INCLUSIVE:
        return "Medio"

    return "Largo"


# ============================================================
# CARGA
# ============================================================

def load_dataset() -> pd.DataFrame:

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        sep=";",
        low_memory=False,
    )

    required_columns = {
        "target_date",
        "target_sleep_hours",
    }

    missing = sorted(
        required_columns.difference(df.columns)
    )

    if missing:
        raise ValueError(
            "Faltan columnas obligatorias:\n- "
            + "\n- ".join(missing)
        )

    df["target_date"] = pd.to_datetime(
        df["target_date"],
        errors="coerce",
    )

    df["target_sleep_hours"] = pd.to_numeric(
        df["target_sleep_hours"],
        errors="coerce",
    )

    return df


# ============================================================
# MAIN
# ============================================================

def main() -> int:

    print("=" * 76)
    print("PREDICCIÓN HISTÓRICA — TIPO DE SUEÑO")
    print("=" * 76)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_dataset()

    original_rows = len(df)

    # --------------------------------------------------------
    # Para calcular la distribución histórica necesitamos
    # únicamente noches con duración real conocida.
    # --------------------------------------------------------

    usable = df.loc[
        df["target_date"].notna()
        & df["target_sleep_hours"].notna()
    ].copy()

    if usable.empty:
        raise ValueError(
            "No existen noches utilizables."
        )

    usable = (
        usable
        .sort_values("target_date")
        .reset_index(drop=True)
    )

    usable["sleep_type"] = (
        usable["target_sleep_hours"]
        .apply(classify_sleep_type)
    )

    # ========================================================
    # DISTRIBUCIÓN
    # ========================================================

    categories = [
        "Corto",
        "Medio",
        "Largo",
    ]

    counts = (
        usable["sleep_type"]
        .value_counts()
        .reindex(
            categories,
            fill_value=0,
        )
    )

    total = int(
        counts.sum()
    )

    probabilities = (
        counts
        / total
    )

    distribution = []

    for category in categories:

        count = int(
            counts[category]
        )

        probability = float(
            probabilities[category]
        )

        distribution.append(
            {
                "sleep_type": category,
                "count": count,
                "probability": probability,
                "probability_pct": round(
                    probability * 100,
                    2,
                ),
            }
        )

    # ========================================================
    # CATEGORÍA MÁS PROBABLE
    # ========================================================

    predicted_type = (
        probabilities.idxmax()
    )

    predicted_probability = float(
        probabilities[
            predicted_type
        ]
    )

    # ========================================================
    # FECHAS
    # ========================================================

    latest_known_date = pd.Timestamp(
        usable["target_date"].max()
    ).normalize()

    target_date = (
        latest_known_date
        + pd.Timedelta(
            1,
            unit="D",
        )
    )

    # ========================================================
    # OUTPUT PARA PRODUCCIÓN
    #
    # El frontend necesitará principalmente:
    #
    # predicted_sleep_type
    # probability_pct
    # ========================================================

    output_row = {
        "as_of_date":
            latest_known_date.date(),

        "target_date":
            target_date.date(),

        "predicted_sleep_type":
            predicted_type,

        "probability":
            round(
                predicted_probability,
                6,
            ),

        "probability_pct":
            round(
                predicted_probability
                * 100,
                2,
            ),

        "prediction_method":
            "historical_frequency",

        "historical_nights":
            total,
    }

    output_df = pd.DataFrame(
        [output_row]
    )

    output_df.to_csv(
        OUTPUT_CSV,
        sep=";",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # INFORME COMPLETO
    # ========================================================

    report = {
        "definition": {
            "Corto": "< 6 h",
            "Medio": "6 h <= duración <= 8 h",
            "Largo": "> 8 h",
        },

        "dataset": {
            "input_file":
                str(INPUT_FILE),

            "original_rows":
                int(original_rows),

            "usable_rows":
                int(len(usable)),

            "excluded_rows":
                int(
                    original_rows
                    - len(usable)
                ),

            "first_sleep_date":
                str(
                    pd.Timestamp(
                        usable[
                            "target_date"
                        ].min()
                    ).date()
                ),

            "last_sleep_date":
                str(
                    latest_known_date.date()
                ),
        },

        "distribution":
            distribution,

        "prediction": {
            "as_of_date":
                str(
                    latest_known_date.date()
                ),

            "target_date":
                str(
                    target_date.date()
                ),

            "predicted_sleep_type":
                predicted_type,

            "probability":
                round(
                    predicted_probability,
                    6,
                ),

            "probability_pct":
                round(
                    predicted_probability
                    * 100,
                    2,
                ),

            "method":
                "historical_frequency",
        },
    }

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # TERMINAL
    # ========================================================

    print()
    print("DEFINICIÓN")
    print("-" * 76)

    print(
        "Corto:                   < 6 h"
    )

    print(
        "Medio:                   6–8 h"
    )

    print(
        "Largo:                   > 8 h"
    )

    print()
    print("HISTÓRICO")
    print("-" * 76)

    print(
        f"Noches utilizables:       "
        f"{total}"
    )

    print(
        f"Desde:                    "
        f"{usable['target_date'].min().date()}"
    )

    print(
        f"Hasta:                    "
        f"{latest_known_date.date()}"
    )

    print()
    print("DISTRIBUCIÓN")
    print("-" * 76)

    for row in distribution:

        print(
            f"{row['sleep_type']:<10}"
            f"{row['count']:>6} noches"
            f"   "
            f"{row['probability_pct']:>6.2f} %"
        )

    print()
    print("=" * 76)
    print("PRÓXIMA NOCHE")
    print("=" * 76)

    print(
        f"Fecha objetivo:           "
        f"{target_date.date()}"
    )

    print(
        f"Tipo más probable:        "
        f"{predicted_type.upper()}"
    )

    print(
        f"Probabilidad histórica:   "
        f"{predicted_probability * 100:.2f} %"
    )

    print(
        f"Método:                   "
        f"historical_frequency"
    )

    print()
    print("=" * 76)
    print("ARCHIVOS GENERADOS")
    print("=" * 76)

    print(
        OUTPUT_CSV
    )

    print(
        OUTPUT_JSON
    )

    print()
    print(
        "La estimación utiliza exclusivamente "
        "la frecuencia observada en el histórico."
    )

    print(
        "No utiliza Machine Learning ni intenta "
        "predecir una duración exacta."
    )

    print("=" * 76)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())