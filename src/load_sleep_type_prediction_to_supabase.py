from pathlib import Path
import os

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client


# ============================================================
# CONFIGURACIÓN
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "prediction"
    / "sleep_type_prediction.csv"
)

TABLE_NAME = "sleep_type_predictions"

VALID_SLEEP_TYPES = {
    "Corto",
    "Medio",
    "Largo",
}


# ============================================================
# ENTORNO
# ============================================================

def load_environment():

    env_file = (
        PROJECT_ROOT
        / ".env"
    )

    load_dotenv(
        env_file
    )

    supabase_url = (
        os.getenv(
            "SUPABASE_URL"
        )
    )

    supabase_secret_key = (
        os.getenv(
            "SUPABASE_SECRET_KEY"
        )
    )

    if not supabase_url:

        raise RuntimeError(
            "Falta SUPABASE_URL "
            "en las variables de entorno."
        )

    if not supabase_secret_key:

        raise RuntimeError(
            "Falta SUPABASE_SECRET_KEY "
            "en las variables de entorno."
        )

    return (
        supabase_url,
        supabase_secret_key,
    )


# ============================================================
# CARGA Y VALIDACIÓN DEL CSV
# ============================================================

def load_prediction():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            "No se encontró el archivo:\n"
            f"{INPUT_FILE}\n\n"
            "Ejecuta primero:\n"
            "src/build_sleep_type_prediction.py"
        )

    df = pd.read_csv(
        INPUT_FILE,
        sep=";",
        low_memory=False,
    )

    if len(df) != 1:

        raise ValueError(
            "sleep_type_prediction.csv "
            "debe contener exactamente una fila."
        )

    required_columns = {
        "as_of_date",
        "target_date",
        "predicted_sleep_type",
        "probability",
        "probability_pct",
        "prediction_method",
        "historical_nights",
    }

    missing_columns = (
        required_columns
        .difference(
            df.columns
        )
    )

    if missing_columns:

        raise ValueError(
            "Faltan columnas obligatorias:\n- "
            + "\n- ".join(
                sorted(
                    missing_columns
                )
            )
        )

    row = df.iloc[0].copy()

    # ========================================================
    # FECHAS
    # ========================================================

    as_of_date = pd.to_datetime(
        row[
            "as_of_date"
        ],
        errors="coerce",
    )

    target_date = pd.to_datetime(
        row[
            "target_date"
        ],
        errors="coerce",
    )

    if pd.isna(
        as_of_date
    ):

        raise ValueError(
            "as_of_date no es válida."
        )

    if pd.isna(
        target_date
    ):

        raise ValueError(
            "target_date no es válida."
        )

    if target_date <= as_of_date:

        raise ValueError(
            "target_date debe ser posterior "
            "a as_of_date."
        )

    # ========================================================
    # TIPO DE SUEÑO
    # ========================================================

    sleep_type = str(
        row[
            "predicted_sleep_type"
        ]
    ).strip()

    if (
        sleep_type
        not in VALID_SLEEP_TYPES
    ):

        raise ValueError(
            "predicted_sleep_type "
            "debe ser Corto, Medio o Largo."
        )

    # ========================================================
    # PROBABILIDAD
    # ========================================================

    probability = float(
        row[
            "probability"
        ]
    )

    probability_pct = float(
        row[
            "probability_pct"
        ]
    )

    if not (
        0.0
        <= probability
        <= 1.0
    ):

        raise ValueError(
            "probability debe estar "
            "entre 0 y 1."
        )

    if not (
        0.0
        <= probability_pct
        <= 100.0
    ):

        raise ValueError(
            "probability_pct debe estar "
            "entre 0 y 100."
        )

    expected_pct = (
        probability
        * 100
    )

    if abs(
        expected_pct
        - probability_pct
    ) > 0.05:

        raise ValueError(
            "probability y probability_pct "
            "no son coherentes entre sí."
        )

    # ========================================================
    # HISTÓRICO
    # ========================================================

    historical_nights = int(
        row[
            "historical_nights"
        ]
    )

    if historical_nights <= 0:

        raise ValueError(
            "historical_nights "
            "debe ser mayor que 0."
        )

    # ========================================================
    # MÉTODO
    # ========================================================

    prediction_method = str(
        row[
            "prediction_method"
        ]
    ).strip()

    if not prediction_method:

        raise ValueError(
            "prediction_method "
            "no puede estar vacío."
        )

    # ========================================================
    # PAYLOAD
    # ========================================================

    payload = {
        "as_of_date":
            str(
                as_of_date.date()
            ),

        "target_date":
            str(
                target_date.date()
            ),

        "predicted_sleep_type":
            sleep_type,

        "probability":
            round(
                probability,
                6,
            ),

        "probability_pct":
            round(
                probability_pct,
                2,
            ),

        "prediction_method":
            prediction_method,

        "historical_nights":
            historical_nights,
    }

    return payload


# ============================================================
# SUPABASE
# ============================================================

def load_to_supabase(
    supabase,
    payload,
):

    response = (
        supabase
        .table(
            TABLE_NAME
        )
        .upsert(
            payload,
            on_conflict="as_of_date",
        )
        .execute()
    )

    if not response.data:

        raise RuntimeError(
            "Supabase no devolvió datos "
            "después del UPSERT."
        )

    return response.data


# ============================================================
# VERIFICACIÓN
# ============================================================

def verify_supabase(
    supabase,
    expected,
):

    response = (
        supabase
        .table(
            TABLE_NAME
        )
        .select(
            "as_of_date,"
            "target_date,"
            "predicted_sleep_type,"
            "probability,"
            "probability_pct,"
            "prediction_method,"
            "historical_nights"
        )
        .eq(
            "as_of_date",
            expected[
                "as_of_date"
            ],
        )
        .limit(1)
        .execute()
    )

    rows = (
        response.data
        or []
    )

    if not rows:

        raise RuntimeError(
            "No se encontró la predicción "
            "después de cargarla."
        )

    actual = rows[0]

    checks = {
        "target_date":
            expected[
                "target_date"
            ],

        "predicted_sleep_type":
            expected[
                "predicted_sleep_type"
            ],

        "prediction_method":
            expected[
                "prediction_method"
            ],

        "historical_nights":
            expected[
                "historical_nights"
            ],
    }

    for key, expected_value in (
        checks.items()
    ):

        actual_value = (
            actual.get(
                key
            )
        )

        if str(
            actual_value
        ) != str(
            expected_value
        ):

            raise RuntimeError(
                "Verificación fallida para "
                f"{key}. "
                f"Esperado: {expected_value}. "
                f"Encontrado: {actual_value}."
            )

    actual_probability = float(
        actual[
            "probability"
        ]
    )

    if abs(
        actual_probability
        - expected[
            "probability"
        ]
    ) > 1e-5:

        raise RuntimeError(
            "La probabilidad guardada "
            "no coincide."
        )

    return actual


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 76)
    print(
        "LOAD SLEEP TYPE PREDICTION "
        "TO SUPABASE"
    )
    print("=" * 76)

    # --------------------------------------------------------
    # Variables de entorno
    # --------------------------------------------------------

    (
        supabase_url,
        supabase_secret_key,
    ) = load_environment()

    # --------------------------------------------------------
    # Validar CSV
    # --------------------------------------------------------

    payload = (
        load_prediction()
    )

    print()
    print(
        "PREDICCIÓN A CARGAR"
    )

    print("-" * 76)

    print(
        f"Datos hasta:              "
        f"{payload['as_of_date']}"
    )

    print(
        f"Noche objetivo:           "
        f"{payload['target_date']}"
    )

    print(
        f"Tipo de sueño:            "
        f"{payload['predicted_sleep_type']}"
    )

    print(
        f"Probabilidad:             "
        f"{payload['probability_pct']:.2f} %"
    )

    print(
        f"Método:                   "
        f"{payload['prediction_method']}"
    )

    print(
        f"Noches históricas:        "
        f"{payload['historical_nights']}"
    )

    # --------------------------------------------------------
    # Cliente Supabase
    # --------------------------------------------------------

    supabase = create_client(
        supabase_url,
        supabase_secret_key,
    )

    # --------------------------------------------------------
    # UPSERT
    # --------------------------------------------------------

    print()
    print(
        "Cargando a Supabase..."
    )

    load_to_supabase(
        supabase,
        payload,
    )

    # --------------------------------------------------------
    # Verificación
    # --------------------------------------------------------

    print(
        "Verificando registro..."
    )

    actual = verify_supabase(
        supabase,
        payload,
    )

    print()
    print("=" * 76)

    print(
        "SLEEP TYPE PREDICTION "
        "VERIFIED"
    )

    print("=" * 76)

    print(
        f"{actual['target_date']} "
        f"→ "
        f"{actual['predicted_sleep_type'].upper()} "
        f"· "
        f"{float(actual['probability_pct']):.2f} %"
    )

    print()
    print(
        "La predicción está disponible "
        "en Supabase."
    )

    print("=" * 76)

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )