from pathlib import Path
import os

import pandas as pd

from dotenv import load_dotenv

from sqlalchemy import (
    MetaData,
    Table,
    create_engine,
    select,
)
from sqlalchemy.engine import URL

from sqlalchemy.dialects.postgresql import insert


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ENV_FILE = PROJECT_ROOT / ".env"

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "prediction"
    / "sleep_forecast_current.csv"
)

TABLE_NAME = "sleep_forecasts"
SCHEMA_NAME = "public"


EXPECTED_COLUMNS = [
    "as_of_date",
    "target_date",
    "horizon_days",
    "prediction_method",
    "uses_machine_learning",
    "training_examples",
    "predicted_sleep_hours",
    "predicted_sleep_minutes",
    "interval_80_low_hours",
    "interval_80_high_hours",
    "interval_90_low_hours",
    "interval_90_high_hours",
    "estimated_mae_minutes",
    "median_absolute_error_minutes",
    "interval_80_half_width_minutes",
    "interval_90_half_width_minutes",
    "within_30_minutes_pct",
    "within_60_minutes_pct",
    "calibration_examples",
]


def load_environment():
    if not ENV_FILE.exists():
        raise FileNotFoundError(
            f"No se encontrÃ³ el archivo .env: {ENV_FILE}"
        )

    load_dotenv(ENV_FILE)

    required = [
        "SUPABASE_DB_HOST",
        "SUPABASE_DB_PORT",
        "SUPABASE_DB_NAME",
        "SUPABASE_DB_USER",
        "SUPABASE_DB_PASSWORD",
    ]

    missing = [
        name
        for name in required
        if not os.getenv(name)
    ]

    if missing:
        raise ValueError(
            "Faltan variables en .env: "
            + ", ".join(missing)
        )


def create_database_engine():
    url = URL.create(
        drivername="postgresql+psycopg",
        username=os.getenv(
            "SUPABASE_DB_USER"
        ),
        password=os.getenv(
            "SUPABASE_DB_PASSWORD"
        ),
        host=os.getenv(
            "SUPABASE_DB_HOST"
        ),
        port=int(
            os.getenv(
                "SUPABASE_DB_PORT"
            )
        ),
        database=os.getenv(
            "SUPABASE_DB_NAME"
        ),
    )

    return create_engine(
        url,
        pool_pre_ping=True,
    )


def load_forecast_csv():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"No se encontrÃ³: {INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        sep=";",
    )

    missing_columns = [
        col
        for col in EXPECTED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Faltan columnas en el CSV: "
            + ", ".join(missing_columns)
        )

    df = df[
        EXPECTED_COLUMNS
    ].copy()

    if df.empty:
        raise ValueError(
            "El archivo de predicciones estÃ¡ vacÃ­o."
        )

    # ---------------------------------------------
    # Fechas
    # ---------------------------------------------

    df["as_of_date"] = pd.to_datetime(
        df["as_of_date"],
        errors="raise",
    ).dt.date

    df["target_date"] = pd.to_datetime(
        df["target_date"],
        errors="raise",
    ).dt.date

    # ---------------------------------------------
    # Validaciones
    # ---------------------------------------------

    valid_horizons = {
        1,
        7,
        30,
    }

    actual_horizons = set(
        df[
            "horizon_days"
        ].astype(int)
    )

    invalid_horizons = (
        actual_horizons
        - valid_horizons
    )

    if invalid_horizons:
        raise ValueError(
            "Horizontes no permitidos: "
            + str(
                sorted(
                    invalid_horizons
                )
            )
        )

    duplicate_keys = (
        df.duplicated(
            subset=[
                "as_of_date",
                "horizon_days",
            ],
            keep=False,
        )
    )

    if duplicate_keys.any():
        duplicates = df.loc[
            duplicate_keys,
            [
                "as_of_date",
                "horizon_days",
            ],
        ]

        raise ValueError(
            "Existen claves duplicadas:\n"
            + duplicates.to_string(
                index=False
            )
        )

    invalid_target_dates = (
        df["target_date"]
        <= df["as_of_date"]
    )

    if invalid_target_dates.any():
        raise ValueError(
            "Hay target_date que no son "
            "posteriores a as_of_date."
        )

    if (
        df["predicted_sleep_hours"] < 0
    ).any():
        raise ValueError(
            "Hay predicciones negativas."
        )

    # Convertir NaN de pandas a None
    # para PostgreSQL.
    df = df.astype(object).where(
        pd.notna(df),
        None,
    )

    return df


def reflect_table(
    engine,
):
    metadata = MetaData()

    return Table(
        TABLE_NAME,
        metadata,
        schema=SCHEMA_NAME,
        autoload_with=engine,
    )


def upsert_forecasts(
    connection,
    table,
    dataframe,
):
    records = dataframe.to_dict(
        orient="records"
    )

    statement = insert(
        table
    ).values(
        records
    )

    # No modificamos:
    # - as_of_date
    # - horizon_days
    # - created_at
    #
    # El resto se actualiza si ya existe
    # la misma predicciÃ³n.
    update_columns = {
        column:
            getattr(
                statement.excluded,
                column,
            )
        for column in EXPECTED_COLUMNS
        if column not in {
            "as_of_date",
            "horizon_days",
        }
    }

    statement = (
        statement
        .on_conflict_do_update(
            index_elements=[
                "as_of_date",
                "horizon_days",
            ],
            set_=update_columns,
        )
    )

    connection.execute(
        statement
    )


def verify_loaded_rows(
    connection,
    table,
    dataframe,
):
    as_of_dates = sorted(
        set(
            dataframe[
                "as_of_date"
            ]
        )
    )

    statement = (
        select(
            table.c.as_of_date,
            table.c.target_date,
            table.c.horizon_days,
            table.c.predicted_sleep_hours,
            table.c.interval_80_low_hours,
            table.c.interval_80_high_hours,
        )
        .where(
            table.c.as_of_date.in_(
                as_of_dates
            )
        )
        .order_by(
            table.c.as_of_date,
            table.c.horizon_days,
        )
    )

    rows = connection.execute(
        statement
    ).mappings().all()

    return rows


def main():
    print("=" * 72)
    print(
        "CARGA DE PREDICCIONES A SUPABASE"
    )
    print("=" * 72)

    load_environment()

    forecast_df = (
        load_forecast_csv()
    )

    print()
    print(
        f"Archivo: {INPUT_FILE}"
    )

    print(
        f"Predicciones encontradas: "
        f"{len(forecast_df)}"
    )

    print()
    print(
        "Predicciones que se cargarÃ¡n:"
    )

    for _, row in forecast_df.iterrows():
        print(
            f"  {row['as_of_date']} "
            f"â†’ +{row['horizon_days']} dÃ­as "
            f"â†’ {row['target_date']} "
            f"â†’ {float(row['predicted_sleep_hours']):.2f} h"
        )

    engine = (
        create_database_engine()
    )

    try:
        table = reflect_table(
            engine
        )

        actual_columns = {
            column.name
            for column in table.columns
        }

        missing_in_database = [
            col
            for col in EXPECTED_COLUMNS
            if col not in actual_columns
        ]

        if missing_in_database:
            raise ValueError(
                "Faltan columnas en Supabase: "
                + ", ".join(
                    missing_in_database
                )
            )

        # Una Ãºnica transacciÃ³n.
        # Si algo falla, se revierte todo.
        with engine.begin() as connection:

            # Advisory lock para evitar dos
            # loaders simultÃ¡neos.
            connection.exec_driver_sql(
                """
                SELECT pg_advisory_xact_lock(
                    hashtext(
                        'sleep_forecasts_loader'
                    )
                )
                """
            )

            upsert_forecasts(
                connection,
                table,
                forecast_df,
            )

            loaded_rows = (
                verify_loaded_rows(
                    connection,
                    table,
                    forecast_df,
                )
            )

        print()
        print("=" * 72)
        print(
            "VERIFICACIÃ“N"
        )
        print("=" * 72)

        for row in loaded_rows:
            print(
                f"{row['as_of_date']} "
                f"| +{row['horizon_days']} dÃ­as "
                f"| objetivo {row['target_date']} "
                f"| predicciÃ³n "
                f"{row['predicted_sleep_hours']:.2f} h "
                f"| 80% "
                f"{row['interval_80_low_hours']:.2f}"
                f"â€“"
                f"{row['interval_80_high_hours']:.2f} h"
            )

        if len(
            loaded_rows
        ) < len(
            forecast_df
        ):
            raise RuntimeError(
                "La verificaciÃ³n devolviÃ³ menos "
                "filas de las esperadas."
            )

        print()
        print(
            "Carga completada correctamente."
        )

        print(
            "No se ha eliminado ningÃºn "
            "histÃ³rico anterior."
        )

    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
