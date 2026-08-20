#!/usr/bin/env python3
"""
Carga las nueve tablas del dashboard de sueño en Supabase PostgreSQL.

El script:
1. Comprueba que existen los nueve CSV.
2. Valida columnas y claves únicas.
3. Se conecta a PostgreSQL usando variables de .env.
4. Comprueba que las tablas remotas coinciden con los CSV.
5. Sustituye todas las tablas mediante inserciones tipadas por lotes dentro de una única transacción.
6. Verifica el número de filas insertadas.

Uso:
    python src/load_dashboard_to_supabase.py \
        --data-dir data/dashboard

Primera comprobación sin conectarse:
    python src/load_dashboard_to_supabase.py \
        --data-dir data/dashboard \
        --check-files-only

Ejecución sin confirmación interactiva:
    python src/load_dashboard_to_supabase.py \
        --data-dir data/dashboard \
        --yes
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import MetaData, Table, URL, create_engine, text
from sqlalchemy.engine import Connection, Engine


@dataclass(frozen=True)
class TableConfig:
    table_name: str
    file_name: str
    primary_key: tuple[str, ...]


TABLES: tuple[TableConfig, ...] = (
    TableConfig(
        "dashboard_kpis",
        "dashboard_kpis.csv",
        ("generated_at",),
    ),
    TableConfig(
        "dashboard_daily",
        "dashboard_daily.csv",
        ("sleep_date",),
    ),
    TableConfig(
        "dashboard_weekly",
        "dashboard_weekly.csv",
        ("week_start",),
    ),
    TableConfig(
        "dashboard_monthly",
        "dashboard_monthly.csv",
        ("month_key",),
    ),
    TableConfig(
        "dashboard_weekday",
        "dashboard_weekday.csv",
        ("day_of_week_num",),
    ),
    TableConfig(
        "dashboard_note_tags",
        "dashboard_note_tags.csv",
        ("tag_key",),
    ),
    TableConfig(
        "dashboard_quality_factors",
        "dashboard_quality_factors.csv",
        ("metric_key",),
    ),
    TableConfig(
        "dashboard_distributions",
        "dashboard_distributions.csv",
        ("dimension_key", "bucket_order"),
    ),
    TableConfig(
        "dashboard_recent_sessions",
        "dashboard_recent_sessions.csv",
        ("session_key",),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Carga las tablas del dashboard en Supabase."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/dashboard"),
        help="Carpeta que contiene los nueve CSV.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Archivo con las credenciales. Predeterminado: .env",
    )
    parser.add_argument(
        "--check-files-only",
        action="store_true",
        help="Valida únicamente los archivos locales, sin conectarse.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Omite la confirmación antes de reemplazar las tablas.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    try:
        dataframe = pd.read_csv(
            path,
            sep=";",
            encoding="utf-8-sig",
            low_memory=False,
            keep_default_na=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"No se pudo leer {path.name}: {exc}"
        ) from exc

    if len(dataframe.columns) == 1:
        raise ValueError(
            f"{path.name} se ha leído como una sola columna. "
            "Comprueba que el separador sea ';'."
        )

    return dataframe


def validate_local_files(
    data_dir: Path,
) -> dict[str, pd.DataFrame]:
    loaded: dict[str, pd.DataFrame] = {}
    errors: list[str] = []

    for config in TABLES:
        path = data_dir / config.file_name

        if not path.exists():
            errors.append(f"Falta el archivo: {path}")
            continue

        try:
            dataframe = read_csv(path)
        except Exception as exc:
            errors.append(str(exc))
            continue

        missing_keys = [
            key
            for key in config.primary_key
            if key not in dataframe.columns
        ]

        if missing_keys:
            errors.append(
                f"{config.file_name}: faltan columnas de clave "
                f"{missing_keys}"
            )
            continue

        null_key_rows = dataframe[
            list(config.primary_key)
        ].isna().any(axis=1)

        if null_key_rows.any():
            errors.append(
                f"{config.file_name}: "
                f"{int(null_key_rows.sum())} filas tienen clave nula."
            )

        duplicate_rows = dataframe.duplicated(
            subset=list(config.primary_key),
            keep=False,
        )

        if duplicate_rows.any():
            errors.append(
                f"{config.file_name}: "
                f"{int(duplicate_rows.sum())} filas tienen clave duplicada."
            )

        loaded[config.table_name] = dataframe

    if errors:
        raise ValueError(
            "Errores en los archivos locales:\n- "
            + "\n- ".join(errors)
        )

    return loaded


def build_database_url(env_file: Path) -> URL | str:
    if not env_file.exists():
        raise FileNotFoundError(
            f"No existe {env_file}. Crea el archivo a partir de "
            ".env.example."
        )

    load_dotenv(env_file, override=True)

    database_url = os.getenv("DATABASE_URL", "").strip()

    if database_url:
        if database_url.startswith("postgres://"):
            database_url = (
                "postgresql+psycopg://"
                + database_url.removeprefix("postgres://")
            )
        elif database_url.startswith("postgresql://"):
            database_url = (
                "postgresql+psycopg://"
                + database_url.removeprefix("postgresql://")
            )

        return database_url

    required = {
        "SUPABASE_DB_HOST": os.getenv("SUPABASE_DB_HOST"),
        "SUPABASE_DB_PORT": os.getenv("SUPABASE_DB_PORT", "5432"),
        "SUPABASE_DB_NAME": os.getenv("SUPABASE_DB_NAME", "postgres"),
        "SUPABASE_DB_USER": os.getenv("SUPABASE_DB_USER"),
        "SUPABASE_DB_PASSWORD": os.getenv("SUPABASE_DB_PASSWORD"),
    }

    missing = [
        name
        for name, value in required.items()
        if value is None or str(value).strip() == ""
    ]

    if missing:
        raise ValueError(
            "Faltan variables en .env: " + ", ".join(missing)
        )

    try:
        port = int(str(required["SUPABASE_DB_PORT"]))
    except ValueError as exc:
        raise ValueError(
            "SUPABASE_DB_PORT debe ser un número."
        ) from exc

    # URL.create codifica de manera segura contraseñas con @, :, /, etc.
    return URL.create(
        drivername="postgresql+psycopg",
        username=str(required["SUPABASE_DB_USER"]),
        password=str(required["SUPABASE_DB_PASSWORD"]),
        host=str(required["SUPABASE_DB_HOST"]),
        port=port,
        database=str(required["SUPABASE_DB_NAME"]),
        query={"sslmode": "require"},
    )


def create_database_engine(
    env_file: Path,
) -> Engine:
    database_url = build_database_url(env_file)

    return create_engine(
        database_url,
        pool_pre_ping=True,
        pool_size=1,
        max_overflow=0,
        connect_args={
            "connect_timeout": 30,
            "application_name": "sleep_dashboard_loader",
        },
    )


def get_remote_columns(
    connection: Connection,
    table_name: str,
) -> list[dict[str, Any]]:
    query = text(
        """
        select
            column_name,
            data_type,
            udt_name,
            is_nullable,
            ordinal_position
        from information_schema.columns
        where table_schema = 'public'
          and table_name = :table_name
        order by ordinal_position
        """
    )

    rows = connection.execute(
        query,
        {"table_name": table_name},
    ).mappings().all()

    if not rows:
        raise ValueError(
            f"No existe la tabla public.{table_name}."
        )

    return [dict(row) for row in rows]


def parse_boolean(series: pd.Series, column: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")

    normalized = series.astype("string").str.strip().str.lower()
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
        "sí": True,
        "si": True,
    }

    converted = normalized.map(mapping).astype("boolean")
    invalid = series.notna() & converted.isna()

    if invalid.any():
        examples = series.loc[invalid].astype(str).head(3).tolist()
        raise ValueError(
            f"La columna {column} contiene booleanos inválidos: "
            f"{examples}"
        )

    return converted


def convert_dataframe_for_postgres(
    dataframe: pd.DataFrame,
    remote_columns: list[dict[str, Any]],
    table_name: str,
) -> pd.DataFrame:
    remote_names = [
        column["column_name"]
        for column in remote_columns
    ]

    local_set = set(dataframe.columns)
    remote_set = set(remote_names)

    missing_local = sorted(remote_set - local_set)
    unexpected_local = sorted(local_set - remote_set)

    if missing_local or unexpected_local:
        messages: list[str] = []

        if missing_local:
            messages.append(
                f"faltan en el CSV: {missing_local}"
            )

        if unexpected_local:
            messages.append(
                f"sobran en el CSV: {unexpected_local}"
            )

        raise ValueError(
            f"{table_name}: las columnas no coinciden; "
            + "; ".join(messages)
        )

    converted = dataframe[remote_names].copy()

    for metadata in remote_columns:
        column = metadata["column_name"]
        data_type = metadata["data_type"]
        series = converted[column]

        if data_type == "date":
            parsed = pd.to_datetime(series, errors="coerce")
            invalid = series.notna() & parsed.isna()

            if invalid.any():
                raise ValueError(
                    f"{table_name}.{column}: "
                    f"{int(invalid.sum())} fechas inválidas."
                )

            converted[column] = parsed.dt.date

        elif data_type.startswith("timestamp"):
            parsed = pd.to_datetime(series, errors="coerce")
            invalid = series.notna() & parsed.isna()

            if invalid.any():
                raise ValueError(
                    f"{table_name}.{column}: "
                    f"{int(invalid.sum())} timestamps inválidos."
                )

            # Las tablas creadas usan timestamp without time zone.
            if getattr(parsed.dt, "tz", None) is not None:
                parsed = parsed.dt.tz_localize(None)

            converted[column] = parsed

        elif data_type.startswith("time"):
            parsed = pd.to_datetime(
                series,
                format="%H:%M",
                errors="coerce",
            )

            # Permite también HH:MM:SS.
            needs_second_attempt = series.notna() & parsed.isna()
            if needs_second_attempt.any():
                parsed_seconds = pd.to_datetime(
                    series.loc[needs_second_attempt],
                    format="%H:%M:%S",
                    errors="coerce",
                )
                parsed.loc[needs_second_attempt] = parsed_seconds

            invalid = series.notna() & parsed.isna()

            if invalid.any():
                raise ValueError(
                    f"{table_name}.{column}: "
                    f"{int(invalid.sum())} horas inválidas."
                )

            converted[column] = parsed.dt.time

        elif data_type == "boolean":
            converted[column] = parse_boolean(
                series,
                f"{table_name}.{column}",
            )

        elif data_type in {
            "smallint",
            "integer",
            "bigint",
        }:
            numeric = pd.to_numeric(series, errors="coerce")
            invalid = series.notna() & numeric.isna()

            if invalid.any():
                raise ValueError(
                    f"{table_name}.{column}: "
                    f"{int(invalid.sum())} enteros inválidos."
                )

            non_integer = numeric.dropna() % 1 != 0
            if non_integer.any():
                raise ValueError(
                    f"{table_name}.{column}: hay valores decimales "
                    "en una columna entera."
                )

            converted[column] = numeric.astype("Int64")

        elif data_type in {
            "numeric",
            "real",
            "double precision",
        }:
            numeric = pd.to_numeric(series, errors="coerce")
            invalid = series.notna() & numeric.isna()

            if invalid.any():
                raise ValueError(
                    f"{table_name}.{column}: "
                    f"{int(invalid.sum())} números inválidos."
                )

            converted[column] = numeric

        else:
            # Convierte NaN/NA en None para columnas de texto.
            converted[column] = (
                series.astype("object")
                .where(series.notna(), None)
            )

    return converted


def confirm_replace(
    local_tables: dict[str, pd.DataFrame],
) -> None:
    print("\nSe reemplazará el contenido de estas tablas:\n")

    for config in TABLES:
        count = len(local_tables[config.table_name])
        print(f"  - {config.table_name}: {count} filas")

    answer = input(
        "\nEscribe CARGAR para continuar: "
    ).strip()

    if answer != "CARGAR":
        raise KeyboardInterrupt(
            "Carga cancelada por el usuario."
        )


def dataframe_to_records(
    dataframe: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    Convierte NaN, NaT y pd.NA en None.

    Los objetos date, time y Timestamp se conservan para que SQLAlchemy
    los adapte usando los tipos reales reflejados desde PostgreSQL.
    """
    clean = dataframe.astype(object).where(
        pd.notna(dataframe),
        None,
    )
    return clean.to_dict(orient="records")


def upload_tables(
    engine: Engine,
    local_tables: dict[str, pd.DataFrame],
) -> dict[str, int]:
    """
    Reemplaza las nueve tablas dentro de una única transacción.

    Las tablas se reflejan desde PostgreSQL y se insertan por lotes.
    Esto respeta los tipos DATE, TIME y TIMESTAMP y evita construir
    una sentencia INSERT enorme con miles de parámetros.
    """
    inserted_counts: dict[str, int] = {}
    batch_size = 100

    with engine.begin() as connection:
        connection.execute(
            text(
                """
                select pg_advisory_xact_lock(
                    hashtext('sleep_dashboard_loader')
                )
                """
            )
        )

        identity = connection.execute(
            text(
                """
                select
                    current_database() as database_name,
                    current_user as connected_user
                """
            )
        ).mappings().one()

        print(
            "\nConexión establecida:"
            f"\n  Base de datos: {identity['database_name']}"
            f"\n  Usuario:       {identity['connected_user']}"
        )

        converted_tables: dict[str, pd.DataFrame] = {}
        reflected_tables: dict[str, Table] = {}
        metadata = MetaData()

        # Validar y convertir todas las tablas antes de truncar.
        for config in TABLES:
            remote_columns = get_remote_columns(
                connection,
                config.table_name,
            )

            converted_tables[config.table_name] = (
                convert_dataframe_for_postgres(
                    local_tables[config.table_name],
                    remote_columns,
                    config.table_name,
                )
            )

            reflected_tables[config.table_name] = Table(
                config.table_name,
                metadata,
                schema="public",
                autoload_with=connection,
            )

        table_list = ", ".join(
            f"public.{config.table_name}"
            for config in TABLES
        )

        connection.execute(
            text(f"truncate table {table_list}")
        )

        for config in TABLES:
            dataframe = converted_tables[config.table_name]
            table = reflected_tables[config.table_name]
            records = dataframe_to_records(dataframe)

            print(
                f"Cargando {config.table_name}: "
                f"{len(records)} filas..."
            )

            for batch_start in range(
                0,
                len(records),
                batch_size,
            ):
                batch = records[
                    batch_start : batch_start + batch_size
                ]

                if batch:
                    connection.execute(
                        table.insert(),
                        batch,
                    )

            remote_count = connection.execute(
                text(
                    f"select count(*) "
                    f"from public.{config.table_name}"
                )
            ).scalar_one()

            expected_count = len(dataframe)

            if remote_count != expected_count:
                raise RuntimeError(
                    f"{config.table_name}: se esperaban "
                    f"{expected_count} filas, pero hay "
                    f"{remote_count}."
                )

            inserted_counts[config.table_name] = int(
                remote_count
            )

    return inserted_counts


def print_local_summary(
    local_tables: dict[str, pd.DataFrame],
) -> None:
    print("\n" + "=" * 72)
    print("ARCHIVOS LOCALES VALIDADOS")
    print("=" * 72)

    for config in TABLES:
        dataframe = local_tables[config.table_name]
        print(
            f"{config.file_name:<38} "
            f"{len(dataframe):>6} filas · "
            f"{len(dataframe.columns):>2} columnas"
        )

    print("=" * 72)


def main() -> int:
    args = parse_args()

    try:
        local_tables = validate_local_files(args.data_dir)
        print_local_summary(local_tables)

        if args.check_files_only:
            print(
                "\nComprobación local terminada. "
                "No se realizó ninguna conexión."
            )
            return 0

        if not args.yes:
            confirm_replace(local_tables)

        engine = create_database_engine(args.env_file)

        try:
            inserted_counts = upload_tables(
                engine,
                local_tables,
            )
        finally:
            engine.dispose()

    except KeyboardInterrupt:
        print("\nProceso cancelado.")
        return 130
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    print("\n" + "=" * 72)
    print("CARGA COMPLETADA CORRECTAMENTE")
    print("=" * 72)

    for config in TABLES:
        print(
            f"{config.table_name:<38} "
            f"{inserted_counts[config.table_name]:>6} filas"
        )

    print("=" * 72)
    print(
        "La transacción se confirmó únicamente después de verificar "
        "las nueve tablas."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
