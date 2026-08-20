#!/usr/bin/env python3
"""
Auditoría inicial del dataset "My complete sleep data.csv".

Este script NO modifica el CSV original. Genera:
- audit_summary.json
- column_profile.csv
- data_dictionary.csv
- row_audit.csv
- anomalous_sessions.csv
- duplicate_bedtimes.csv

Uso:
    python audit_sleep_data.py "My complete sleep data.csv"
    python audit_sleep_data.py "My complete sleep data.csv" --output reports
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


EXPECTED_COLUMNS = [
    "Went to bed",
    "Woke up",
    "Sleep Quality",
    "Time in bed (seconds)",
    "Time asleep (seconds)",
    "Asleep after (seconds)",
    "Regularity",
    "Did snore",
    "Snore time (seconds)",
    "Not my snoring (seconds)",
    "Coughing (per hour)",
    "Steps",
    "Weather temperature (°C)",
    "Weather type",
    "Air Pressure (Pa)",
    "City",
    "Breathing disruptions (per hour)",
    "Ambient noise (dB)",
    "Ambient light (lux)",
    "Alertness score",
    "Alertness reaction time (seconds)",
    "Alertness accuracy",
    "Movements per hour",
    "Wake up window start",
    "Wake up window stop",
    "Notes",
]

# Los diferentes guiones se consideran valores ausentes.
NA_VALUES = ["", "—", "–", "-", "null", "NULL", "None", "N/A", "n/a"]

PERCENT_COLUMNS = {
    "Sleep Quality",
    "Regularity",
    "Alertness score",
    "Alertness accuracy",
}

DATETIME_COLUMNS = {
    "Went to bed",
    "Woke up",
    "Wake up window start",
    "Wake up window stop",
}

BOOLEAN_COLUMNS = {"Did snore"}

INTEGER_COLUMNS = {
    "Time in bed (seconds)",
    "Time asleep (seconds)",
    "Asleep after (seconds)",
    "Snore time (seconds)",
    "Not my snoring (seconds)",
    "Steps",
}

FLOAT_COLUMNS = {
    "Coughing (per hour)",
    "Weather temperature (°C)",
    "Air Pressure (Pa)",
    "Breathing disruptions (per hour)",
    "Ambient noise (dB)",
    "Ambient light (lux)",
    "Alertness reaction time (seconds)",
    "Movements per hour",
}


DATA_DICTIONARY = [
    {
        "original_column": "Went to bed",
        "normalized_name": "went_to_bed",
        "suggested_type": "datetime",
        "description": "Fecha y hora de inicio de la sesión.",
        "null_rule": "Obligatoria. Marcar la fila como error si no se puede interpretar.",
        "mvp": True,
    },
    {
        "original_column": "Woke up",
        "normalized_name": "woke_up",
        "suggested_type": "datetime",
        "description": "Fecha y hora de finalización de la sesión.",
        "null_rule": "Obligatoria. Fechas anteriores al año 2000 se consideran inválidas.",
        "mvp": True,
    },
    {
        "original_column": "Sleep Quality",
        "normalized_name": "sleep_quality_pct",
        "suggested_type": "float",
        "description": "Puntuación porcentual de calidad del sueño.",
        "null_rule": "Obligatoria. Debe estar entre 0 y 100.",
        "mvp": True,
    },
    {
        "original_column": "Time in bed (seconds)",
        "normalized_name": "time_in_bed_seconds",
        "suggested_type": "integer",
        "description": "Duración total de la sesión en segundos.",
        "null_rule": "Obligatoria. Debe ser mayor que cero para una sesión válida.",
        "mvp": True,
    },
    {
        "original_column": "Time asleep (seconds)",
        "normalized_name": "time_asleep_seconds",
        "suggested_type": "integer",
        "description": "Tiempo estimado dormido durante la sesión.",
        "null_rule": "Obligatoria. No debe superar el tiempo en cama.",
        "mvp": True,
    },
    {
        "original_column": "Asleep after (seconds)",
        "normalized_name": "asleep_after_seconds",
        "suggested_type": "integer",
        "description": "Latencia estimada hasta conciliar el sueño.",
        "null_rule": "Conservar como nulo si falta; no sustituir por cero.",
        "mvp": True,
    },
    {
        "original_column": "Regularity",
        "normalized_name": "regularity_pct",
        "suggested_type": "float",
        "description": "Puntuación porcentual de regularidad.",
        "null_rule": "Admite nulos. Los valores presentes deben estar entre 0 y 100.",
        "mvp": True,
    },
    {
        "original_column": "Did snore",
        "normalized_name": "did_snore",
        "suggested_type": "boolean",
        "description": "Indica si se detectaron ronquidos.",
        "null_rule": "Admite nulos; interpretar únicamente true/false.",
        "mvp": True,
    },
    {
        "original_column": "Snore time (seconds)",
        "normalized_name": "snore_time_seconds",
        "suggested_type": "integer",
        "description": "Tiempo de ronquido detectado.",
        "null_rule": "Admite nulos; cero significa que no se registró tiempo de ronquido.",
        "mvp": True,
    },
    {
        "original_column": "Not my snoring (seconds)",
        "normalized_name": "not_my_snoring_seconds",
        "suggested_type": "integer",
        "description": "Tiempo atribuido a ronquidos ajenos.",
        "null_rule": "Admite nulos.",
        "mvp": False,
    },
    {
        "original_column": "Coughing (per hour)",
        "normalized_name": "coughing_per_hour",
        "suggested_type": "float",
        "description": "Frecuencia estimada de tos por hora.",
        "null_rule": "Admite nulos; no sustituir por cero.",
        "mvp": True,
    },
    {
        "original_column": "Steps",
        "normalized_name": "steps",
        "suggested_type": "integer",
        "description": "Pasos registrados.",
        "null_rule": "Excluir inicialmente si permanece constante.",
        "mvp": False,
    },
    {
        "original_column": "Weather temperature (°C)",
        "normalized_name": "weather_temperature_c",
        "suggested_type": "float",
        "description": "Temperatura ambiental o meteorológica.",
        "null_rule": "Admite nulos.",
        "mvp": False,
    },
    {
        "original_column": "Weather type",
        "normalized_name": "weather_type",
        "suggested_type": "text",
        "description": "Descripción del tiempo meteorológico.",
        "null_rule": "Admite nulos.",
        "mvp": False,
    },
    {
        "original_column": "Air Pressure (Pa)",
        "normalized_name": "air_pressure_pa",
        "suggested_type": "float",
        "description": "Presión atmosférica registrada.",
        "null_rule": "Admite nulos; excluir del primer modelo si la cobertura es baja.",
        "mvp": False,
    },
    {
        "original_column": "City",
        "normalized_name": "city",
        "suggested_type": "text",
        "description": "Ciudad asociada a la sesión.",
        "null_rule": "Admite nulos.",
        "mvp": False,
    },
    {
        "original_column": "Breathing disruptions (per hour)",
        "normalized_name": "breathing_disruptions_per_hour",
        "suggested_type": "float",
        "description": "Interrupciones respiratorias estimadas por hora.",
        "null_rule": "Admite nulos. Uso únicamente descriptivo, no diagnóstico.",
        "mvp": True,
    },
    {
        "original_column": "Ambient noise (dB)",
        "normalized_name": "ambient_noise_db",
        "suggested_type": "float",
        "description": "Nivel de ruido ambiental.",
        "null_rule": "Admite nulos.",
        "mvp": True,
    },
    {
        "original_column": "Ambient light (lux)",
        "normalized_name": "ambient_light_lux",
        "suggested_type": "float",
        "description": "Nivel de luz ambiental.",
        "null_rule": "Admite nulos.",
        "mvp": True,
    },
    {
        "original_column": "Alertness score",
        "normalized_name": "alertness_score_pct",
        "suggested_type": "float",
        "description": "Puntuación de alerta.",
        "null_rule": "Excluir si la columna está completamente vacía.",
        "mvp": False,
    },
    {
        "original_column": "Alertness reaction time (seconds)",
        "normalized_name": "alertness_reaction_seconds",
        "suggested_type": "float",
        "description": "Tiempo de reacción de la prueba de alerta.",
        "null_rule": "Excluir si la columna está completamente vacía.",
        "mvp": False,
    },
    {
        "original_column": "Alertness accuracy",
        "normalized_name": "alertness_accuracy_pct",
        "suggested_type": "float",
        "description": "Precisión de la prueba de alerta.",
        "null_rule": "Excluir si la columna está completamente vacía.",
        "mvp": False,
    },
    {
        "original_column": "Movements per hour",
        "normalized_name": "movements_per_hour",
        "suggested_type": "float",
        "description": "Movimientos detectados por hora.",
        "null_rule": "Admite nulos.",
        "mvp": True,
    },
    {
        "original_column": "Wake up window start",
        "normalized_name": "wake_up_window_start",
        "suggested_type": "datetime",
        "description": "Inicio de la ventana configurada para despertarse.",
        "null_rule": "Admite nulos.",
        "mvp": False,
    },
    {
        "original_column": "Wake up window stop",
        "normalized_name": "wake_up_window_stop",
        "suggested_type": "datetime",
        "description": "Fin de la ventana configurada para despertarse.",
        "null_rule": "Admite nulos.",
        "mvp": False,
    },
    {
        "original_column": "Notes",
        "normalized_name": "notes",
        "suggested_type": "text",
        "description": "Etiquetas o notas personales asociadas a la sesión.",
        "null_rule": "Admite nulos; se dividirá en etiquetas en una fase posterior.",
        "mvp": True,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audita el CSV de sueño sin modificar el archivo original."
    )
    parser.add_argument("csv_path", type=Path, help="Ruta del archivo CSV.")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("reports"),
        help="Carpeta de salida. Valor predeterminado: reports",
    )
    parser.add_argument(
        "--separator",
        default=";",
        help="Separador del CSV. Valor predeterminado: ';'",
    )
    return parser.parse_args()


def parse_percentage(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype("string").str.strip().str.removesuffix("%"),
        errors="coerce",
    )


def parse_boolean(series: pd.Series) -> pd.Series:
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
    return normalized.map(mapping).astype("boolean")


def suggested_type(column: str) -> str:
    if column in PERCENT_COLUMNS:
        return "percentage"
    if column in DATETIME_COLUMNS:
        return "datetime"
    if column in BOOLEAN_COLUMNS:
        return "boolean"
    if column in INTEGER_COLUMNS:
        return "integer"
    if column in FLOAT_COLUMNS:
        return "float"
    return "text"


def convert_for_profile(df: pd.DataFrame, column: str) -> pd.Series | None:
    if column in PERCENT_COLUMNS:
        return parse_percentage(df[column])
    if column in INTEGER_COLUMNS or column in FLOAT_COLUMNS:
        return pd.to_numeric(df[column], errors="coerce")
    if column in DATETIME_COLUMNS:
        return pd.to_datetime(df[column], errors="coerce", format="mixed")
    return None


def json_safe(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    if pd.isna(value):
        return None
    return value


def profile_columns(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for column in df.columns:
        series = df[column]
        non_null = series.dropna()
        converted = convert_for_profile(df, column)

        samples = [
            str(value)[:80]
            for value in non_null.drop_duplicates().head(3).tolist()
        ]

        row: dict[str, Any] = {
            "column": column,
            "suggested_type": suggested_type(column),
            "total_rows": len(df),
            "non_null_count": int(series.notna().sum()),
            "missing_count": int(series.isna().sum()),
            "missing_pct": round(float(series.isna().mean() * 100), 2),
            "unique_non_null": int(non_null.nunique(dropna=True)),
            "sample_values": " | ".join(samples),
            "conversion_failures": None,
            "minimum": None,
            "mean": None,
            "median": None,
            "maximum": None,
        }

        if converted is not None:
            source_present = series.notna()
            conversion_failures = source_present & converted.isna()
            row["conversion_failures"] = int(conversion_failures.sum())

            valid = converted.dropna()
            if not valid.empty:
                if pd.api.types.is_datetime64_any_dtype(valid):
                    row["minimum"] = valid.min().isoformat()
                    row["maximum"] = valid.max().isoformat()
                else:
                    row["minimum"] = float(valid.min())
                    row["mean"] = round(float(valid.mean()), 4)
                    row["median"] = round(float(valid.median()), 4)
                    row["maximum"] = float(valid.max())

        rows.append(row)

    return pd.DataFrame(rows)


def build_row_audit(df: pd.DataFrame) -> pd.DataFrame:
    audit = pd.DataFrame(index=df.index)

    # +2: una línea para la cabecera y la numeración humana comienza en 1.
    audit["csv_row_number"] = df.index + 2
    audit["went_to_bed_raw"] = df["Went to bed"]
    audit["woke_up_raw"] = df["Woke up"]

    bed = pd.to_datetime(df["Went to bed"], errors="coerce", format="mixed")
    wake = pd.to_datetime(df["Woke up"], errors="coerce", format="mixed")

    sleep_quality = parse_percentage(df["Sleep Quality"])
    regularity = parse_percentage(df["Regularity"])

    time_in_bed = pd.to_numeric(df["Time in bed (seconds)"], errors="coerce")
    time_asleep = pd.to_numeric(df["Time asleep (seconds)"], errors="coerce")
    asleep_after = pd.to_numeric(df["Asleep after (seconds)"], errors="coerce")

    clock_duration = (wake - bed).dt.total_seconds()
    duration_difference = (clock_duration - time_in_bed).abs()

    audit["went_to_bed_parsed"] = bed
    audit["woke_up_parsed"] = wake
    audit["sleep_quality_pct"] = sleep_quality
    audit["regularity_pct"] = regularity
    audit["time_in_bed_seconds"] = time_in_bed
    audit["time_asleep_seconds"] = time_asleep
    audit["asleep_after_seconds"] = asleep_after
    audit["clock_duration_seconds"] = clock_duration
    audit["duration_difference_seconds"] = duration_difference

    duplicate_bedtime = (
        df["Went to bed"].notna()
        & df["Went to bed"].duplicated(keep=False)
    )

    error_flags: list[list[str]] = [[] for _ in range(len(df))]
    warning_flags: list[list[str]] = [[] for _ in range(len(df))]

    def add_flag(mask: pd.Series, flag: str, target: list[list[str]]) -> None:
        positions = mask.fillna(False).to_numpy().nonzero()[0]
        for position in positions:
            target[position].append(flag)

    add_flag(bed.isna(), "invalid_bedtime", error_flags)
    add_flag(wake.isna(), "invalid_wake_time", error_flags)
    add_flag(wake.notna() & (wake.dt.year < 2000), "invalid_wake_year", error_flags)
    add_flag(
        bed.notna() & wake.notna() & (wake >= pd.Timestamp("2000-01-01")) & (wake < bed),
        "wake_before_bed",
        error_flags,
    )

    add_flag(time_in_bed.isna(), "invalid_time_in_bed", error_flags)
    add_flag(time_in_bed.notna() & (time_in_bed <= 0), "nonpositive_time_in_bed", error_flags)
    add_flag(time_asleep.isna(), "invalid_time_asleep", error_flags)
    add_flag(time_asleep.notna() & (time_asleep < 0), "negative_time_asleep", error_flags)
    add_flag(
        time_in_bed.notna()
        & time_asleep.notna()
        & (time_in_bed > 0)
        & (time_asleep > time_in_bed),
        "time_asleep_exceeds_time_in_bed",
        error_flags,
    )

    add_flag(
        sleep_quality.isna() | ~sleep_quality.between(0, 100, inclusive="both"),
        "invalid_sleep_quality",
        error_flags,
    )
    add_flag(
        df["Regularity"].notna()
        & (regularity.isna() | ~regularity.between(0, 100, inclusive="both")),
        "invalid_regularity",
        error_flags,
    )

    add_flag(
        time_in_bed.notna() & time_in_bed.between(1, 7199, inclusive="both"),
        "very_short_session_under_2h",
        warning_flags,
    )
    add_flag(
        time_in_bed.notna() & (time_in_bed > 57600),
        "very_long_session_over_16h",
        warning_flags,
    )
    add_flag(
        time_asleep.notna() & (time_asleep == 0),
        "zero_time_asleep",
        warning_flags,
    )
    add_flag(duplicate_bedtime, "duplicate_bedtime", warning_flags)

    valid_date_pair = (
        bed.notna()
        & wake.notna()
        & (wake >= pd.Timestamp("2000-01-01"))
        & (wake >= bed)
        & time_in_bed.notna()
    )

    possible_dst = (
        valid_date_pair
        & duration_difference.between(3300, 3900, inclusive="both")
    )
    duration_mismatch = (
        valid_date_pair
        & (duration_difference > 300)
        & ~possible_dst
    )

    add_flag(possible_dst, "possible_dst_transition", warning_flags)
    add_flag(duration_mismatch, "duration_mismatch_over_5min", warning_flags)

    audit["error_flags"] = ["|".join(flags) for flags in error_flags]
    audit["warning_flags"] = ["|".join(flags) for flags in warning_flags]
    audit["has_error"] = audit["error_flags"].ne("")
    audit["has_warning"] = audit["warning_flags"].ne("")
    audit["is_clean"] = ~audit["has_error"] & ~audit["has_warning"]

    # Incluimos campos útiles para revisar la fila sin abrir el CSV.
    for column in [
        "Did snore",
        "Snore time (seconds)",
        "Coughing (per hour)",
        "Breathing disruptions (per hour)",
        "Ambient noise (dB)",
        "Ambient light (lux)",
        "Movements per hour",
        "Notes",
    ]:
        if column in df.columns:
            audit[column] = df[column]

    return audit


def build_summary(
    csv_path: Path,
    df: pd.DataFrame,
    profile: pd.DataFrame,
    row_audit: pd.DataFrame,
    missing_columns: list[str],
    unexpected_columns: list[str],
) -> dict[str, Any]:
    bed = pd.to_datetime(df.get("Went to bed"), errors="coerce", format="mixed")
    wake = pd.to_datetime(df.get("Woke up"), errors="coerce", format="mixed")
    valid_wake = wake[wake.dt.year >= 2000]

    all_flags: Counter[str] = Counter()
    for column in ("error_flags", "warning_flags"):
        for flags in row_audit[column]:
            if flags:
                all_flags.update(flags.split("|"))

    constant_columns = []
    empty_columns = []
    for column in df.columns:
        unique_count = int(df[column].nunique(dropna=True))
        if int(df[column].notna().sum()) == 0:
            empty_columns.append(column)
        elif unique_count <= 1:
            constant_columns.append(column)

    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "source_file": str(csv_path.resolve()),
        "file_size_bytes": csv_path.stat().st_size,
        "separator": ";",
        "rows": len(df),
        "columns": len(df.columns),
        "expected_columns": len(EXPECTED_COLUMNS),
        "missing_expected_columns": missing_columns,
        "unexpected_columns": unexpected_columns,
        "full_duplicate_rows": int(df.duplicated().sum()),
        "duplicate_bedtime_rows": int((
            df["Went to bed"].notna()
            & df["Went to bed"].duplicated(keep=False)
        ).sum()),
        "date_range": {
            "first_bedtime": json_safe(bed.min()),
            "last_bedtime": json_safe(bed.max()),
            "first_valid_wake_time": json_safe(valid_wake.min()),
            "last_valid_wake_time": json_safe(valid_wake.max()),
        },
        "row_quality": {
            "rows_with_errors": int(row_audit["has_error"].sum()),
            "rows_with_warnings": int(row_audit["has_warning"].sum()),
            "clean_rows": int(row_audit["is_clean"].sum()),
        },
        "anomaly_counts": dict(sorted(all_flags.items())),
        "empty_columns": empty_columns,
        "constant_columns": constant_columns,
        "missing_values_by_column": {
            str(row["column"]): int(row["missing_count"])
            for _, row in profile.iterrows()
        },
    }


def save_outputs(
    output_dir: Path,
    df: pd.DataFrame,
    profile: pd.DataFrame,
    row_audit: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    profile.to_csv(
        output_dir / "column_profile.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(DATA_DICTIONARY).to_csv(
        output_dir / "data_dictionary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    row_audit.to_csv(
        output_dir / "row_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    row_audit.loc[
        row_audit["has_error"] | row_audit["has_warning"]
    ].to_csv(
        output_dir / "anomalous_sessions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    duplicate_mask = (
        df["Went to bed"].notna()
        & df["Went to bed"].duplicated(keep=False)
    )
    duplicate_rows = df.loc[duplicate_mask].copy()
    duplicate_rows.insert(0, "csv_row_number", duplicate_rows.index + 2)
    duplicate_rows.to_csv(
        output_dir / "duplicate_bedtimes.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with (output_dir / "audit_summary.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, ensure_ascii=False, indent=2, default=json_safe)


def print_summary(summary: dict[str, Any], output_dir: Path) -> None:
    quality = summary["row_quality"]

    print("\n" + "=" * 64)
    print("AUDITORÍA DEL DATASET DE SUEÑO")
    print("=" * 64)
    print(f"Archivo: {summary['source_file']}")
    print(f"Dimensiones: {summary['rows']} filas × {summary['columns']} columnas")
    print(
        "Rango de fechas: "
        f"{summary['date_range']['first_bedtime']} "
        f"→ {summary['date_range']['last_bedtime']}"
    )
    print(f"Filas limpias: {quality['clean_rows']}")
    print(f"Filas con advertencias: {quality['rows_with_warnings']}")
    print(f"Filas con errores: {quality['rows_with_errors']}")
    print(f"Duplicados completos: {summary['full_duplicate_rows']}")
    print(f"Filas con hora de inicio duplicada: {summary['duplicate_bedtime_rows']}")

    print("\nIncidencias detectadas:")
    if summary["anomaly_counts"]:
        for name, count in summary["anomaly_counts"].items():
            print(f"  - {name}: {count}")
    else:
        print("  No se detectaron incidencias.")

    if summary["empty_columns"]:
        print("\nColumnas completamente vacías:")
        for column in summary["empty_columns"]:
            print(f"  - {column}")

    if summary["constant_columns"]:
        print("\nColumnas constantes:")
        for column in summary["constant_columns"]:
            print(f"  - {column}")

    print(f"\nInformes guardados en: {output_dir.resolve()}")
    print("=" * 64)


def main() -> int:
    args = parse_args()
    csv_path: Path = args.csv_path
    output_dir: Path = args.output

    if not csv_path.exists():
        print(f"ERROR: No existe el archivo: {csv_path}", file=sys.stderr)
        return 1

    if not csv_path.is_file():
        print(f"ERROR: La ruta no es un archivo: {csv_path}", file=sys.stderr)
        return 1

    try:
        df = pd.read_csv(
            csv_path,
            sep=args.separator,
            encoding="utf-8-sig",
            dtype="string",
            na_values=NA_VALUES,
            keep_default_na=True,
        )
    except Exception as exc:
        print(f"ERROR al leer el CSV: {exc}", file=sys.stderr)
        return 1

    missing_columns = [
        column for column in EXPECTED_COLUMNS if column not in df.columns
    ]
    unexpected_columns = [
        column for column in df.columns if column not in EXPECTED_COLUMNS
    ]

    if missing_columns:
        print(
            "ERROR: faltan columnas necesarias:\n  - "
            + "\n  - ".join(missing_columns),
            file=sys.stderr,
        )
        return 2

    profile = profile_columns(df)
    row_audit = build_row_audit(df)
    summary = build_summary(
        csv_path=csv_path,
        df=df,
        profile=profile,
        row_audit=row_audit,
        missing_columns=missing_columns,
        unexpected_columns=unexpected_columns,
    )

    save_outputs(
        output_dir=output_dir,
        df=df,
        profile=profile,
        row_audit=row_audit,
        summary=summary,
    )
    print_summary(summary, output_dir)

    # Código de salida 0: la auditoría se ejecutó correctamente,
    # aunque el dataset pueda contener filas con errores.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
