#!/usr/bin/env python3
"""
Limpia y transforma el dataset "My complete sleep data.csv".

El CSV original nunca se modifica.

Salidas:
- data/processed/sleep_sessions_clean.csv
- reports/cleaning_summary.json
- reports/rejected_sessions.csv
- reports/excluded_from_analytics.csv
- reports/processed_column_dictionary.csv

Uso:
    python src/clean_sleep_data.py \
        "data/raw/My complete sleep data.csv" \
        --processed-dir data/processed \
        --reports-dir reports
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
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

NA_VALUES = ["", "—", "–", "-", "null", "NULL", "None", "N/A", "n/a"]

DROP_COLUMNS = [
    "Steps",
    "Alertness score",
    "Alertness reaction time (seconds)",
    "Alertness accuracy",
]

SPANISH_WEEKDAYS = {
    0: "lunes",
    1: "martes",
    2: "miércoles",
    3: "jueves",
    4: "viernes",
    5: "sábado",
    6: "domingo",
}

SPANISH_MONTHS = {
    1: "enero",
    2: "febrero",
    3: "marzo",
    4: "abril",
    5: "mayo",
    6: "junio",
    7: "julio",
    8: "agosto",
    9: "septiembre",
    10: "octubre",
    11: "noviembre",
    12: "diciembre",
}


PROCESSED_DICTIONARY = [
    ("session_key", "text", "Identificador estable generado a partir de la fila original."),
    ("source_row_number", "integer", "Número de fila en el CSV original, contando la cabecera."),
    ("went_to_bed", "datetime", "Fecha y hora local de inicio de la sesión."),
    ("woke_up", "datetime", "Fecha y hora local de finalización de la sesión."),
    ("sleep_date", "date", "Fecha asignada a la sesión; corresponde al día de despertar."),
    ("bed_date", "date", "Fecha de inicio de la sesión."),
    ("wake_date", "date", "Fecha de finalización de la sesión."),
    ("bedtime", "time", "Hora local de acostarse."),
    ("wake_time", "time", "Hora local de despertarse."),
    ("sleep_quality_pct", "float", "Calidad del sueño entre 0 y 100."),
    ("regularity_pct", "float", "Regularidad entre 0 y 100."),
    ("time_in_bed_seconds", "integer", "Tiempo total en cama en segundos."),
    ("time_in_bed_hours", "float", "Tiempo total en cama en horas."),
    ("time_asleep_seconds", "integer", "Tiempo estimado dormido en segundos."),
    ("time_asleep_hours", "float", "Tiempo estimado dormido en horas."),
    ("sleep_latency_seconds", "integer", "Tiempo estimado hasta conciliar el sueño."),
    ("sleep_latency_minutes", "float", "Latencia del sueño en minutos."),
    ("awake_time_seconds", "integer", "Diferencia entre tiempo en cama y tiempo dormido."),
    ("awake_time_minutes", "float", "Tiempo no dormido durante la sesión en minutos."),
    ("sleep_efficiency_pct", "float", "Tiempo dormido dividido por tiempo en cama, en porcentaje."),
    ("did_snore", "boolean", "Indica si se detectaron ronquidos."),
    ("snore_time_seconds", "integer", "Tiempo de ronquido detectado en segundos."),
    ("snore_time_minutes", "float", "Tiempo de ronquido detectado en minutos."),
    ("snore_percentage", "float", "Porcentaje del tiempo en cama con ronquidos detectados."),
    ("not_my_snoring_seconds", "integer", "Tiempo atribuido a ronquidos ajenos."),
    ("coughing_per_hour", "float", "Frecuencia de tos estimada por hora."),
    ("breathing_disruptions_per_hour", "float", "Interrupciones respiratorias estimadas por hora."),
    ("ambient_noise_db", "float", "Ruido ambiental registrado."),
    ("ambient_light_lux", "float", "Luz ambiental registrada."),
    ("movements_per_hour", "float", "Movimientos detectados por hora."),
    ("weather_temperature_c", "float", "Temperatura meteorológica disponible."),
    ("weather_type", "text", "Descripción meteorológica disponible."),
    ("air_pressure_pa", "float", "Presión atmosférica disponible."),
    ("city", "text", "Ciudad disponible en el registro."),
    ("wake_up_window_start", "datetime", "Inicio de la ventana configurada para despertarse."),
    ("wake_up_window_stop", "datetime", "Fin de la ventana configurada para despertarse."),
    ("notes", "text", "Notas o etiquetas personales originales."),
    ("has_notes", "boolean", "Indica si la sesión contiene notas."),
    ("year", "integer", "Año de la fecha de sueño."),
    ("month", "integer", "Mes de la fecha de sueño."),
    ("month_name", "text", "Nombre del mes en español."),
    ("week_of_year", "integer", "Semana ISO del año."),
    ("day_of_week_num", "integer", "Día ISO de la semana: lunes=1 y domingo=7."),
    ("day_of_week", "text", "Nombre del día de la semana en español."),
    ("is_weekend", "boolean", "Indica sábado o domingo."),
    ("bedtime_minutes_clock", "integer", "Minutos desde medianoche de la hora de acostarse."),
    ("bedtime_minutes_adjusted", "integer", "Hora de acostarse ajustada para ordenar horas posteriores a medianoche."),
    ("wake_time_minutes_clock", "integer", "Minutos desde medianoche de la hora de despertarse."),
    ("clock_duration_seconds", "float", "Duración calculada restando las marcas de fecha y hora."),
    ("duration_difference_seconds", "float", "Diferencia absoluta entre duración de reloj y tiempo en cama."),
    ("possible_dst_transition", "boolean", "Posible diferencia de una hora por cambio horario."),
    ("duplicate_bedtime_original", "boolean", "La hora de inicio se repetía en el CSV original."),
    ("data_quality_flags", "text", "Advertencias de calidad separadas por barra vertical."),
    ("session_status", "text", "valid, failed_sleep_detection o unusually_long."),
    ("include_in_analytics", "boolean", "Indica si la sesión entra en métricas y modelos iniciales."),
    ("exclusion_reason", "text", "Motivo por el que la sesión se excluye del análisis."),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Limpia el dataset de sueño sin modificar el CSV original."
    )
    parser.add_argument("csv_path", type=Path, help="Ruta del CSV original.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Carpeta de salida del CSV limpio.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
        help="Carpeta de salida de los informes.",
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


def parse_main_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(
        series,
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )


def parse_wake_window(series: pd.Series) -> pd.Series:
    # El archivo utiliza año de dos cifras: 25-05-31 07:50:00.
    return pd.to_datetime(
        series,
        format="%y-%m-%d %H:%M:%S",
        errors="coerce",
    )


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def make_session_key(row_number: int, bedtime: Any, wake_time: Any) -> str:
    raw = f"{row_number}|{bedtime}|{wake_time}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def json_safe(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    if pd.isna(value):
        return None
    return value


def load_source(csv_path: Path) -> pd.DataFrame:
    try:
        source = pd.read_csv(
            csv_path,
            sep=";",
            encoding="utf-8-sig",
            dtype="string",
            na_values=NA_VALUES,
            keep_default_na=True,
        )
    except Exception as exc:
        raise RuntimeError(f"No se pudo leer el CSV: {exc}") from exc

    missing = [column for column in EXPECTED_COLUMNS if column not in source.columns]
    unexpected = [column for column in source.columns if column not in EXPECTED_COLUMNS]

    if missing:
        raise ValueError(
            "Faltan columnas obligatorias:\n- " + "\n- ".join(missing)
        )

    if unexpected:
        print(
            "ADVERTENCIA: se encontraron columnas inesperadas: "
            + ", ".join(unexpected),
            file=sys.stderr,
        )

    return source


def build_transformed(source: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = source.copy()
    source["source_row_number"] = source.index + 2

    bed = parse_main_datetime(source["Went to bed"])
    wake = parse_main_datetime(source["Woke up"])

    sleep_quality = parse_percentage(source["Sleep Quality"])
    regularity = parse_percentage(source["Regularity"])

    time_in_bed = to_numeric(source["Time in bed (seconds)"])
    time_asleep = to_numeric(source["Time asleep (seconds)"])
    sleep_latency = to_numeric(source["Asleep after (seconds)"])

    snore_time = to_numeric(source["Snore time (seconds)"])
    not_my_snoring = to_numeric(source["Not my snoring (seconds)"])

    clock_duration = (wake - bed).dt.total_seconds()
    duration_difference = (clock_duration - time_in_bed).abs()

    duplicate_bedtime = (
        source["Went to bed"].notna()
        & source["Went to bed"].duplicated(keep=False)
    )

    invalid_reasons = pd.Series("", index=source.index, dtype="string")

    def add_reason(mask: pd.Series, reason: str) -> None:
        nonlocal invalid_reasons
        mask = mask.fillna(False)
        invalid_reasons.loc[mask] = invalid_reasons.loc[mask].apply(
            lambda current: reason if not current else f"{current}|{reason}"
        )

    add_reason(bed.isna(), "invalid_bedtime")
    add_reason(wake.isna(), "invalid_wake_time")
    add_reason(wake.notna() & (wake.dt.year < 2000), "invalid_wake_year")
    add_reason(
        bed.notna()
        & wake.notna()
        & (wake >= pd.Timestamp("2000-01-01"))
        & (wake < bed),
        "wake_before_bed",
    )
    add_reason(time_in_bed.isna(), "invalid_time_in_bed")
    add_reason(time_in_bed.notna() & (time_in_bed <= 0), "nonpositive_time_in_bed")
    add_reason(time_asleep.isna(), "invalid_time_asleep")
    add_reason(time_asleep.notna() & (time_asleep < 0), "negative_time_asleep")
    add_reason(
        time_in_bed.notna()
        & time_asleep.notna()
        & (time_in_bed > 0)
        & (time_asleep > time_in_bed),
        "time_asleep_exceeds_time_in_bed",
    )
    add_reason(
        sleep_quality.isna()
        | ~sleep_quality.between(0, 100, inclusive="both"),
        "invalid_sleep_quality",
    )
    add_reason(
        source["Regularity"].notna()
        & (
            regularity.isna()
            | ~regularity.between(0, 100, inclusive="both")
        ),
        "invalid_regularity",
    )

    invalid_mask = invalid_reasons.ne("")

    failed_detection = (
        ~invalid_mask
        & time_in_bed.between(1, 7199, inclusive="both")
        & time_asleep.eq(0)
    )
    unusually_long = ~invalid_mask & time_in_bed.gt(57600)

    session_status = pd.Series("valid", index=source.index, dtype="string")
    session_status.loc[invalid_mask] = "invalid_record"
    session_status.loc[failed_detection] = "failed_sleep_detection"
    session_status.loc[unusually_long] = "unusually_long"

    include_in_analytics = session_status.eq("valid")

    exclusion_reason = pd.Series(pd.NA, index=source.index, dtype="string")
    exclusion_reason.loc[invalid_mask] = invalid_reasons.loc[invalid_mask]
    exclusion_reason.loc[failed_detection] = (
        "sesión inferior a 2 horas sin sueño detectado"
    )
    exclusion_reason.loc[unusually_long] = (
        "sesión superior a 16 horas; revisar manualmente"
    )

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
        & duration_difference.gt(300)
        & ~possible_dst
    )

    quality_flags: list[str] = []
    for idx in source.index:
        flags: list[str] = []
        if bool(duplicate_bedtime.loc[idx]):
            flags.append("duplicate_bedtime_original")
        if bool(possible_dst.loc[idx]):
            flags.append("possible_dst_transition")
        if bool(duration_mismatch.loc[idx]):
            flags.append("duration_mismatch_over_5min")
        if session_status.loc[idx] == "failed_sleep_detection":
            flags.append("zero_sleep_detected")
        if session_status.loc[idx] == "unusually_long":
            flags.append("very_long_session_over_16h")
        quality_flags.append("|".join(flags))

    awake_time = time_in_bed - time_asleep
    sleep_efficiency = np.where(
        time_in_bed.gt(0),
        (time_asleep / time_in_bed) * 100,
        np.nan,
    )
    snore_percentage = np.where(
        time_in_bed.gt(0),
        (snore_time / time_in_bed) * 100,
        np.nan,
    )

    sleep_date = wake.dt.normalize()
    iso_calendar = sleep_date.dt.isocalendar()

    bedtime_minutes_clock = bed.dt.hour * 60 + bed.dt.minute
    bedtime_minutes_adjusted = bedtime_minutes_clock.where(
        bedtime_minutes_clock >= 720,
        bedtime_minutes_clock + 1440,
    )
    wake_time_minutes_clock = wake.dt.hour * 60 + wake.dt.minute

    transformed = pd.DataFrame(
        {
            "session_key": [
                make_session_key(
                    int(row_number),
                    source.loc[idx, "Went to bed"],
                    source.loc[idx, "Woke up"],
                )
                for idx, row_number in source["source_row_number"].items()
            ],
            "source_row_number": source["source_row_number"].astype("Int64"),
            "went_to_bed": bed,
            "woke_up": wake,
            "sleep_date": sleep_date.dt.date,
            "bed_date": bed.dt.date,
            "wake_date": wake.dt.date,
            "bedtime": bed.dt.strftime("%H:%M:%S"),
            "wake_time": wake.dt.strftime("%H:%M:%S"),
            "sleep_quality_pct": sleep_quality.round(2),
            "regularity_pct": regularity.round(2),
            "time_in_bed_seconds": time_in_bed.round().astype("Int64"),
            "time_in_bed_hours": (time_in_bed / 3600).round(3),
            "time_asleep_seconds": time_asleep.round().astype("Int64"),
            "time_asleep_hours": (time_asleep / 3600).round(3),
            "sleep_latency_seconds": sleep_latency.round().astype("Int64"),
            "sleep_latency_minutes": (sleep_latency / 60).round(2),
            "awake_time_seconds": awake_time.round().astype("Int64"),
            "awake_time_minutes": (awake_time / 60).round(2),
            "sleep_efficiency_pct": pd.Series(sleep_efficiency).round(2),
            "did_snore": parse_boolean(source["Did snore"]),
            "snore_time_seconds": snore_time.round().astype("Int64"),
            "snore_time_minutes": (snore_time / 60).round(2),
            "snore_percentage": pd.Series(snore_percentage).round(3),
            "not_my_snoring_seconds": not_my_snoring.round().astype("Int64"),
            "coughing_per_hour": to_numeric(source["Coughing (per hour)"]).round(4),
            "breathing_disruptions_per_hour": to_numeric(
                source["Breathing disruptions (per hour)"]
            ).round(4),
            "ambient_noise_db": to_numeric(source["Ambient noise (dB)"]).round(4),
            "ambient_light_lux": to_numeric(source["Ambient light (lux)"]).round(4),
            "movements_per_hour": to_numeric(source["Movements per hour"]).round(4),
            "weather_temperature_c": to_numeric(
                source["Weather temperature (°C)"]
            ).round(2),
            "weather_type": source["Weather type"].str.strip(),
            "air_pressure_pa": to_numeric(source["Air Pressure (Pa)"]).round(2),
            "city": source["City"].str.strip(),
            "wake_up_window_start": parse_wake_window(
                source["Wake up window start"]
            ),
            "wake_up_window_stop": parse_wake_window(
                source["Wake up window stop"]
            ),
            "notes": source["Notes"].str.strip(),
            "has_notes": source["Notes"].notna(),
            "year": sleep_date.dt.year.astype("Int64"),
            "month": sleep_date.dt.month.astype("Int64"),
            "month_name": sleep_date.dt.month.map(SPANISH_MONTHS),
            "week_of_year": iso_calendar.week.astype("Int64"),
            "day_of_week_num": (sleep_date.dt.dayofweek + 1).astype("Int64"),
            "day_of_week": sleep_date.dt.dayofweek.map(SPANISH_WEEKDAYS),
            "is_weekend": sleep_date.dt.dayofweek.ge(5).astype("boolean"),
            "bedtime_minutes_clock": bedtime_minutes_clock.astype("Int64"),
            "bedtime_minutes_adjusted": bedtime_minutes_adjusted.astype("Int64"),
            "wake_time_minutes_clock": wake_time_minutes_clock.astype("Int64"),
            "clock_duration_seconds": clock_duration.round(0),
            "duration_difference_seconds": duration_difference.round(0),
            "possible_dst_transition": possible_dst.astype("boolean"),
            "duplicate_bedtime_original": duplicate_bedtime.astype("boolean"),
            "data_quality_flags": pd.Series(quality_flags, dtype="string").replace("", pd.NA),
            "session_status": session_status,
            "include_in_analytics": include_in_analytics.astype("boolean"),
            "exclusion_reason": exclusion_reason,
        }
    )

    rejected = transformed.loc[invalid_mask].copy()

    # El dataset limpio no contiene registros estructuralmente inválidos.
    cleaned = transformed.loc[~invalid_mask].copy()
    cleaned = cleaned.sort_values(
        ["went_to_bed", "source_row_number"],
        kind="stable",
    ).reset_index(drop=True)

    return cleaned, rejected


def build_summary(
    source_path: Path,
    source: pd.DataFrame,
    cleaned: pd.DataFrame,
    rejected: pd.DataFrame,
) -> dict[str, Any]:
    status_counts = {
        str(key): int(value)
        for key, value in cleaned["session_status"].value_counts().items()
    }

    eligible = cleaned["include_in_analytics"].fillna(False)

    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "source_file": str(source_path.resolve()),
        "source_rows": int(len(source)),
        "cleaned_rows": int(len(cleaned)),
        "rejected_rows": int(len(rejected)),
        "analytics_rows": int(eligible.sum()),
        "excluded_from_analytics_rows": int((~eligible).sum()),
        "status_counts": status_counts,
        "date_range_cleaned": {
            "first_bedtime": json_safe(cleaned["went_to_bed"].min()),
            "last_bedtime": json_safe(cleaned["went_to_bed"].max()),
            "first_sleep_date": json_safe(cleaned["sleep_date"].min()),
            "last_sleep_date": json_safe(cleaned["sleep_date"].max()),
        },
        "analytics_metrics_preview": {
            "mean_sleep_quality_pct": round(
                float(cleaned.loc[eligible, "sleep_quality_pct"].mean()), 2
            ),
            "mean_time_asleep_hours": round(
                float(cleaned.loc[eligible, "time_asleep_hours"].mean()), 2
            ),
            "mean_time_in_bed_hours": round(
                float(cleaned.loc[eligible, "time_in_bed_hours"].mean()), 2
            ),
            "mean_sleep_efficiency_pct": round(
                float(cleaned.loc[eligible, "sleep_efficiency_pct"].mean()), 2
            ),
            "mean_sleep_latency_minutes": round(
                float(cleaned.loc[eligible, "sleep_latency_minutes"].mean()), 2
            ),
            "mean_regularity_pct": round(
                float(cleaned.loc[eligible, "regularity_pct"].mean()), 2
            ),
        },
        "cleaning_rules": {
            "invalid_records": (
                "Eliminados del CSV limpio y guardados en rejected_sessions.csv."
            ),
            "failed_sleep_detection": (
                "Conservados, pero include_in_analytics=false."
            ),
            "unusually_long": (
                "Conservados, pero include_in_analytics=false."
            ),
            "possible_dst_transition": (
                "Conservados como válidos; se usa time_in_bed_seconds como duración."
            ),
            "missing_values": (
                "Se mantienen como nulos; no se sustituyen por cero."
            ),
            "dropped_source_columns": DROP_COLUMNS,
            "sleep_date_definition": (
                "La fecha de sueño corresponde al día de despertar."
            ),
        },
    }


def save_outputs(
    processed_dir: Path,
    reports_dir: Path,
    cleaned: pd.DataFrame,
    rejected: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    cleaned_path = processed_dir / "sleep_sessions_clean.csv"
    rejected_path = reports_dir / "rejected_sessions.csv"
    excluded_path = reports_dir / "excluded_from_analytics.csv"
    summary_path = reports_dir / "cleaning_summary.json"
    dictionary_path = reports_dir / "processed_column_dictionary.csv"

    # Punto y coma para mantener compatibilidad con el CSV original y Excel en español.
    cleaned.to_csv(
        cleaned_path,
        sep=";",
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d %H:%M:%S",
    )

    rejected.to_csv(
        rejected_path,
        sep=";",
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d %H:%M:%S",
    )

    cleaned.loc[
        ~cleaned["include_in_analytics"].fillna(False)
    ].to_csv(
        excluded_path,
        sep=";",
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d %H:%M:%S",
    )

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
            default=json_safe,
        )

    dictionary = pd.DataFrame(
        PROCESSED_DICTIONARY,
        columns=["column", "type", "description"],
    )
    dictionary.to_csv(
        dictionary_path,
        sep=";",
        index=False,
        encoding="utf-8-sig",
    )


def print_summary(summary: dict[str, Any], processed_dir: Path, reports_dir: Path) -> None:
    metrics = summary["analytics_metrics_preview"]

    print("\n" + "=" * 68)
    print("LIMPIEZA DEL DATASET DE SUEÑO COMPLETADA")
    print("=" * 68)
    print(f"Filas originales:                {summary['source_rows']}")
    print(f"Filas en el dataset limpio:      {summary['cleaned_rows']}")
    print(f"Registros rechazados:            {summary['rejected_rows']}")
    print(f"Sesiones incluidas en análisis:  {summary['analytics_rows']}")
    print(
        "Sesiones excluidas del análisis: "
        f"{summary['excluded_from_analytics_rows']}"
    )

    print("\nEstados:")
    for status, count in summary["status_counts"].items():
        print(f"  - {status}: {count}")

    print("\nVista previa de indicadores (sesiones analíticas):")
    print(f"  - Calidad media:       {metrics['mean_sleep_quality_pct']} %")
    print(f"  - Tiempo dormido:      {metrics['mean_time_asleep_hours']} h")
    print(f"  - Tiempo en cama:      {metrics['mean_time_in_bed_hours']} h")
    print(f"  - Eficiencia media:    {metrics['mean_sleep_efficiency_pct']} %")
    print(f"  - Latencia media:      {metrics['mean_sleep_latency_minutes']} min")
    print(f"  - Regularidad media:   {metrics['mean_regularity_pct']} %")

    print(f"\nCSV limpio: {processed_dir.resolve() / 'sleep_sessions_clean.csv'}")
    print(f"Informes:   {reports_dir.resolve()}")
    print("=" * 68)


def main() -> int:
    args = parse_args()

    if not args.csv_path.exists():
        print(
            f"ERROR: no existe el archivo {args.csv_path}",
            file=sys.stderr,
        )
        return 1

    try:
        source = load_source(args.csv_path)
        cleaned, rejected = build_transformed(source)
        summary = build_summary(
            args.csv_path,
            source,
            cleaned,
            rejected,
        )
        save_outputs(
            args.processed_dir,
            args.reports_dir,
            cleaned,
            rejected,
            summary,
        )
        print_summary(
            summary,
            args.processed_dir,
            args.reports_dir,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
