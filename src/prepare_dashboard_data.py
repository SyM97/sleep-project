#!/usr/bin/env python3
"""
Prepara las tablas finales que utilizará el dashboard de Lovable/Supabase.

Entrada:
    data/processed/sleep_sessions_clean.csv

Salidas en data/dashboard:
    dashboard_kpis.csv
    dashboard_daily.csv
    dashboard_weekly.csv
    dashboard_monthly.csv
    dashboard_weekday.csv
    dashboard_note_tags.csv
    dashboard_quality_factors.csv
    dashboard_distributions.csv
    dashboard_recent_sessions.csv

Salida adicional:
    reports/dashboard/dashboard_build_summary.json

Características importantes:
- Solo usa sesiones con include_in_analytics=true.
- Mantiene todos los días del calendario en dashboard_daily.csv.
- Los días sin datos tienen has_data=false y métricas nulas.
- Las medias móviles son de 7 y 30 días naturales, no de sesiones.
- Los meses y semanas sin registros también aparecen.
- Marca los periodos inicial y final como parciales.
- No interpola ni inventa datos ausentes.
- La calidad y la regularidad usan media aritmética por sesión.
- Conserva fecha y hora completas en went_to_bed y woke_up.

Uso:
    python src/prepare_dashboard_data.py \
        data/processed/sleep_sessions_clean.csv \
        --output-dir data/dashboard \
        --reports-dir reports/dashboard
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SPANISH_WEEKDAYS = [
    "lunes",
    "martes",
    "miércoles",
    "jueves",
    "viernes",
    "sábado",
    "domingo",
]

SPANISH_MONTHS = [
    "enero",
    "febrero",
    "marzo",
    "abril",
    "mayo",
    "junio",
    "julio",
    "agosto",
    "septiembre",
    "octubre",
    "noviembre",
    "diciembre",
]

REQUIRED_COLUMNS = {
    "sleep_date",
    "went_to_bed",
    "woke_up",
    "sleep_quality_pct",
    "time_asleep_hours",
    "time_in_bed_hours",
    "sleep_latency_minutes",
    "sleep_efficiency_pct",
    "regularity_pct",
    "movements_per_hour",
    "ambient_noise_db",
    "ambient_light_lux",
    "coughing_per_hour",
    "breathing_disruptions_per_hour",
    "snore_percentage",
    "include_in_analytics",
    "session_status",
}

NUMERIC_COLUMNS = [
    "sleep_quality_pct",
    "time_asleep_seconds",
    "time_asleep_hours",
    "time_in_bed_seconds",
    "time_in_bed_hours",
    "sleep_latency_seconds",
    "sleep_latency_minutes",
    "awake_time_seconds",
    "awake_time_minutes",
    "sleep_efficiency_pct",
    "regularity_pct",
    "movements_per_hour",
    "ambient_noise_db",
    "ambient_light_lux",
    "coughing_per_hour",
    "breathing_disruptions_per_hour",
    "snore_time_seconds",
    "snore_time_minutes",
    "snore_percentage",
    "weather_temperature_c",
    "bedtime_minutes_adjusted",
    "wake_time_minutes_clock",
]

QUALITY_FACTOR_COLUMNS = [
    "time_asleep_hours",
    "time_in_bed_hours",
    "sleep_latency_minutes",
    "sleep_efficiency_pct",
    "regularity_pct",
    "movements_per_hour",
    "ambient_noise_db",
    "ambient_light_lux",
    "coughing_per_hour",
    "breathing_disruptions_per_hour",
    "snore_percentage",
    "weather_temperature_c",
]

METRIC_LABELS = {
    "time_asleep_hours": "Horas dormidas",
    "time_in_bed_hours": "Horas en cama",
    "sleep_latency_minutes": "Latencia del sueño",
    "sleep_efficiency_pct": "Eficiencia del sueño",
    "regularity_pct": "Regularidad",
    "movements_per_hour": "Movimientos por hora",
    "ambient_noise_db": "Ruido ambiental",
    "ambient_light_lux": "Luz ambiental",
    "coughing_per_hour": "Tos por hora",
    "breathing_disruptions_per_hour": "Interrupciones respiratorias",
    "snore_percentage": "Porcentaje de ronquido",
    "weather_temperature_c": "Temperatura",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepara las tablas CSV para el dashboard de sueño."
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Ruta de sleep_sessions_clean.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/dashboard"),
        help="Carpeta para las tablas del dashboard.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports/dashboard"),
        help="Carpeta para el resumen de construcción.",
    )
    return parser.parse_args()


def normalize_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    normalized = series.astype("string").str.strip().str.lower()
    return (
        normalized.map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
                "yes": True,
                "no": False,
                "sí": True,
                "si": True,
            }
        )
        .fillna(False)
        .astype(bool)
    )


def read_clean_data(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        df = pd.read_csv(
            csv_path,
            sep=";",
            encoding="utf-8-sig",
            low_memory=False,
        )
    except Exception as exc:
        raise RuntimeError(f"No se pudo leer el CSV: {exc}") from exc

    if len(df.columns) == 1:
        raise RuntimeError(
            "El CSV se leyó como una sola columna. "
            "Comprueba que sea el archivo limpio y que use ';' como separador."
        )

    missing = sorted(REQUIRED_COLUMNS.difference(df.columns))
    if missing:
        raise ValueError(
            "Faltan columnas necesarias:\n  - " + "\n  - ".join(missing)
        )

    prepared = df.copy()

    for column in ["sleep_date", "went_to_bed", "woke_up"]:
        prepared[column] = pd.to_datetime(
            prepared[column],
            errors="coerce",
        )

    for column in NUMERIC_COLUMNS:
        if column in prepared.columns:
            prepared[column] = pd.to_numeric(
                prepared[column],
                errors="coerce",
            )

    prepared["include_in_analytics"] = normalize_boolean(
        prepared["include_in_analytics"]
    )

    analytics = prepared.loc[
        prepared["include_in_analytics"]
        & prepared["sleep_date"].notna()
    ].copy()

    analytics = analytics.sort_values(
        ["sleep_date", "went_to_bed"]
    ).reset_index(drop=True)

    if analytics.empty:
        raise ValueError(
            "No hay sesiones con include_in_analytics=true."
        )

    return prepared, analytics


def safe_round(value: Any, digits: int = 2) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def weighted_mean(
    df: pd.DataFrame,
    value_column: str,
    weight_column: str,
) -> float | None:
    if value_column not in df.columns or weight_column not in df.columns:
        return None

    valid = (
        df[value_column].notna()
        & df[weight_column].notna()
        & (df[weight_column] > 0)
    )

    if valid.any():
        return float(
            np.average(
                df.loc[valid, value_column],
                weights=df.loc[valid, weight_column],
            )
        )

    fallback = df[value_column].dropna()
    if fallback.empty:
        return None

    return float(fallback.mean())


def median_clock_string(
    values: pd.Series,
    adjusted: bool = False,
) -> str | None:
    valid = pd.to_numeric(values, errors="coerce").dropna()
    if valid.empty:
        return None

    minutes = float(valid.median())
    if adjusted:
        minutes = minutes % 1440

    hours = int(minutes // 60) % 24
    mins = int(round(minutes % 60))

    if mins == 60:
        hours = (hours + 1) % 24
        mins = 0

    return f"{hours:02d}:{mins:02d}"


def aggregate_sessions(df: pd.DataFrame) -> dict[str, Any]:
    """
    Resume un grupo de sesiones. Las métricas sensibles a la duración
    se ponderan por horas dormidas o tiempo en cama.
    """
    if df.empty:
        return {
            "sessions_count": 0,
            "days_with_data": 0,
            "mean_sleep_quality_pct": None,
            "total_time_asleep_hours": None,
            "mean_time_asleep_hours_per_session": None,
            "mean_time_asleep_hours_per_recorded_day": None,
            "mean_time_in_bed_hours_per_session": None,
            "overall_sleep_efficiency_pct": None,
            "mean_sleep_latency_minutes": None,
            "mean_regularity_pct": None,
            "mean_movements_per_hour": None,
            "mean_ambient_noise_db": None,
            "mean_ambient_light_lux": None,
            "mean_coughing_per_hour": None,
            "mean_breathing_disruptions_per_hour": None,
            "overall_snore_percentage": None,
            "median_bedtime": None,
            "median_wake_time": None,
        }

    total_asleep = df["time_asleep_hours"].sum(min_count=1)
    total_in_bed = df["time_in_bed_hours"].sum(min_count=1)

    efficiency = None
    if (
        pd.notna(total_asleep)
        and pd.notna(total_in_bed)
        and total_in_bed > 0
    ):
        efficiency = total_asleep / total_in_bed * 100

    snore_percentage = None
    if (
        "snore_time_seconds" in df.columns
        and "time_asleep_seconds" in df.columns
    ):
        snore_seconds = df["snore_time_seconds"].sum(min_count=1)
        asleep_seconds = df["time_asleep_seconds"].sum(min_count=1)
        if (
            pd.notna(snore_seconds)
            and pd.notna(asleep_seconds)
            and asleep_seconds > 0
        ):
            snore_percentage = snore_seconds / asleep_seconds * 100

    days_with_data = int(df["sleep_date"].dt.normalize().nunique())

    bedtime = None
    if "bedtime_minutes_adjusted" in df.columns:
        bedtime = median_clock_string(
            df["bedtime_minutes_adjusted"],
            adjusted=True,
        )

    wake_time = None
    if "wake_time_minutes_clock" in df.columns:
        wake_time = median_clock_string(
            df["wake_time_minutes_clock"],
            adjusted=False,
        )

    return {
        "sessions_count": int(len(df)),
        "days_with_data": days_with_data,
        "mean_sleep_quality_pct": safe_round(
            df["sleep_quality_pct"].mean()
        ),
        "total_time_asleep_hours": safe_round(total_asleep),
        "mean_time_asleep_hours_per_session": safe_round(
            df["time_asleep_hours"].mean()
        ),
        "mean_time_asleep_hours_per_recorded_day": safe_round(
            total_asleep / days_with_data
            if pd.notna(total_asleep) and days_with_data > 0
            else None
        ),
        "mean_time_in_bed_hours_per_session": safe_round(
            df["time_in_bed_hours"].mean()
        ),
        "overall_sleep_efficiency_pct": safe_round(efficiency),
        "mean_sleep_latency_minutes": safe_round(
            df["sleep_latency_minutes"].mean()
        ),
        "mean_regularity_pct": safe_round(
            df["regularity_pct"].mean()
        ),
        "mean_movements_per_hour": safe_round(
            weighted_mean(
                df,
                "movements_per_hour",
                "time_asleep_hours",
            )
        ),
        "mean_ambient_noise_db": safe_round(
            weighted_mean(
                df,
                "ambient_noise_db",
                "time_in_bed_hours",
            )
        ),
        "mean_ambient_light_lux": safe_round(
            weighted_mean(
                df,
                "ambient_light_lux",
                "time_in_bed_hours",
            )
        ),
        "mean_coughing_per_hour": safe_round(
            weighted_mean(
                df,
                "coughing_per_hour",
                "time_asleep_hours",
            )
        ),
        "mean_breathing_disruptions_per_hour": safe_round(
            weighted_mean(
                df,
                "breathing_disruptions_per_hour",
                "time_asleep_hours",
            )
        ),
        "overall_snore_percentage": safe_round(snore_percentage),
        "median_bedtime": bedtime,
        "median_wake_time": wake_time,
    }


def build_daily(analytics: pd.DataFrame) -> pd.DataFrame:
    first_date = analytics["sleep_date"].min().normalize()
    last_date = analytics["sleep_date"].max().normalize()

    rows: list[dict[str, Any]] = []

    for sleep_date, group in analytics.groupby(
        analytics["sleep_date"].dt.normalize()
    ):
        metrics = aggregate_sessions(group)
        rows.append(
            {
                "sleep_date": sleep_date,
                "has_data": True,
                **metrics,
            }
        )

    observed = pd.DataFrame(rows)

    calendar = pd.DataFrame(
        {
            "sleep_date": pd.date_range(
                first_date,
                last_date,
                freq="D",
            )
        }
    )

    daily = calendar.merge(
        observed,
        on="sleep_date",
        how="left",
    )

    daily["has_data"] = daily["has_data"].eq(True)
    daily["sessions_count"] = (
        daily["sessions_count"].fillna(0).astype(int)
    )
    daily["days_with_data"] = (
        daily["days_with_data"].fillna(0).astype(int)
    )

    daily["year"] = daily["sleep_date"].dt.year
    daily["month"] = daily["sleep_date"].dt.month
    daily["month_label"] = daily.apply(
        lambda row: (
            f"{SPANISH_MONTHS[int(row['month']) - 1]} "
            f"{int(row['year'])}"
        ),
        axis=1,
    )
    daily["day_of_week_num"] = daily["sleep_date"].dt.weekday
    daily["day_of_week"] = daily["day_of_week_num"].map(
        dict(enumerate(SPANISH_WEEKDAYS))
    )
    daily["is_weekend"] = daily["day_of_week_num"] >= 5
    daily["is_gap_day"] = ~daily["has_data"]

    iso = daily["sleep_date"].dt.isocalendar()
    daily["iso_year"] = iso["year"].astype(int)
    daily["iso_week"] = iso["week"].astype(int)

    # Medias de días naturales. Los nulos no se rellenan ni interpolan.
    daily["observations_in_7d"] = (
        daily["has_data"].astype(int).rolling(7, min_periods=1).sum().astype(int)
    )
    daily["observations_in_30d"] = (
        daily["has_data"].astype(int).rolling(30, min_periods=1).sum().astype(int)
    )

    daily["sleep_quality_rolling_7d"] = (
        daily["mean_sleep_quality_pct"]
        .rolling(7, min_periods=3)
        .mean()
        .round(2)
    )
    daily["sleep_quality_rolling_30d"] = (
        daily["mean_sleep_quality_pct"]
        .rolling(30, min_periods=10)
        .mean()
        .round(2)
    )
    daily["time_asleep_rolling_7d"] = (
        daily["total_time_asleep_hours"]
        .rolling(7, min_periods=3)
        .mean()
        .round(2)
    )
    daily["efficiency_rolling_7d"] = (
        daily["overall_sleep_efficiency_pct"]
        .rolling(7, min_periods=3)
        .mean()
        .round(2)
    )

    ordered = [
        "sleep_date",
        "has_data",
        "is_gap_day",
        "sessions_count",
        "year",
        "month",
        "month_label",
        "iso_year",
        "iso_week",
        "day_of_week_num",
        "day_of_week",
        "is_weekend",
        "mean_sleep_quality_pct",
        "sleep_quality_rolling_7d",
        "sleep_quality_rolling_30d",
        "total_time_asleep_hours",
        "time_asleep_rolling_7d",
        "mean_time_in_bed_hours_per_session",
        "overall_sleep_efficiency_pct",
        "efficiency_rolling_7d",
        "mean_sleep_latency_minutes",
        "mean_regularity_pct",
        "mean_movements_per_hour",
        "mean_ambient_noise_db",
        "mean_ambient_light_lux",
        "mean_coughing_per_hour",
        "mean_breathing_disruptions_per_hour",
        "overall_snore_percentage",
        "median_bedtime",
        "median_wake_time",
        "observations_in_7d",
        "observations_in_30d",
    ]

    return daily[ordered]


def period_row(
    group: pd.DataFrame,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    first_date: pd.Timestamp,
    last_date: pd.Timestamp,
) -> dict[str, Any]:
    covered_start = max(period_start, first_date)
    covered_end = min(period_end, last_date)
    expected_days = int((covered_end - covered_start).days + 1)

    metrics = aggregate_sessions(group)
    days_with_data = metrics["days_with_data"]

    coverage = (
        days_with_data / expected_days * 100
        if expected_days > 0
        else None
    )

    return {
        "period_start": period_start,
        "period_end": period_end,
        "covered_start": covered_start,
        "covered_end": covered_end,
        "expected_calendar_days": expected_days,
        "has_data": not group.empty,
        "is_partial_period": (
            covered_start > period_start
            or covered_end < period_end
        ),
        "coverage_pct": safe_round(coverage),
        **metrics,
    }


def build_weekly(analytics: pd.DataFrame) -> pd.DataFrame:
    # Normalizamos explícitamente a pandas.Timestamp y usamos una
    # unidad temporal específica ("D") para evitar el DeprecationWarning
    # de NumPy sobre unidades timedelta genéricas.
    first_date = pd.Timestamp(
        analytics["sleep_date"].min()
    ).normalize()
    last_date = pd.Timestamp(
        analytics["sleep_date"].max()
    ).normalize()

    first_week_start = first_date - pd.Timedelta(
        int(first_date.weekday()),
        unit="D",
    )
    last_week_start = last_date - pd.Timedelta(
        int(last_date.weekday()),
        unit="D",
    )

    rows: list[dict[str, Any]] = []

    for week_start in pd.date_range(
        first_week_start,
        last_week_start,
        freq="7D",
    ):
        week_start = pd.Timestamp(week_start).normalize()
        week_end = week_start + pd.Timedelta(
            6,
            unit="D",
        )
        mask = (
            analytics["sleep_date"].dt.normalize().between(
                week_start,
                week_end,
                inclusive="both",
            )
        )
        group = analytics.loc[mask]

        base = period_row(
            group,
            week_start,
            week_end,
            first_date,
            last_date,
        )
        iso = week_start.isocalendar()

        rows.append(
            {
                "week_start": week_start,
                "week_end": week_end,
                "iso_year": int(iso.year),
                "iso_week": int(iso.week),
                "week_label": f"{int(iso.year)}-W{int(iso.week):02d}",
                **{
                    key: value
                    for key, value in base.items()
                    if key not in {"period_start", "period_end"}
                },
            }
        )

    return pd.DataFrame(rows)


def build_monthly(analytics: pd.DataFrame) -> pd.DataFrame:
    first_date = analytics["sleep_date"].min().normalize()
    last_date = analytics["sleep_date"].max().normalize()

    first_month = first_date.to_period("M")
    last_month = last_date.to_period("M")

    rows: list[dict[str, Any]] = []

    for period in pd.period_range(
        first_month,
        last_month,
        freq="M",
    ):
        month_start = period.start_time.normalize()
        month_end = period.end_time.normalize()

        mask = (
            analytics["sleep_date"].dt.normalize().between(
                month_start,
                month_end,
                inclusive="both",
            )
        )
        group = analytics.loc[mask]

        base = period_row(
            group,
            month_start,
            month_end,
            first_date,
            last_date,
        )

        rows.append(
            {
                "month_start": month_start,
                "month_end": month_end,
                "year": int(period.year),
                "month": int(period.month),
                "month_key": f"{period.year}-{period.month:02d}",
                "month_label": (
                    f"{SPANISH_MONTHS[period.month - 1]} "
                    f"{period.year}"
                ),
                **{
                    key: value
                    for key, value in base.items()
                    if key not in {"period_start", "period_end"}
                },
            }
        )

    return pd.DataFrame(rows)


def build_weekday(analytics: pd.DataFrame) -> pd.DataFrame:
    data = analytics.copy()
    data["weekday_num"] = data["sleep_date"].dt.weekday

    rows: list[dict[str, Any]] = []

    for weekday_num, weekday_name in enumerate(SPANISH_WEEKDAYS):
        group = data.loc[data["weekday_num"] == weekday_num]
        rows.append(
            {
                "day_of_week_num": weekday_num,
                "day_of_week": weekday_name,
                **aggregate_sessions(group),
            }
        )

    return pd.DataFrame(rows)


def split_note_tags(value: Any) -> list[str]:
    if value is None or pd.isna(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    tags: list[str] = []

    for raw_tag in text.replace(";", ":").split(":"):
        display = " ".join(raw_tag.strip().split())
        if display:
            tags.append(display)

    # Elimina duplicados de la misma sesión sin cambiar el orden.
    return list(dict.fromkeys(tags))


def slugify_tag(value: str) -> str:
    translations = str.maketrans(
        {
            "á": "a",
            "é": "e",
            "í": "i",
            "ó": "o",
            "ú": "u",
            "ü": "u",
            "ñ": "n",
        }
    )
    normalized = value.lower().translate(translations)
    return "_".join(
        "".join(
            character
            if character.isalnum() or character.isspace()
            else " "
            for character in normalized
        ).split()
    )


def build_note_tags(analytics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "tag_key",
        "tag_label",
        "sessions_count",
        "sessions_pct",
        "mean_sleep_quality_pct",
        "mean_time_asleep_hours",
        "mean_sleep_efficiency_pct",
        "quality_difference_vs_other_sessions",
        "duration_difference_vs_other_sessions",
    ]

    if "notes" not in analytics.columns:
        return pd.DataFrame(columns=columns)

    tags_by_index = analytics["notes"].apply(split_note_tags)

    display_by_key: dict[str, str] = {}
    indices_by_key: dict[str, set[int]] = {}

    for index, tags in tags_by_index.items():
        for tag in tags:
            key = slugify_tag(tag)
            if not key:
                continue
            display_by_key.setdefault(key, tag)
            indices_by_key.setdefault(key, set()).add(index)

    rows: list[dict[str, Any]] = []

    for key, indices in indices_by_key.items():
        mask = analytics.index.isin(indices)
        tagged = analytics.loc[mask]
        other = analytics.loc[~mask]

        rows.append(
            {
                "tag_key": key,
                "tag_label": display_by_key[key],
                "sessions_count": int(len(tagged)),
                "sessions_pct": safe_round(
                    len(tagged) / len(analytics) * 100
                ),
                "mean_sleep_quality_pct": safe_round(
                    tagged["sleep_quality_pct"].mean()
                ),
                "mean_time_asleep_hours": safe_round(
                    tagged["time_asleep_hours"].mean()
                ),
                "mean_sleep_efficiency_pct": safe_round(
                    tagged["sleep_efficiency_pct"].mean()
                ),
                "quality_difference_vs_other_sessions": safe_round(
                    tagged["sleep_quality_pct"].mean()
                    - other["sleep_quality_pct"].mean()
                    if not other.empty
                    else None
                ),
                "duration_difference_vs_other_sessions": safe_round(
                    tagged["time_asleep_hours"].mean()
                    - other["time_asleep_hours"].mean()
                    if not other.empty
                    else None
                ),
            }
        )

    if not rows:
        return pd.DataFrame(columns=columns)

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["sessions_count", "tag_label"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )


def correlation_strength(value: float) -> str:
    absolute = abs(value)
    if absolute < 0.10:
        return "prácticamente nula"
    if absolute < 0.30:
        return "débil"
    if absolute < 0.50:
        return "moderada"
    if absolute < 0.70:
        return "fuerte"
    return "muy fuerte"


def build_quality_factors(analytics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for column in QUALITY_FACTOR_COLUMNS:
        if column not in analytics.columns:
            continue

        pair = analytics[
            ["sleep_quality_pct", column]
        ].dropna()

        if len(pair) < 20 or pair[column].nunique() <= 1:
            continue

        correlation = float(
            pair["sleep_quality_pct"].corr(pair[column])
        )

        if pd.isna(correlation):
            continue

        if correlation >= 0.10:
            direction = "positiva"
        elif correlation <= -0.10:
            direction = "negativa"
        else:
            direction = "sin dirección clara"

        rows.append(
            {
                "metric_key": column,
                "metric_label": METRIC_LABELS.get(column, column),
                "correlation_with_quality": round(correlation, 3),
                "absolute_correlation": round(abs(correlation), 3),
                "direction": direction,
                "strength": correlation_strength(correlation),
                "observations": int(len(pair)),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "absolute_correlation",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def distribution_rows(
    series: pd.Series,
    dimension_key: str,
    dimension_label: str,
    bins: list[float],
    labels: list[str],
) -> list[dict[str, Any]]:
    valid = pd.to_numeric(series, errors="coerce").dropna()
    categories = pd.cut(
        valid,
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    )
    counts = categories.value_counts(sort=False)

    rows: list[dict[str, Any]] = []

    for order, label in enumerate(labels, start=1):
        count = int(counts.get(label, 0))
        rows.append(
            {
                "dimension_key": dimension_key,
                "dimension_label": dimension_label,
                "bucket_order": order,
                "bucket_label": label,
                "sessions_count": count,
                "sessions_pct": safe_round(
                    count / len(valid) * 100
                    if len(valid) > 0
                    else None
                ),
            }
        )

    return rows


def build_distributions(analytics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    rows.extend(
        distribution_rows(
            analytics["sleep_quality_pct"],
            "sleep_quality",
            "Calidad del sueño",
            [-np.inf, 70, 85, np.inf],
            ["Menos de 70 %", "70–84 %", "85 % o más"],
        )
    )
    rows.extend(
        distribution_rows(
            analytics["time_asleep_hours"],
            "sleep_duration",
            "Duración del sueño",
            [-np.inf, 6, 7, 9, np.inf],
            [
                "Menos de 6 h",
                "6–6,99 h",
                "7–8,99 h",
                "9 h o más",
            ],
        )
    )
    rows.extend(
        distribution_rows(
            analytics["sleep_efficiency_pct"],
            "sleep_efficiency",
            "Eficiencia del sueño",
            [-np.inf, 80, 85, 90, np.inf],
            [
                "Menos de 80 %",
                "80–84 %",
                "85–89 %",
                "90 % o más",
            ],
        )
    )

    return pd.DataFrame(rows)


def build_recent_sessions(
    analytics: pd.DataFrame,
    limit: int = 100,
) -> pd.DataFrame:
    preferred = [
        "session_key",
        "source_row_number",
        "sleep_date",
        "went_to_bed",
        "woke_up",
        "sleep_quality_pct",
        "time_asleep_hours",
        "time_in_bed_hours",
        "sleep_efficiency_pct",
        "sleep_latency_minutes",
        "regularity_pct",
        "movements_per_hour",
        "ambient_noise_db",
        "ambient_light_lux",
        "coughing_per_hour",
        "breathing_disruptions_per_hour",
        "snore_percentage",
        "did_snore",
        "notes",
        "possible_dst_transition",
        "data_quality_flags",
    ]

    columns = [
        column for column in preferred if column in analytics.columns
    ]

    return (
        analytics.sort_values(
            ["sleep_date", "went_to_bed"],
            ascending=[False, False],
        )
        .head(limit)[columns]
        .reset_index(drop=True)
    )


def build_kpis(
    all_rows: pd.DataFrame,
    analytics: pd.DataFrame,
    daily: pd.DataFrame,
    monthly: pd.DataFrame,
    weekday: pd.DataFrame,
) -> pd.DataFrame:
    first_date = analytics["sleep_date"].min().normalize()
    last_date = analytics["sleep_date"].max().normalize()
    calendar_days = int((last_date - first_date).days + 1)
    days_with_data = int(
        analytics["sleep_date"].dt.normalize().nunique()
    )

    recent_7 = analytics.tail(7)
    previous_7 = analytics.iloc[-14:-7] if len(analytics) >= 14 else pd.DataFrame()

    recent_quality = recent_7["sleep_quality_pct"].mean()
    previous_quality = (
        previous_7["sleep_quality_pct"].mean()
        if not previous_7.empty
        else None
    )
    recent_duration = recent_7["time_asleep_hours"].mean()
    previous_duration = (
        previous_7["time_asleep_hours"].mean()
        if not previous_7.empty
        else None
    )

    eligible_months = monthly.loc[
        monthly["has_data"]
        & (monthly["sessions_count"] >= 10)
    ]
    best_month = (
        eligible_months.sort_values(
            "mean_sleep_quality_pct",
            ascending=False,
        ).iloc[0]
        if not eligible_months.empty
        else None
    )

    best_weekday = (
        weekday.sort_values(
            "mean_sleep_quality_pct",
            ascending=False,
        ).iloc[0]
        if not weekday.empty
        else None
    )

    duration_7_9 = analytics["time_asleep_hours"].between(
        7,
        9,
        inclusive="both",
    )
    quality_high = analytics["sleep_quality_pct"] >= 85
    quality_low = analytics["sleep_quality_pct"] < 70
    efficiency_high = analytics["sleep_efficiency_pct"] >= 85

    row = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "first_sleep_date": first_date.date().isoformat(),
        "last_sleep_date": last_date.date().isoformat(),
        "cleaned_sessions_count": int(len(all_rows)),
        "analytics_sessions_count": int(len(analytics)),
        "excluded_sessions_count": int(
            (~all_rows["include_in_analytics"]).sum()
        ),
        "calendar_days_covered": calendar_days,
        "days_with_data": days_with_data,
        "recording_coverage_pct": safe_round(
            days_with_data / calendar_days * 100
        ),
        "mean_sleep_quality_pct": safe_round(
            analytics["sleep_quality_pct"].mean()
        ),
        "median_sleep_quality_pct": safe_round(
            analytics["sleep_quality_pct"].median()
        ),
        "mean_time_asleep_hours": safe_round(
            analytics["time_asleep_hours"].mean()
        ),
        "mean_time_in_bed_hours": safe_round(
            analytics["time_in_bed_hours"].mean()
        ),
        "mean_sleep_efficiency_pct": safe_round(
            analytics["sleep_efficiency_pct"].mean()
        ),
        "mean_sleep_latency_minutes": safe_round(
            analytics["sleep_latency_minutes"].mean()
        ),
        "mean_regularity_pct": safe_round(
            analytics["regularity_pct"].mean()
        ),
        "sessions_7_to_9_hours_count": int(duration_7_9.sum()),
        "sessions_7_to_9_hours_pct": safe_round(
            duration_7_9.mean() * 100
        ),
        "sessions_quality_85_plus_count": int(quality_high.sum()),
        "sessions_quality_85_plus_pct": safe_round(
            quality_high.mean() * 100
        ),
        "sessions_quality_below_70_count": int(quality_low.sum()),
        "sessions_quality_below_70_pct": safe_round(
            quality_low.mean() * 100
        ),
        "sessions_efficiency_85_plus_count": int(
            efficiency_high.sum()
        ),
        "sessions_efficiency_85_plus_pct": safe_round(
            efficiency_high.mean() * 100
        ),
        "recent_7_sessions_quality_pct": safe_round(recent_quality),
        "previous_7_sessions_quality_pct": safe_round(previous_quality),
        "quality_change_vs_previous_7_pp": safe_round(
            recent_quality - previous_quality
            if previous_quality is not None
            else None
        ),
        "recent_7_sessions_duration_hours": safe_round(recent_duration),
        "previous_7_sessions_duration_hours": safe_round(
            previous_duration
        ),
        "duration_change_vs_previous_7_hours": safe_round(
            recent_duration - previous_duration
            if previous_duration is not None
            else None
        ),
        "best_month_key_min_10_sessions": (
            best_month["month_key"]
            if best_month is not None
            else None
        ),
        "best_month_label_min_10_sessions": (
            best_month["month_label"]
            if best_month is not None
            else None
        ),
        "best_month_quality_pct_min_10_sessions": (
            safe_round(best_month["mean_sleep_quality_pct"])
            if best_month is not None
            else None
        ),
        "best_weekday": (
            best_weekday["day_of_week"]
            if best_weekday is not None
            else None
        ),
        "best_weekday_quality_pct": (
            safe_round(best_weekday["mean_sleep_quality_pct"])
            if best_weekday is not None
            else None
        ),
    }

    return pd.DataFrame([row])


def save_csv(df: pd.DataFrame, path: Path) -> None:
    """
    Exporta fechas de calendario como YYYY-MM-DD y conserva la hora
    completa de los timestamps de las sesiones.
    """
    output = df.copy()

    timestamp_columns = {
        "went_to_bed",
        "woke_up",
        "generated_at",
    }

    date_columns = {
        "sleep_date",
        "first_sleep_date",
        "last_sleep_date",
        "week_start",
        "week_end",
        "month_start",
        "month_end",
        "covered_start",
        "covered_end",
    }

    for column in output.columns:
        if column in timestamp_columns:
            parsed = pd.to_datetime(output[column], errors="coerce")
            output[column] = parsed.dt.strftime("%Y-%m-%d %H:%M:%S")
        elif column in date_columns:
            parsed = pd.to_datetime(output[column], errors="coerce")
            output[column] = parsed.dt.strftime("%Y-%m-%d")
        elif pd.api.types.is_datetime64_any_dtype(output[column]):
            output[column] = output[column].dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            )

    output.to_csv(
        path,
        sep=";",
        index=False,
        encoding="utf-8-sig",
        na_rep="",
    )


def build_summary(
    source_path: Path,
    all_rows: pd.DataFrame,
    analytics: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "source_file": str(source_path.resolve()),
        "source_cleaned_rows": int(len(all_rows)),
        "analytics_rows": int(len(analytics)),
        "excluded_rows": int(
            (~all_rows["include_in_analytics"]).sum()
        ),
        "date_range": {
            "first_sleep_date": (
                analytics["sleep_date"].min().date().isoformat()
            ),
            "last_sleep_date": (
                analytics["sleep_date"].max().date().isoformat()
            ),
        },
        "generated_tables": {
            name: {
                "rows": int(len(table)),
                "columns": int(len(table.columns)),
            }
            for name, table in tables.items()
        },
        "definitions": {
            "sleep_date": "Día de despertar asociado a la sesión.",
            "daily_table": (
                "Incluye todos los días naturales del intervalo. "
                "Los días sin registro tienen has_data=false y métricas nulas."
            ),
            "daily_rolling_windows": (
                "Las medias móviles usan 7 y 30 días naturales. "
                "No se interpolan valores ausentes."
            ),
            "monthly_and_weekly_coverage": (
                "coverage_pct = días con datos / días esperados dentro "
                "del periodo cubierto por el dataset."
            ),
            "partial_period": (
                "is_partial_period=true cuando el dataset comienza o termina "
                "a mitad de una semana o un mes."
            ),
            "daily_multiple_sessions": (
                "Si hay varias sesiones en un día, se suman las horas y "
                "la calidad se pondera por tiempo dormido."
            ),
            "quality_factors": (
                "Correlaciones de Pearson descriptivas; no demuestran causalidad."
            ),
        },
        "recommended_frontend_rules": {
            "timeline": (
                "Usar dashboard_daily y mantener connectNulls=false "
                "para no unir periodos sin datos."
            ),
            "monthly_chart": (
                "Mostrar sessions_count, coverage_pct e is_partial_period "
                "en el tooltip."
            ),
            "empty_periods": (
                "Mostrar explícitamente semanas o meses con has_data=false."
            ),
        },
    }


def main() -> int:
    args = parse_args()

    if not args.csv_path.exists():
        print(
            f"ERROR: no existe {args.csv_path}",
            file=sys.stderr,
        )
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    try:
        all_rows, analytics = read_clean_data(args.csv_path)

        daily = build_daily(analytics)
        weekly = build_weekly(analytics)
        monthly = build_monthly(analytics)
        weekday = build_weekday(analytics)
        note_tags = build_note_tags(analytics)
        quality_factors = build_quality_factors(analytics)
        distributions = build_distributions(analytics)
        recent_sessions = build_recent_sessions(analytics)

        kpis = build_kpis(
            all_rows,
            analytics,
            daily,
            monthly,
            weekday,
        )

        tables = {
            "dashboard_kpis": kpis,
            "dashboard_daily": daily,
            "dashboard_weekly": weekly,
            "dashboard_monthly": monthly,
            "dashboard_weekday": weekday,
            "dashboard_note_tags": note_tags,
            "dashboard_quality_factors": quality_factors,
            "dashboard_distributions": distributions,
            "dashboard_recent_sessions": recent_sessions,
        }

        for name, table in tables.items():
            save_csv(
                table,
                args.output_dir / f"{name}.csv",
            )

        summary = build_summary(
            args.csv_path,
            all_rows,
            analytics,
            tables,
        )

        with (
            args.reports_dir / "dashboard_build_summary.json"
        ).open("w", encoding="utf-8") as file:
            json.dump(
                summary,
                file,
                ensure_ascii=False,
                indent=2,
            )

    except Exception as exc:
        print(
            f"ERROR al preparar el dashboard: {exc}",
            file=sys.stderr,
        )
        return 2

    print("\n" + "=" * 72)
    print("TABLAS DEL DASHBOARD GENERADAS")
    print("=" * 72)
    print(f"Sesiones analíticas:        {len(analytics)}")
    print(f"Días del calendario:        {len(daily)}")
    print(f"Días con registros:         {int(daily['has_data'].sum())}")
    print(f"Semanas generadas:          {len(weekly)}")
    print(f"Meses generados:            {len(monthly)}")
    print(f"Etiquetas encontradas:      {len(note_tags)}")
    print(f"Carpeta de tablas:          {args.output_dir.resolve()}")
    print(f"Resumen:                    {args.reports_dir.resolve()}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
