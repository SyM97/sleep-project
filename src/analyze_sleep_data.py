#!/usr/bin/env python3
"""
Análisis exploratorio del dataset limpio de sueño.

Entrada esperada:
    data/processed/sleep_sessions_clean.csv

Salidas:
    reports/eda/
        eda_summary.json
        monthly_metrics.csv
        weekday_metrics.csv
        correlation_matrix.csv
        note_tag_metrics.csv
        best_sessions.csv
        worst_sessions.csv
        charts/
            01_sleep_quality_timeline.png
            02_sleep_duration_timeline.png
            03_monthly_quality.png
            04_weekday_quality.png
            05_duration_vs_quality.png
            06_correlation_matrix.png

Uso:
    python src/analyze_sleep_data.py \
        data/processed/sleep_sessions_clean.csv \
        --output reports/eda
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

# Permite crear imágenes sin abrir ventanas.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WEEKDAY_ORDER = [
    "lunes",
    "martes",
    "miércoles",
    "jueves",
    "viernes",
    "sábado",
    "domingo",
]

MONTH_ORDER = [
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
    "day_of_week",
    "month_name",
    "include_in_analytics",
    "session_status",
}

CORRELATION_COLUMNS = [
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
]

DISPLAY_NAMES = {
    "sleep_quality_pct": "Calidad",
    "time_asleep_hours": "Horas dormidas",
    "time_in_bed_hours": "Horas en cama",
    "sleep_latency_minutes": "Latencia",
    "sleep_efficiency_pct": "Eficiencia",
    "regularity_pct": "Regularidad",
    "movements_per_hour": "Movimientos",
    "ambient_noise_db": "Ruido",
    "ambient_light_lux": "Luz",
    "coughing_per_hour": "Tos",
    "breathing_disruptions_per_hour": "Interrupciones respiratorias",
    "snore_percentage": "Ronquido",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera un análisis exploratorio del dataset limpio de sueño."
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Ruta de sleep_sessions_clean.csv",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("reports/eda"),
        help="Carpeta de salida. Predeterminado: reports/eda",
    )
    return parser.parse_args()


def read_clean_csv(csv_path: Path) -> pd.DataFrame:
    """
    El CSV limpio generado en el paso anterior utiliza punto y coma.
    Se incluye una comprobación para detectar accidentalmente una lectura
    como una sola columna.
    """
    try:
        df = pd.read_csv(
            csv_path,
            sep=";",
            encoding="utf-8-sig",
            low_memory=False,
        )
    except Exception as exc:
        raise RuntimeError(f"No se pudo leer el CSV limpio: {exc}") from exc

    if len(df.columns) == 1:
        raise RuntimeError(
            "El archivo se ha leído como una sola columna. "
            "Comprueba que sea sleep_sessions_clean.csv y que use ';' como separador."
        )

    return df


def normalize_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    normalized = series.astype("string").str.strip().str.lower()
    return normalized.map(
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
    ).fillna(False).astype(bool)


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = sorted(REQUIRED_COLUMNS.difference(df.columns))
    if missing:
        raise ValueError(
            "Faltan columnas necesarias en el CSV limpio:\n  - "
            + "\n  - ".join(missing)
        )

    prepared = df.copy()

    prepared["sleep_date"] = pd.to_datetime(
        prepared["sleep_date"],
        errors="coerce",
    )

    prepared["include_in_analytics"] = normalize_boolean(
        prepared["include_in_analytics"]
    )

    numeric_columns = set(CORRELATION_COLUMNS).union(
        {
            "time_in_bed_seconds",
            "time_asleep_seconds",
            "sleep_latency_seconds",
            "awake_time_minutes",
            "snore_time_minutes",
            "year",
            "month",
            "week_of_year",
            "day_of_week_num",
            "bedtime_minutes_adjusted",
            "wake_time_minutes_clock",
        }
    )

    for column in numeric_columns:
        if column in prepared.columns:
            prepared[column] = pd.to_numeric(
                prepared[column],
                errors="coerce",
            )

    analytics = prepared.loc[
        prepared["include_in_analytics"]
        & prepared["sleep_date"].notna()
    ].copy()

    analytics = analytics.sort_values("sleep_date").reset_index(drop=True)

    if analytics.empty:
        raise ValueError(
            "No existen sesiones marcadas con include_in_analytics=true."
        )

    # Medias móviles basadas en sesiones registradas, no en días naturales.
    analytics["quality_rolling_7"] = (
        analytics["sleep_quality_pct"]
        .rolling(window=7, min_periods=3)
        .mean()
    )
    analytics["quality_rolling_30"] = (
        analytics["sleep_quality_pct"]
        .rolling(window=30, min_periods=7)
        .mean()
    )
    analytics["duration_rolling_7"] = (
        analytics["time_asleep_hours"]
        .rolling(window=7, min_periods=3)
        .mean()
    )

    return prepared, analytics


def safe_float(value: Any, digits: int = 2) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def describe_metric(series: pd.Series) -> dict[str, float | None]:
    valid = pd.to_numeric(series, errors="coerce").dropna()

    if valid.empty:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "minimum": None,
            "maximum": None,
            "standard_deviation": None,
        }

    return {
        "count": int(valid.count()),
        "mean": safe_float(valid.mean()),
        "median": safe_float(valid.median()),
        "minimum": safe_float(valid.min()),
        "maximum": safe_float(valid.max()),
        "standard_deviation": safe_float(valid.std()),
    }


def build_summary(
    all_rows: pd.DataFrame,
    analytics: pd.DataFrame,
) -> dict[str, Any]:
    duration = analytics["time_asleep_hours"]
    quality = analytics["sleep_quality_pct"]
    efficiency = analytics["sleep_efficiency_pct"]

    valid_duration = duration.notna()
    recommended_duration = duration.between(7, 9, inclusive="both")

    high_quality = quality >= 85
    low_quality = quality < 70
    high_efficiency = efficiency >= 85

    first_date = analytics["sleep_date"].min()
    last_date = analytics["sleep_date"].max()

    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "total_cleaned_rows": int(len(all_rows)),
        "analytics_rows": int(len(analytics)),
        "excluded_rows": int((~all_rows["include_in_analytics"]).sum()),
        "date_range": {
            "first_sleep_date": first_date.date().isoformat(),
            "last_sleep_date": last_date.date().isoformat(),
            "calendar_days_covered": int((last_date - first_date).days + 1),
        },
        "session_status_counts": {
            str(key): int(value)
            for key, value in all_rows["session_status"]
            .fillna("unknown")
            .value_counts()
            .to_dict()
            .items()
        },
        "metrics": {
            "sleep_quality_pct": describe_metric(
                analytics["sleep_quality_pct"]
            ),
            "time_asleep_hours": describe_metric(
                analytics["time_asleep_hours"]
            ),
            "time_in_bed_hours": describe_metric(
                analytics["time_in_bed_hours"]
            ),
            "sleep_latency_minutes": describe_metric(
                analytics["sleep_latency_minutes"]
            ),
            "sleep_efficiency_pct": describe_metric(
                analytics["sleep_efficiency_pct"]
            ),
            "regularity_pct": describe_metric(
                analytics["regularity_pct"]
            ),
            "movements_per_hour": describe_metric(
                analytics["movements_per_hour"]
            ),
        },
        "threshold_counts": {
            "sessions_with_7_to_9_hours": int(
                (recommended_duration & valid_duration).sum()
            ),
            "sessions_with_7_to_9_hours_pct": safe_float(
                (recommended_duration & valid_duration).mean() * 100
            ),
            "sessions_quality_85_or_more": int(high_quality.sum()),
            "sessions_quality_85_or_more_pct": safe_float(
                high_quality.mean() * 100
            ),
            "sessions_quality_below_70": int(low_quality.sum()),
            "sessions_quality_below_70_pct": safe_float(
                low_quality.mean() * 100
            ),
            "sessions_efficiency_85_or_more": int(high_efficiency.sum()),
            "sessions_efficiency_85_or_more_pct": safe_float(
                high_efficiency.mean() * 100
            ),
        },
    }


def build_monthly_metrics(analytics: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        analytics.set_index("sleep_date")
        .resample("MS")
        .agg(
            sessions=("sleep_quality_pct", "count"),
            mean_sleep_quality_pct=("sleep_quality_pct", "mean"),
            median_sleep_quality_pct=("sleep_quality_pct", "median"),
            mean_time_asleep_hours=("time_asleep_hours", "mean"),
            mean_time_in_bed_hours=("time_in_bed_hours", "mean"),
            mean_sleep_efficiency_pct=("sleep_efficiency_pct", "mean"),
            mean_sleep_latency_minutes=("sleep_latency_minutes", "mean"),
            mean_regularity_pct=("regularity_pct", "mean"),
            mean_movements_per_hour=("movements_per_hour", "mean"),
        )
        .reset_index()
        .rename(columns={"sleep_date": "month"})
    )

    monthly["month_label"] = monthly["month"].dt.strftime("%Y-%m")

    numeric = monthly.select_dtypes(include="number").columns
    monthly[numeric] = monthly[numeric].round(2)

    columns = ["month", "month_label"] + [
        column
        for column in monthly.columns
        if column not in {"month", "month_label"}
    ]
    return monthly[columns]


def build_weekday_metrics(analytics: pd.DataFrame) -> pd.DataFrame:
    weekday = (
        analytics.groupby("day_of_week", dropna=False)
        .agg(
            sessions=("sleep_quality_pct", "count"),
            mean_sleep_quality_pct=("sleep_quality_pct", "mean"),
            median_sleep_quality_pct=("sleep_quality_pct", "median"),
            mean_time_asleep_hours=("time_asleep_hours", "mean"),
            mean_time_in_bed_hours=("time_in_bed_hours", "mean"),
            mean_sleep_efficiency_pct=("sleep_efficiency_pct", "mean"),
            mean_sleep_latency_minutes=("sleep_latency_minutes", "mean"),
            mean_regularity_pct=("regularity_pct", "mean"),
        )
        .reset_index()
    )

    weekday["day_of_week"] = pd.Categorical(
        weekday["day_of_week"],
        categories=WEEKDAY_ORDER,
        ordered=True,
    )
    weekday = weekday.sort_values("day_of_week")

    numeric = weekday.select_dtypes(include="number").columns
    weekday[numeric] = weekday[numeric].round(2)

    return weekday


def build_correlation_matrix(analytics: pd.DataFrame) -> pd.DataFrame:
    available = [
        column
        for column in CORRELATION_COLUMNS
        if column in analytics.columns
        and analytics[column].notna().sum() >= 20
        and analytics[column].nunique(dropna=True) > 1
    ]

    return analytics[available].corr(method="pearson").round(3)


def split_note_tags(value: Any) -> list[str]:
    if value is None or pd.isna(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    # Las notas del dataset se separan principalmente mediante dos puntos.
    tags = []
    for raw_tag in text.replace(";", ":").split(":"):
        tag = " ".join(raw_tag.strip().lower().split())
        if tag:
            tags.append(tag)

    return list(dict.fromkeys(tags))


def build_note_tag_metrics(analytics: pd.DataFrame) -> pd.DataFrame:
    if "notes" not in analytics.columns:
        return pd.DataFrame(
            columns=[
                "tag",
                "sessions",
                "mean_sleep_quality_pct",
                "mean_time_asleep_hours",
                "mean_sleep_efficiency_pct",
                "quality_difference_vs_other_sessions",
            ]
        )

    rows: list[dict[str, Any]] = []

    tags_by_index = analytics["notes"].apply(split_note_tags)
    unique_tags = sorted(
        {
            tag
            for tags in tags_by_index
            for tag in tags
        }
    )

    for tag in unique_tags:
        mask = tags_by_index.apply(lambda tags: tag in tags)
        tagged = analytics.loc[mask]
        untagged = analytics.loc[~mask]

        if tagged.empty:
            continue

        quality_difference = (
            tagged["sleep_quality_pct"].mean()
            - untagged["sleep_quality_pct"].mean()
        )

        rows.append(
            {
                "tag": tag,
                "sessions": int(len(tagged)),
                "mean_sleep_quality_pct": safe_float(
                    tagged["sleep_quality_pct"].mean()
                ),
                "mean_time_asleep_hours": safe_float(
                    tagged["time_asleep_hours"].mean()
                ),
                "mean_sleep_efficiency_pct": safe_float(
                    tagged["sleep_efficiency_pct"].mean()
                ),
                "quality_difference_vs_other_sessions": safe_float(
                    quality_difference
                ),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "tag",
                "sessions",
                "mean_sleep_quality_pct",
                "mean_time_asleep_hours",
                "mean_sleep_efficiency_pct",
                "quality_difference_vs_other_sessions",
            ]
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["sessions", "mean_sleep_quality_pct"],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )


def select_session_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
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
        "did_snore",
        "notes",
    ]
    return [column for column in preferred if column in df.columns]


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(
        path,
        sep=";",
        index=False,
        encoding="utf-8-sig",
    )


def create_quality_timeline(
    analytics: pd.DataFrame,
    charts_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        analytics["sleep_date"],
        analytics["sleep_quality_pct"],
        linewidth=0.8,
        alpha=0.45,
        label="Calidad por sesión",
    )
    ax.plot(
        analytics["sleep_date"],
        analytics["quality_rolling_7"],
        linewidth=2,
        label="Media móvil de 7 sesiones",
    )
    ax.plot(
        analytics["sleep_date"],
        analytics["quality_rolling_30"],
        linewidth=2,
        label="Media móvil de 30 sesiones",
    )

    ax.set_title("Evolución de la calidad del sueño")
    ax.set_xlabel("Fecha de sueño")
    ax.set_ylabel("Calidad (%)")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.25)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(
        charts_dir / "01_sleep_quality_timeline.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)


def create_duration_timeline(
    analytics: pd.DataFrame,
    charts_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        analytics["sleep_date"],
        analytics["time_asleep_hours"],
        linewidth=0.8,
        alpha=0.5,
        label="Horas dormidas",
    )
    ax.plot(
        analytics["sleep_date"],
        analytics["duration_rolling_7"],
        linewidth=2,
        label="Media móvil de 7 sesiones",
    )

    ax.axhspan(
        7,
        9,
        alpha=0.12,
        label="Intervalo analítico de 7–9 horas",
    )

    ax.set_title("Evolución de la duración del sueño")
    ax.set_xlabel("Fecha de sueño")
    ax.set_ylabel("Horas dormidas")
    ax.legend()
    ax.grid(True, alpha=0.25)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(
        charts_dir / "02_sleep_duration_timeline.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)


def create_monthly_quality(
    monthly: pd.DataFrame,
    charts_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        monthly["month"],
        monthly["mean_sleep_quality_pct"],
        marker="o",
        linewidth=2,
    )

    ax.set_title("Calidad media del sueño por mes")
    ax.set_xlabel("Mes")
    ax.set_ylabel("Calidad media (%)")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.25)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(
        charts_dir / "03_monthly_quality.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)


def create_weekday_quality(
    weekday: pd.DataFrame,
    charts_dir: Path,
) -> None:
    plot_df = weekday.dropna(subset=["day_of_week"]).copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        plot_df["day_of_week"].astype(str),
        plot_df["mean_sleep_quality_pct"],
    )

    ax.set_title("Calidad media por día de la semana")
    ax.set_xlabel("Día de la semana")
    ax.set_ylabel("Calidad media (%)")
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", alpha=0.25)

    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(
        charts_dir / "04_weekday_quality.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)


def create_duration_quality_scatter(
    analytics: pd.DataFrame,
    charts_dir: Path,
) -> None:
    plot_df = analytics[
        ["time_asleep_hours", "sleep_quality_pct"]
    ].dropna()

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(
        plot_df["time_asleep_hours"],
        plot_df["sleep_quality_pct"],
        alpha=0.55,
    )

    if len(plot_df) >= 2:
        coefficients = np.polyfit(
            plot_df["time_asleep_hours"],
            plot_df["sleep_quality_pct"],
            deg=1,
        )
        x_values = np.linspace(
            plot_df["time_asleep_hours"].min(),
            plot_df["time_asleep_hours"].max(),
            100,
        )
        y_values = (
            coefficients[0] * x_values
            + coefficients[1]
        )
        ax.plot(
            x_values,
            y_values,
            linewidth=2,
            label="Tendencia lineal",
        )
        ax.legend()

    correlation = plot_df.corr().iloc[0, 1]

    ax.set_title(
        "Duración frente a calidad del sueño "
        f"(correlación: {correlation:.2f})"
    )
    ax.set_xlabel("Horas dormidas")
    ax.set_ylabel("Calidad (%)")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        charts_dir / "05_duration_vs_quality.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)


def create_correlation_chart(
    correlation: pd.DataFrame,
    charts_dir: Path,
) -> None:
    if correlation.empty:
        return

    labels = [
        DISPLAY_NAMES.get(column, column)
        for column in correlation.columns
    ]

    fig_size = max(9, len(labels) * 0.8)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    image = ax.imshow(
        correlation.to_numpy(),
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
        aspect="auto",
    )

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title("Matriz de correlaciones")

    for row in range(len(labels)):
        for column in range(len(labels)):
            value = correlation.iloc[row, column]
            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    fig.colorbar(image, ax=ax, label="Correlación de Pearson")
    fig.tight_layout()
    fig.savefig(
        charts_dir / "06_correlation_matrix.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> int:
    args = parse_args()
    csv_path: Path = args.csv_path
    output_dir: Path = args.output
    charts_dir = output_dir / "charts"

    if not csv_path.exists():
        print(
            f"ERROR: no existe el archivo {csv_path}",
            file=sys.stderr,
        )
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    try:
        raw_cleaned = read_clean_csv(csv_path)
        all_rows, analytics = prepare_data(raw_cleaned)

        summary = build_summary(all_rows, analytics)
        monthly = build_monthly_metrics(analytics)
        weekday = build_weekday_metrics(analytics)
        correlation = build_correlation_matrix(analytics)
        note_tags = build_note_tag_metrics(analytics)

        session_columns = select_session_columns(analytics)

        best_sessions = (
            analytics.sort_values(
                ["sleep_quality_pct", "time_asleep_hours"],
                ascending=[False, False],
            )
            .head(15)[session_columns]
        )

        worst_sessions = (
            analytics.sort_values(
                ["sleep_quality_pct", "time_asleep_hours"],
                ascending=[True, True],
            )
            .head(15)[session_columns]
        )

        with (output_dir / "eda_summary.json").open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                summary,
                file,
                ensure_ascii=False,
                indent=2,
            )

        save_csv(monthly, output_dir / "monthly_metrics.csv")
        save_csv(weekday, output_dir / "weekday_metrics.csv")
        correlation.to_csv(
            output_dir / "correlation_matrix.csv",
            sep=";",
            encoding="utf-8-sig",
        )
        save_csv(note_tags, output_dir / "note_tag_metrics.csv")
        save_csv(best_sessions, output_dir / "best_sessions.csv")
        save_csv(worst_sessions, output_dir / "worst_sessions.csv")

        create_quality_timeline(analytics, charts_dir)
        create_duration_timeline(analytics, charts_dir)
        create_monthly_quality(monthly, charts_dir)
        create_weekday_quality(weekday, charts_dir)
        create_duration_quality_scatter(analytics, charts_dir)
        create_correlation_chart(correlation, charts_dir)

    except Exception as exc:
        print(f"ERROR durante el análisis: {exc}", file=sys.stderr)
        return 2

    metrics = summary["metrics"]
    thresholds = summary["threshold_counts"]

    print("\n" + "=" * 70)
    print("ANÁLISIS EXPLORATORIO DEL SUEÑO COMPLETADO")
    print("=" * 70)
    print(f"Sesiones analizadas:        {summary['analytics_rows']}")
    print(
        "Periodo:                    "
        f"{summary['date_range']['first_sleep_date']} "
        f"→ {summary['date_range']['last_sleep_date']}"
    )
    print(
        "Calidad media:              "
        f"{metrics['sleep_quality_pct']['mean']} %"
    )
    print(
        "Tiempo medio dormido:       "
        f"{metrics['time_asleep_hours']['mean']} h"
    )
    print(
        "Eficiencia media:           "
        f"{metrics['sleep_efficiency_pct']['mean']} %"
    )
    print(
        "Latencia media:             "
        f"{metrics['sleep_latency_minutes']['mean']} min"
    )
    print(
        "Sesiones entre 7 y 9 h:     "
        f"{thresholds['sessions_with_7_to_9_hours']} "
        f"({thresholds['sessions_with_7_to_9_hours_pct']} %)"
    )
    print(f"Informes:                    {output_dir.resolve()}")
    print(f"Gráficas:                    {charts_dir.resolve()}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
