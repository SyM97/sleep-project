from pathlib import Path
import json

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sleep_sessions_clean.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "prediction"
REPORT_DIR = PROJECT_ROOT / "reports" / "prediction"

HORIZONS = [1, 7, 30]


def weighted_mean(group, value_column, weight_column="time_asleep_hours"):
    values = pd.to_numeric(group[value_column], errors="coerce")
    weights = pd.to_numeric(group[weight_column], errors="coerce")

    mask = values.notna() & weights.notna() & (weights > 0)

    if not mask.any():
        return values.mean()

    return np.average(
        values.loc[mask],
        weights=weights.loc[mask],
    )


def prepare_daily_data(df):
    """
    Convierte las sesiones válidas en una serie diaria.

    Si existe más de una sesión en una misma sleep_date:
    - las horas dormidas se suman;
    - las métricas porcentuales se ponderan por duración.
    """

    rows = []

    for sleep_date, group in df.groupby("sleep_date", sort=True):

        rows.append(
            {
                "sleep_date": sleep_date,

                "sessions_count": len(group),

                "sleep_hours":
                    group["time_asleep_hours"].sum(min_count=1),

                "sleep_quality_pct":
                    weighted_mean(
                        group,
                        "sleep_quality_pct"
                    ),

                "sleep_efficiency_pct":
                    weighted_mean(
                        group,
                        "sleep_efficiency_pct"
                    ),

                "regularity_pct":
                    weighted_mean(
                        group,
                        "regularity_pct"
                    ),

                "sleep_latency_minutes":
                    weighted_mean(
                        group,
                        "sleep_latency_minutes"
                    ),

                "bedtime_minutes_adjusted":
                    weighted_mean(
                        group,
                        "bedtime_minutes_adjusted"
                    ),
            }
        )

    daily = pd.DataFrame(rows)

    daily = daily.sort_values("sleep_date")

    return daily


def create_calendar(daily):
    """
    Crea un calendario completo.

    Los días sin registro permanecen como NaN.
    NO se interpolan horas de sueño.
    """

    start_date = daily["sleep_date"].min()
    end_date = daily["sleep_date"].max()

    calendar_index = pd.date_range(
        start=start_date,
        end=end_date,
        freq="D",
        name="date",
    )

    daily = daily.set_index("sleep_date")

    calendar = daily.reindex(calendar_index)

    calendar["has_data"] = (
        calendar["sessions_count"]
        .notna()
    )

    calendar["sessions_count"] = (
        calendar["sessions_count"]
        .fillna(0)
        .astype(int)
    )

    return calendar


def add_historical_features(calendar):
    """
    Crea únicamente variables que podrían conocerse
    en la fecha de predicción.
    """

    data = calendar.copy()

    # ---------------------------------------------------------
    # Último registro conocido
    # ---------------------------------------------------------

    historical_columns = [
        "sleep_hours",
        "sleep_quality_pct",
        "sleep_efficiency_pct",
        "regularity_pct",
        "sleep_latency_minutes",
        "bedtime_minutes_adjusted",
    ]

    for col in historical_columns:
        data[f"last_{col}"] = data[col].ffill()

    # ---------------------------------------------------------
    # Días desde el último registro
    # ---------------------------------------------------------

    last_record_date = pd.Series(
        pd.NaT,
        index=data.index,
        dtype="datetime64[ns]",
    )

    last_record_date.loc[data["has_data"]] = (
        data.index[data["has_data"]]
    )

    last_record_date = last_record_date.ffill()

    data["days_since_last_record"] = (
        data.index.to_series() - last_record_date
    ).dt.days

    # ---------------------------------------------------------
    # Duración: medias y variabilidad
    # ---------------------------------------------------------

    for window in [3, 7, 14, 30]:

        data[f"sleep_mean_{window}d"] = (
            data["sleep_hours"]
            .rolling(
                window=window,
                min_periods=1,
            )
            .mean()
        )

    data["sleep_std_7d"] = (
        data["sleep_hours"]
        .rolling(
            window=7,
            min_periods=2,
        )
        .std()
    )

    data["sleep_std_30d"] = (
        data["sleep_hours"]
        .rolling(
            window=30,
            min_periods=2,
        )
        .std()
    )

    # ---------------------------------------------------------
    # Calidad
    # ---------------------------------------------------------

    data["quality_mean_7d"] = (
        data["sleep_quality_pct"]
        .rolling(7, min_periods=1)
        .mean()
    )

    data["quality_mean_30d"] = (
        data["sleep_quality_pct"]
        .rolling(30, min_periods=1)
        .mean()
    )

    # ---------------------------------------------------------
    # Eficiencia
    # ---------------------------------------------------------

    data["efficiency_mean_7d"] = (
        data["sleep_efficiency_pct"]
        .rolling(7, min_periods=1)
        .mean()
    )

    data["efficiency_mean_30d"] = (
        data["sleep_efficiency_pct"]
        .rolling(30, min_periods=1)
        .mean()
    )

    # ---------------------------------------------------------
    # Regularidad
    # ---------------------------------------------------------

    data["regularity_mean_7d"] = (
        data["regularity_pct"]
        .rolling(7, min_periods=1)
        .mean()
    )

    data["regularity_mean_30d"] = (
        data["regularity_pct"]
        .rolling(30, min_periods=1)
        .mean()
    )

    # ---------------------------------------------------------
    # Latencia
    # ---------------------------------------------------------

    data["latency_mean_7d"] = (
        data["sleep_latency_minutes"]
        .rolling(7, min_periods=1)
        .mean()
    )

    data["latency_mean_30d"] = (
        data["sleep_latency_minutes"]
        .rolling(30, min_periods=1)
        .mean()
    )

    # ---------------------------------------------------------
    # Horario habitual
    # ---------------------------------------------------------

    data["bedtime_median_7d"] = (
        data["bedtime_minutes_adjusted"]
        .rolling(7, min_periods=1)
        .median()
    )

    data["bedtime_median_30d"] = (
        data["bedtime_minutes_adjusted"]
        .rolling(30, min_periods=1)
        .median()
    )

    # ---------------------------------------------------------
    # Cobertura de registros
    # ---------------------------------------------------------

    observed = data["has_data"].astype(int)

    data["records_7d"] = (
        observed
        .rolling(7, min_periods=1)
        .sum()
    )

    data["records_14d"] = (
        observed
        .rolling(14, min_periods=1)
        .sum()
    )

    data["records_30d"] = (
        observed
        .rolling(30, min_periods=1)
        .sum()
    )

    data["coverage_7d"] = (
        data["records_7d"] / 7
    )

    data["coverage_30d"] = (
        data["records_30d"] / 30
    )

    # ---------------------------------------------------------
    # Tendencia reciente
    # ---------------------------------------------------------

    data["sleep_trend_7d_vs_30d"] = (
        data["sleep_mean_7d"]
        - data["sleep_mean_30d"]
    )

    return data


def build_horizon_dataset(features, horizon):
    """
    Construye el dataset para un horizonte concreto.

    Ejemplo:
    horizon = 7

    Features:
        información conocida en fecha t

    Target:
        horas de sueño observadas en t + 7 días
    """

    dataset = features.copy()

    dataset["origin_date"] = dataset.index

    dataset["target_date"] = (
        dataset.index
        + pd.to_timedelta(horizon, unit="D")
    )

    # Target exacto en el futuro
    dataset["target_sleep_hours"] = (
        dataset["sleep_hours"]
        .shift(-horizon)
    )

    shifted_has_data = (
    dataset["has_data"]
    .astype("boolean")
    .shift(-horizon)
    )

    dataset["target_has_data"] = (
    shifted_has_data
    .fillna(False)
    .astype(bool)
    )

    # ---------------------------------------------------------
    # Variables conocidas del calendario futuro
    # ---------------------------------------------------------

    target_date = dataset["target_date"]

    # Monday = 0 ... Sunday = 6
    target_day_of_week = target_date.dt.dayofweek

    dataset["target_day_of_week"] = target_day_of_week

    dataset["target_is_weekend"] = (
        target_day_of_week >= 5
    ).astype(int)

    # Codificación cíclica del día de la semana
    dataset["target_dow_sin"] = np.sin(
        2
        * np.pi
        * target_day_of_week
        / 7
    )

    dataset["target_dow_cos"] = np.cos(
        2
        * np.pi
        * target_day_of_week
        / 7
    )

    target_month = target_date.dt.month

    # Codificación cíclica del mes
    dataset["target_month_sin"] = np.sin(
        2
        * np.pi
        * target_month
        / 12
    )

    dataset["target_month_cos"] = np.cos(
        2
        * np.pi
        * target_month
        / 12
    )

    dataset["horizon_days"] = horizon

    # ---------------------------------------------------------
    # Solo podemos entrenar cuando existe el valor real futuro
    # ---------------------------------------------------------

    dataset = dataset[
        dataset["target_has_data"]
    ].copy()

    # Exigimos un mínimo de historia reciente.
    dataset = dataset[
        dataset["records_30d"] >= 5
    ].copy()

    return dataset


def main():

    print("=" * 72)
    print("PREPARACIÓN DEL DATASET PARA PREDICCIÓN")
    print("=" * 72)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró: {INPUT_FILE}"
        )

    # ---------------------------------------------------------
    # Leer dataset limpio
    # ---------------------------------------------------------

    df = pd.read_csv(
        INPUT_FILE,
        sep=";",
    )

    source_rows = len(df)

    print(f"\nSesiones en archivo limpio: {source_rows}")

    # ---------------------------------------------------------
    # Parsear fechas
    # ---------------------------------------------------------

    df["sleep_date"] = pd.to_datetime(
        df["sleep_date"],
        errors="coerce",
    )

    if df["sleep_date"].isna().any():
        invalid_dates = df["sleep_date"].isna().sum()

        raise ValueError(
            f"Hay {invalid_dates} sleep_date inválidas."
        )

    # ---------------------------------------------------------
    # Filtrar sesiones válidas para analytics
    # ---------------------------------------------------------

    if df["include_in_analytics"].dtype == bool:

        eligible_mask = (
            df["include_in_analytics"]
        )

    else:

        eligible_mask = (
            df["include_in_analytics"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("true")
        )

    eligible = df[
        eligible_mask
    ].copy()

    excluded_rows = (
        source_rows
        - len(eligible)
    )

    print(
        f"Sesiones incluidas en analytics: "
        f"{len(eligible)}"
    )

    print(
        f"Sesiones excluidas: "
        f"{excluded_rows}"
    )

    # ---------------------------------------------------------
    # Orden cronológico
    # ---------------------------------------------------------

    eligible = eligible.sort_values(
        [
            "sleep_date",
            "went_to_bed",
        ]
    )

    # ---------------------------------------------------------
    # Agregación diaria
    # ---------------------------------------------------------

    daily = prepare_daily_data(
        eligible
    )

    multiple_session_days = (
        daily["sessions_count"] > 1
    ).sum()

    print(
        f"Días con datos: "
        f"{len(daily)}"
    )

    print(
        f"Días con múltiples sesiones: "
        f"{multiple_session_days}"
    )

    # ---------------------------------------------------------
    # Calendario completo
    # ---------------------------------------------------------

    calendar = create_calendar(
        daily
    )

    print(
        f"Días naturales en calendario: "
        f"{len(calendar)}"
    )

    print(
        f"Desde: {calendar.index.min().date()}"
    )

    print(
        f"Hasta: {calendar.index.max().date()}"
    )

    # ---------------------------------------------------------
    # Features históricas
    # ---------------------------------------------------------

    features = add_historical_features(
        calendar
    )

    # Guardamos el histórico diario completo
    history_output = (
        OUTPUT_DIR
        / "prediction_daily_history.csv"
    )

    history_export = (
        features
        .reset_index()
        .rename(columns={"date": "calendar_date"})
    )

    history_export.to_csv(
        history_output,
        sep=";",
        index=False,
    )

    print(
        "\nGuardado:"
    )

    print(
        f"  {history_output}"
    )

    # ---------------------------------------------------------
    # Datasets por horizonte
    # ---------------------------------------------------------

    report = {
        "source_rows": int(source_rows),
        "eligible_sessions": int(len(eligible)),
        "excluded_sessions": int(excluded_rows),
        "observed_days": int(len(daily)),
        "multiple_session_days": int(
            multiple_session_days
        ),
        "calendar_days": int(len(calendar)),
        "start_date": str(
            calendar.index.min().date()
        ),
        "end_date": str(
            calendar.index.max().date()
        ),
        "horizons": {},
    }

    for horizon in HORIZONS:

        dataset = build_horizon_dataset(
            features,
            horizon,
        )

        output_file = (
            OUTPUT_DIR
            / f"prediction_dataset_h{horizon}.csv"
        )

        dataset.to_csv(
            output_file,
            sep=";",
            index=False,
        )

        report["horizons"][str(horizon)] = {
            "rows": int(len(dataset)),
            "target_mean_hours": (
                round(
                    dataset[
                        "target_sleep_hours"
                    ].mean(),
                    4,
                )
                if len(dataset)
                else None
            ),
            "target_std_hours": (
                round(
                    dataset[
                        "target_sleep_hours"
                    ].std(),
                    4,
                )
                if len(dataset) > 1
                else None
            ),
        }

        print(
            f"\nHorizonte +{horizon} día(s)"
        )

        print(
            f"  Ejemplos utilizables: "
            f"{len(dataset)}"
        )

        print(
            f"  Archivo: {output_file}"
        )

        if len(dataset):

            print(
                "  Objetivo medio: "
                f"{dataset['target_sleep_hours'].mean():.2f} h"
            )

            print(
                "  Primer origen: "
                f"{dataset['origin_date'].min().date()}"
            )

            print(
                "  Último origen: "
                f"{dataset['origin_date'].max().date()}"
            )

    # ---------------------------------------------------------
    # Informe
    # ---------------------------------------------------------

    report_file = (
        REPORT_DIR
        / "prediction_dataset_summary.json"
    )

    with open(
        report_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "\nInforme:"
    )

    print(
        f"  {report_file}"
    )

    print(
        "\nIMPORTANTE:"
    )

    print(
        "No se ha interpolado ninguna noche sin datos."
    )

    print(
        "Las variables predictoras utilizan solamente "
        "información disponible hasta la fecha de origen."
    )

    print(
        "Los objetivos corresponden a días naturales exactos."
    )

    print("\nPreparación terminada.")
    print("=" * 72)


if __name__ == "__main__":
    main()