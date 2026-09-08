from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "sleep_sessions_clean.csv"


def main():
    print("=" * 70)
    print("AUDITORÃA DEL DATASET PARA PREDICCIÃ“N")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"No se encontrÃ³ el archivo: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, sep=";")

    print(f"\nArchivo:")
    print(INPUT_FILE)

    print(f"\nDimensiones:")
    print(f"Filas:    {len(df):,}")
    print(f"Columnas: {len(df.columns):,}")

    print("\nCOLUMNAS DISPONIBLES")
    print("-" * 70)

    for i, col in enumerate(df.columns, start=1):
        print(f"{i:02d}. {col}")

    print("\nTIPOS DE DATOS")
    print("-" * 70)
    print(df.dtypes.to_string())

    print("\nVALORES NULOS")
    print("-" * 70)

    nulls = pd.DataFrame({
        "null_count": df.isna().sum(),
        "null_pct": (df.isna().mean() * 100).round(2),
    })

    nulls = nulls[nulls["null_count"] > 0].sort_values(
        "null_pct",
        ascending=False
    )

    if nulls.empty:
        print("No hay valores nulos.")
    else:
        print(nulls.to_string())

    print("\nPOSIBLES COLUMNAS DE FECHA/HORA")
    print("-" * 70)

    date_keywords = (
        "date",
        "time",
        "bed",
        "wake",
        "woke",
        "start",
        "end",
    )

    date_cols = [
        col
        for col in df.columns
        if any(keyword in col.lower() for keyword in date_keywords)
    ]

    if date_cols:
        for col in date_cols:
            print(f"- {col}")
    else:
        print("No se detectaron automÃ¡ticamente.")

    print("\nPOSIBLES VARIABLES OBJETIVO")
    print("-" * 70)

    target_keywords = (
        "asleep",
        "duration",
        "quality",
        "efficiency",
    )

    target_cols = [
        col
        for col in df.columns
        if any(keyword in col.lower() for keyword in target_keywords)
    ]

    if target_cols:
        for col in target_cols:
            print(f"- {col}")
    else:
        print("No se detectaron automÃ¡ticamente.")

    print("\nPOSIBLES FLAGS / COLUMNAS DE CONTROL")
    print("-" * 70)

    flag_keywords = (
        "flag",
        "valid",
        "exclude",
        "eligible",
        "quality",
        "failed",
        "analytics",
    )

    flag_cols = [
        col
        for col in df.columns
        if any(keyword in col.lower() for keyword in flag_keywords)
    ]

    if flag_cols:
        for col in flag_cols:
            print(f"- {col}")

            unique_values = df[col].dropna().unique()

            if len(unique_values) <= 20:
                print(f"    Valores: {unique_values}")
    else:
        print("No se detectaron automÃ¡ticamente.")

    print("\nPRIMERAS 3 FILAS")
    print("-" * 70)
    print(df.head(3).to_string())

    print("\nÃšLTIMAS 3 FILAS")
    print("-" * 70)
    print(df.tail(3).to_string())

    print("\nAUDITORÃA FINALIZADA")
    print("=" * 70)


if __name__ == "__main__":
    main()
