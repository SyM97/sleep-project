# Auditoría local del dataset de sueño

Este pequeño proyecto ejecuta el primer paso del análisis: **auditar el CSV sin modificarlo**.

## 1. Requisitos

- Python 3.10 o superior
- Visual Studio Code
- El archivo `My complete sleep data.csv`

## 2. Preparar la carpeta

Coloca estos archivos en una carpeta:

```text
sleep-audit/
├── audit_sleep_data.py
├── requirements.txt
└── My complete sleep data.csv
```

El CSV no está incluido en este paquete para evitar duplicar tus datos personales.

## 3. Crear el entorno virtual

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Si PowerShell bloquea la activación:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

### macOS o Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Ejecutar la auditoría

```bash
python audit_sleep_data.py "My complete sleep data.csv"
```

Para elegir otra carpeta de salida:

```bash
python audit_sleep_data.py "My complete sleep data.csv" --output reports
```

## 5. Archivos generados

La carpeta `reports/` contendrá:

- `audit_summary.json`: resumen general legible por programas.
- `column_profile.csv`: cobertura, valores únicos, ejemplos y estadísticas por columna.
- `data_dictionary.csv`: propuesta de nombres, tipos y reglas de cada campo.
- `row_audit.csv`: resultado de validación para las 438 filas.
- `anomalous_sessions.csv`: únicamente las sesiones con errores o advertencias.
- `duplicate_bedtimes.csv`: registros que comparten la misma hora de inicio.

Los CSV de salida utilizan codificación UTF-8 con BOM para abrirse correctamente en Excel.

## 6. Qué significa cada tipo de incidencia

### Errores

- `invalid_bedtime`
- `invalid_wake_time`
- `invalid_wake_year`
- `wake_before_bed`
- `nonpositive_time_in_bed`
- `time_asleep_exceeds_time_in_bed`
- `invalid_sleep_quality`
- `invalid_regularity`

### Advertencias

- `very_short_session_under_2h`
- `very_long_session_over_16h`
- `zero_time_asleep`
- `duplicate_bedtime`
- `possible_dst_transition`
- `duration_mismatch_over_5min`

Una advertencia no implica necesariamente que el dato sea incorrecto. Por ejemplo, una diferencia cercana a una hora puede coincidir con el cambio de horario de verano/invierno.

## 7. Importante

El script solo realiza una auditoría. No elimina, corrige ni sobrescribe registros. La limpieza se implementará en el siguiente paso después de revisar los informes.
