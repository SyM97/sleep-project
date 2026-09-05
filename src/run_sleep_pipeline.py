from pathlib import Path
from datetime import datetime
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback


# ============================================================
# RUTAS DEL PROYECTO
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RAW_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "My complete sleep data.csv"
)

CLEAN_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sleep_sessions_clean.csv"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "pipeline"
)


# ============================================================
# ARGUMENTOS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Run the complete sleep analytics "
            "and next-night sleep type pipeline."
        )
    )

    parser.add_argument(
        "--input-file",
        type=Path,
        default=DEFAULT_RAW_FILE,
        help=(
            "CSV completo de Sleep Cycle que se utilizará "
            "como fuente. Si no se indica, se usa "
            "data/raw/My complete sleep data.csv."
        ),
    )

    parser.add_argument(
        "--skip-db",
        action="store_true",
        help=(
            "Ejecuta todos los cálculos, análisis y la "
            "estimación histórica sin escribir nada "
            "en Supabase."
        ),
    )

    return parser.parse_args()


# ============================================================
# ETAPAS DEL PIPELINE
# ============================================================

def build_stages(raw_file):

    return [
        {
            "name": "Audit raw sleep data",
            "script": "audit_sleep_data.py",
            "args": [
                str(raw_file),
            ],
            "db_write": False,
        },

        {
            "name": "Clean sleep data",
            "script": "clean_sleep_data.py",
            "args": [
                str(raw_file),
            ],
            "db_write": False,
        },

        {
            "name": "Analyze sleep data",
            "script": "analyze_sleep_data.py",
            "args": [
                str(CLEAN_FILE),
            ],
            "db_write": False,
        },

        {
            "name": "Prepare dashboard data",
            "script": "prepare_dashboard_data.py",
            "args": [
                str(CLEAN_FILE),
            ],
            "db_write": False,
        },

        {
            "name": "Prepare next-night prediction dataset",
            "script": "prepare_prediction_dataset.py",
            "args": [],
            "db_write": False,
        },

        {
            "name": "Build historical sleep type prediction",
            "script": "build_sleep_type_prediction.py",
            "args": [],
            "db_write": False,
        },

        {
            "name": "Load dashboard to Supabase",
            "script": "load_dashboard_to_supabase.py",
            "args": [
                "--yes",
            ],
            "db_write": True,
        },

        {
            "name": "Load sleep type prediction to Supabase",
            "script": "load_sleep_type_prediction_to_supabase.py",
            "args": [],
            "db_write": True,
        },
    ]


# ============================================================
# HASH SHA256 DEL DATASET
# ============================================================

def sha256_file(path):

    sha = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as f:

        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):

            sha.update(
                chunk
            )

    return sha.hexdigest()


# ============================================================
# VALIDACIÓN INICIAL
# ============================================================

def validate_environment(
    raw_file,
    stages,
):

    print("=" * 80)
    print("VALIDACIÓN DEL PIPELINE")
    print("=" * 80)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    if not raw_file.exists():

        raise FileNotFoundError(
            f"No se encontró el dataset:\n{raw_file}"
        )

    if not raw_file.is_file():

        raise ValueError(
            f"La ruta indicada no es un archivo:\n{raw_file}"
        )

    if raw_file.suffix.lower() != ".csv":

        raise ValueError(
            "El fichero de entrada debe tener extensión .csv"
        )

    if raw_file.stat().st_size == 0:

        raise ValueError(
            "El dataset de entrada está vacío."
        )

    print()
    print("Dataset de entrada:")

    print(
        f"  {raw_file}"
    )

    print(
        f"  Nombre: "
        f"{raw_file.name}"
    )

    print(
        f"  Tamaño: "
        f"{raw_file.stat().st_size:,} bytes"
    )

    print(
        f"  SHA256: "
        f"{sha256_file(raw_file)}"
    )

    # --------------------------------------------------------
    # Archivo limpio derivado
    # --------------------------------------------------------

    print()
    print("Archivo limpio esperado:")

    print(
        f"  {CLEAN_FILE}"
    )

    # --------------------------------------------------------
    # Python
    # --------------------------------------------------------

    print()
    print("Python:")

    print(
        f"  {sys.executable}"
    )

    # --------------------------------------------------------
    # Scripts
    # --------------------------------------------------------

    src_dir = (
        PROJECT_ROOT
        / "src"
    )

    missing_scripts = []

    for stage in stages:

        script_path = (
            src_dir
            / stage["script"]
        )

        if not script_path.exists():

            missing_scripts.append(
                stage["script"]
            )

    if missing_scripts:

        raise FileNotFoundError(
            "Faltan scripts necesarios:\n"
            + "\n".join(
                f"- {name}"
                for name in missing_scripts
            )
        )

    print()
    print("Scripts requeridos:")

    print(
        f"  {len(stages)} encontrados"
    )

    # --------------------------------------------------------
    # .env
    # --------------------------------------------------------

    env_file = (
        PROJECT_ROOT
        / ".env"
    )

    print()
    print(".env:")

    if env_file.exists():

        print(
            "  Encontrado"
        )

    else:

        print(
            "  No encontrado"
        )

        print(
            "  Las etapas locales podrán ejecutarse, "
            "pero las cargas a Supabase fallarán."
        )

    print()
    print(
        "Validación inicial correcta."
    )


# ============================================================
# EJECUTAR UNA ETAPA
# ============================================================

def run_stage(
    stage,
    log_file,
):

    script_path = (
        PROJECT_ROOT
        / "src"
        / stage["script"]
    )

    command = [
        sys.executable,
        str(script_path),
        *stage.get(
            "args",
            [],
        ),
    ]

    env = (
        os.environ.copy()
    )

    # Mejora compatibilidad Unicode en Windows.
    env["PYTHONUTF8"] = "1"

    env[
        "PYTHONIOENCODING"
    ] = "utf-8"

    print()
    print("=" * 80)

    print(
        stage["name"].upper()
    )

    print("=" * 80)

    print()
    print("Ejecutando:")

    print(
        "  "
        + subprocess.list2cmdline(
            command
        )
    )

    start_time = (
        time.perf_counter()
    )

    with open(
        log_file,
        "a",
        encoding="utf-8",
    ) as log:

        log.write(
            "\n"
            + "=" * 80
            + "\n"
        )

        log.write(
            stage["name"]
            + "\n"
        )

        log.write(
            "=" * 80
            + "\n"
        )

        log.write(
            "Command: "
            + subprocess.list2cmdline(
                command
            )
            + "\n\n"
        )

        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )

        assert (
            process.stdout
            is not None
        )

        for line in process.stdout:

            print(
                line,
                end="",
            )

            log.write(
                line
            )

        return_code = (
            process.wait()
        )

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    if return_code != 0:

        raise RuntimeError(
            f"La etapa '{stage['name']}' "
            f"falló con código "
            f"{return_code}."
        )

    print()

    print(
        f"✓ Etapa completada "
        f"en {elapsed_seconds:.1f} s"
    )

    return {
        "name":
            stage["name"],

        "script":
            stage["script"],

        "arguments":
            stage.get(
                "args",
                [],
            ),

        "status":
            "success",

        "duration_seconds":
            round(
                elapsed_seconds,
                2,
            ),
    }


# ============================================================
# GUARDAR RESUMEN
# ============================================================

def write_summary(
    summary,
    timestamp_id,
):

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp_file = (
        REPORT_DIR
        / (
            "pipeline_run_"
            + timestamp_id
            + ".json"
        )
    )

    latest_file = (
        REPORT_DIR
        / "pipeline_run_latest.json"
    )

    for path in [
        timestamp_file,
        latest_file,
    ]:

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                summary,
                f,
                indent=2,
                ensure_ascii=False,
            )

    return (
        timestamp_file,
        latest_file,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    # --------------------------------------------------------
    # Resolver fichero de entrada
    # --------------------------------------------------------

    raw_file = (
        args.input_file
        .expanduser()
    )

    if not raw_file.is_absolute():

        raw_file = (
            PROJECT_ROOT
            / raw_file
        )

    raw_file = (
        raw_file.resolve()
    )

    stages = (
        build_stages(
            raw_file
        )
    )

    # --------------------------------------------------------
    # Datos de ejecución
    # --------------------------------------------------------

    started_at = (
        datetime
        .now()
        .astimezone()
    )

    timestamp_id = (
        started_at
        .strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_file = (
        REPORT_DIR
        / (
            "pipeline_run_"
            + timestamp_id
            + ".log"
        )
    )

    summary = {

        "started_at":
            started_at.isoformat(),

        "finished_at":
            None,

        "status":
            "running",

        "skip_database_writes":
            args.skip_db,

        "dataset": {

            "path":
                str(raw_file),

            "filename":
                raw_file.name,

            "size_bytes":
                None,

            "sha256":
                None,
        },

        "clean_dataset": {

            "path":
                str(CLEAN_FILE),
        },

        "stages":
            [],

        "error":
            None,
    }

    pipeline_start = (
        time.perf_counter()
    )

    try:

        # ----------------------------------------------------
        # Validación
        # ----------------------------------------------------

        validate_environment(
            raw_file,
            stages,
        )

        summary[
            "dataset"
        ][
            "size_bytes"
        ] = (
            raw_file
            .stat()
            .st_size
        )

        summary[
            "dataset"
        ][
            "sha256"
        ] = (
            sha256_file(
                raw_file
            )
        )

        # ----------------------------------------------------
        # Ejecutar pipeline
        # ----------------------------------------------------

        for stage in stages:

            if (
                args.skip_db
                and stage[
                    "db_write"
                ]
            ):

                print()
                print("=" * 80)

                print(
                    f"OMITIENDO: "
                    f"{stage['name']}"
                )

                print("=" * 80)

                print(
                    "Motivo: --skip-db"
                )

                summary[
                    "stages"
                ].append(
                    {
                        "name":
                            stage[
                                "name"
                            ],

                        "script":
                            stage[
                                "script"
                            ],

                        "arguments":
                            stage.get(
                                "args",
                                [],
                            ),

                        "status":
                            "skipped",

                        "reason":
                            "--skip-db",
                    }
                )

                continue

            result = (
                run_stage(
                    stage,
                    log_file,
                )
            )

            summary[
                "stages"
            ].append(
                result
            )

        summary[
            "status"
        ] = "success"

    except Exception as exc:

        summary[
            "status"
        ] = "failed"

        summary[
            "error"
        ] = {

            "type":
                type(
                    exc
                ).__name__,

            "message":
                str(exc),

            "traceback":
                traceback.format_exc(),
        }

        print()
        print("=" * 80)
        print("PIPELINE DETENIDO")
        print("=" * 80)

        print()

        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )

    finally:

        # ----------------------------------------------------
        # Finalización
        # ----------------------------------------------------

        finished_at = (
            datetime
            .now()
            .astimezone()
        )

        summary[
            "finished_at"
        ] = (
            finished_at.isoformat()
        )

        summary[
            "total_duration_seconds"
        ] = round(
            time.perf_counter()
            - pipeline_start,
            2,
        )

        (
            timestamp_summary,
            latest_summary,
        ) = (
            write_summary(
                summary,
                timestamp_id,
            )
        )

        print()
        print("=" * 80)

        if (
            summary[
                "status"
            ]
            == "success"
        ):

            print(
                "PIPELINE COMPLETADO "
                "CORRECTAMENTE"
            )

        else:

            print(
                "PIPELINE FINALIZADO "
                "CON ERROR"
            )

        print("=" * 80)

        print()

        print(
            f"Duración total: "
            f"{summary['total_duration_seconds']:.1f} s"
        )

        print()
        print("Dataset procesado:")

        print(
            f"  {raw_file.name}"
        )

        if (
            summary[
                "dataset"
            ][
                "sha256"
            ]
        ):

            print(
                f"  SHA256: "
                f"{summary['dataset']['sha256']}"
            )

        print()
        print("Log:")

        print(
            f"  {log_file}"
        )

        print()
        print("Resumen:")

        print(
            f"  {timestamp_summary}"
        )

        print()
        print(
            "Última ejecución:"
        )

        print(
            f"  {latest_summary}"
        )

    # --------------------------------------------------------
    # Exit code útil para GitHub Actions / Make
    # --------------------------------------------------------

    if (
        summary[
            "status"
        ]
        != "success"
    ):

        sys.exit(1)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()