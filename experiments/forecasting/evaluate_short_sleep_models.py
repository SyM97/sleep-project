from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import (
    ParameterGrid,
    TimeSeriesSplit,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


# ============================================================
# CONFIGURACIÃ“N
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "prediction"
    / "prediction_dataset_h1.csv"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "prediction"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "prediction"
)

SHORT_SLEEP_THRESHOLD_HOURS = 6.0

TEST_FRACTION = 0.20

N_TEMPORAL_FOLDS = 4

RANDOM_STATE = 42


# ============================================================
# UTILIDADES
# ============================================================

def make_one_hot_encoder():
    """
    Compatibilidad con diferentes versiones
    de scikit-learn.
    """

    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
        )


def clip_probabilities(values):
    return np.clip(
        np.asarray(
            values,
            dtype=float,
        ),
        1e-6,
        1 - 1e-6,
    )


def safe_roc_auc(
    y_true,
    probabilities,
):
    if len(np.unique(y_true)) < 2:
        return None

    return float(
        roc_auc_score(
            y_true,
            probabilities,
        )
    )


def safe_pr_auc(
    y_true,
    probabilities,
):
    if len(np.unique(y_true)) < 2:
        return None

    return float(
        average_precision_score(
            y_true,
            probabilities,
        )
    )


def probability_metrics(
    y_true,
    probabilities,
):
    probabilities = clip_probabilities(
        probabilities
    )

    return {
        "brier_score": float(
            brier_score_loss(
                y_true,
                probabilities,
            )
        ),

        "log_loss": float(
            log_loss(
                y_true,
                probabilities,
                labels=[0, 1],
            )
        ),

        "roc_auc": safe_roc_auc(
            y_true,
            probabilities,
        ),

        "pr_auc": safe_pr_auc(
            y_true,
            probabilities,
        ),
    }


def mean_optional(values):
    clean = [
        float(value)
        for value in values
        if value is not None
        and not pd.isna(value)
    ]

    if not clean:
        return None

    return float(
        np.mean(clean)
    )


def percentage(value):
    if value is None:
        return None

    return round(
        float(value) * 100,
        2,
    )


# ============================================================
# CARGA
# ============================================================

def load_dataset():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"No existe:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        sep=";",
        low_memory=False,
    )

    required = {
        "origin_date",
        "target_date",
        "target_sleep_hours",
    }

    missing = sorted(
        required.difference(
            df.columns
        )
    )

    if missing:
        raise ValueError(
            "Faltan columnas:\n- "
            + "\n- ".join(missing)
        )

    df[
        "origin_date"
    ] = pd.to_datetime(
        df["origin_date"],
        errors="coerce",
    )

    df[
        "target_date"
    ] = pd.to_datetime(
        df["target_date"],
        errors="coerce",
    )

    df[
        "target_sleep_hours"
    ] = pd.to_numeric(
        df["target_sleep_hours"],
        errors="coerce",
    )

    df = df.loc[
        df["origin_date"].notna()
        & df["target_date"].notna()
        & df[
            "target_sleep_hours"
        ].notna()
    ].copy()

    df[
        "short_sleep_target"
    ] = (
        df[
            "target_sleep_hours"
        ]
        < SHORT_SLEEP_THRESHOLD_HOURS
    ).astype(int)

    df = (
        df
        .sort_values(
            "origin_date"
        )
        .reset_index(
            drop=True
        )
    )

    if df.empty:
        raise ValueError(
            "Dataset vacÃ­o."
        )

    return df


# ============================================================
# SPLIT FINAL TEMPORAL + PURGING
# ============================================================

def build_final_split(df):

    split_index = int(
        len(df)
        * (
            1
            - TEST_FRACTION
        )
    )

    if (
        split_index <= 0
        or split_index >= len(df)
    ):
        raise ValueError(
            "Split temporal invÃ¡lido."
        )

    preliminary_train = (
        df
        .iloc[
            :split_index
        ]
        .copy()
    )

    test = (
        df
        .iloc[
            split_index:
        ]
        .copy()
    )

    test_start = (
        test[
            "origin_date"
        ]
        .min()
    )

    # --------------------------------------------------------
    # PURGING:
    #
    # Una observaciÃ³n de entrenamiento solamente puede
    # utilizarse si su target ya habrÃ­a ocurrido antes
    # de comenzar el periodo de test.
    # --------------------------------------------------------

    train = (
        preliminary_train
        .loc[
            preliminary_train[
                "target_date"
            ]
            < test_start
        ]
        .copy()
    )

    purged_rows = (
        len(
            preliminary_train
        )
        - len(train)
    )

    if train.empty or test.empty:
        raise ValueError(
            "Train o test vacÃ­o."
        )

    return (
        train.reset_index(
            drop=True
        ),
        test.reset_index(
            drop=True
        ),
        purged_rows,
    )


# ============================================================
# FEATURES
# ============================================================

def select_features(
    train,
    full_df,
):

    excluded_exact = {
        "origin_date",
        "target_date",
        "target_sleep_hours",
        "short_sleep_target",
        "horizon_days",
    }

    candidates = []

    for column in full_df.columns:

        if column in excluded_exact:
            continue

        # Cualquier campo que empiece por target_
        # queda fuera para proteger contra leakage.
        if column.startswith(
            "target_"
        ):
            continue

        candidates.append(
            column
        )

    feature_columns = []

    constant_columns = []

    all_null_columns = []

    for column in candidates:

        non_null = (
            train[column]
            .dropna()
        )

        if non_null.empty:

            all_null_columns.append(
                column
            )

            continue

        unique_count = (
            non_null
            .nunique()
        )

        if unique_count <= 1:

            constant_columns.append(
                column
            )

            continue

        feature_columns.append(
            column
        )

    if not feature_columns:
        raise ValueError(
            "No quedan features utilizables."
        )

    numeric_columns = []

    categorical_columns = []

    for column in feature_columns:

        if pd.api.types.is_numeric_dtype(
            train[column]
        ):
            numeric_columns.append(
                column
            )
        else:
            categorical_columns.append(
                column
            )

    return {
        "feature_columns":
            feature_columns,

        "numeric_columns":
            numeric_columns,

        "categorical_columns":
            categorical_columns,

        "constant_columns":
            constant_columns,

        "all_null_columns":
            all_null_columns,
    }


# ============================================================
# PREPROCESAMIENTO
# ============================================================

def build_preprocessor(
    numeric_columns,
    categorical_columns,
):

    transformers = []

    if numeric_columns:

        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                        add_indicator=True,
                    ),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
            ]
        )

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_columns,
            )
        )

    if categorical_columns:

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent",
                    ),
                ),
                (
                    "onehot",
                    make_one_hot_encoder(),
                ),
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


# ============================================================
# MODELOS
# ============================================================

def build_model(
    model_name,
    params,
):

    if model_name == "logistic_regression":

        return LogisticRegression(
            C=params["C"],
            max_iter=5000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        )

    if model_name == "random_forest":

        return RandomForestClassifier(
            n_estimators=500,
            max_depth=params[
                "max_depth"
            ],
            min_samples_leaf=params[
                "min_samples_leaf"
            ],
            max_features=params[
                "max_features"
            ],
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if model_name == "hist_gradient_boosting":

        return HistGradientBoostingClassifier(
            learning_rate=params[
                "learning_rate"
            ],
            max_leaf_nodes=params[
                "max_leaf_nodes"
            ],
            min_samples_leaf=params[
                "min_samples_leaf"
            ],
            l2_regularization=params[
                "l2_regularization"
            ],
            random_state=RANDOM_STATE,
        )

    raise ValueError(
        f"Modelo desconocido: {model_name}"
    )


def build_pipeline(
    model_name,
    params,
    numeric_columns,
    categorical_columns,
):

    preprocessor = (
        build_preprocessor(
            numeric_columns,
            categorical_columns,
        )
    )

    model = build_model(
        model_name,
        params,
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )


# ============================================================
# GRIDS
# ============================================================

MODEL_GRIDS = {

    "logistic_regression": {
        "C": [
            0.05,
            0.2,
            1.0,
            5.0,
        ],
    },

    "random_forest": {
        "max_depth": [
            3,
            5,
            None,
        ],

        "min_samples_leaf": [
            5,
            10,
        ],

        "max_features": [
            0.5,
            1.0,
        ],
    },

    "hist_gradient_boosting": {
        "learning_rate": [
            0.03,
            0.07,
        ],

        "max_leaf_nodes": [
            7,
            15,
        ],

        "min_samples_leaf": [
            10,
            20,
        ],

        "l2_regularization": [
            10.0,
        ],
    },
}


# ============================================================
# VALIDACIÃ“N TEMPORAL
# ============================================================

def evaluate_configuration_cv(
    train,
    feature_info,
    model_name,
    params,
):

    splitter = TimeSeriesSplit(
        n_splits=N_TEMPORAL_FOLDS
    )

    feature_columns = (
        feature_info[
            "feature_columns"
        ]
    )

    numeric_columns = (
        feature_info[
            "numeric_columns"
        ]
    )

    categorical_columns = (
        feature_info[
            "categorical_columns"
        ]
    )

    fold_metrics = []

    for fold_number, (
        raw_train_index,
        validation_index,
    ) in enumerate(
        splitter.split(train),
        start=1,
    ):

        raw_fold_train = (
            train
            .iloc[
                raw_train_index
            ]
            .copy()
        )

        validation = (
            train
            .iloc[
                validation_index
            ]
            .copy()
        )

        validation_start = (
            validation[
                "origin_date"
            ]
            .min()
        )

        # ----------------------------------------------------
        # PURGE tambiÃ©n dentro de cada fold.
        # ----------------------------------------------------

        fold_train = (
            raw_fold_train
            .loc[
                raw_fold_train[
                    "target_date"
                ]
                < validation_start
            ]
            .copy()
        )

        if len(fold_train) < 50:
            continue

        y_train = (
            fold_train[
                "short_sleep_target"
            ]
            .to_numpy(
                dtype=int
            )
        )

        y_validation = (
            validation[
                "short_sleep_target"
            ]
            .to_numpy(
                dtype=int
            )
        )

        if (
            len(
                np.unique(
                    y_train
                )
            )
            < 2
        ):
            continue

        pipeline = (
            build_pipeline(
                model_name,
                params,
                numeric_columns,
                categorical_columns,
            )
        )

        pipeline.fit(
            fold_train[
                feature_columns
            ],
            y_train,
        )

        probabilities = (
            pipeline
            .predict_proba(
                validation[
                    feature_columns
                ]
            )[:, 1]
        )

        model_metrics = (
            probability_metrics(
                y_validation,
                probabilities,
            )
        )

        # ----------------------------------------------------
        # BASELINE DEL FOLD
        #
        # Solo usa prevalencia conocida en fold_train.
        # ----------------------------------------------------

        baseline_probability = float(
            np.mean(
                y_train
            )
        )

        baseline_predictions = np.full(
            len(validation),
            baseline_probability,
            dtype=float,
        )

        baseline_metrics = (
            probability_metrics(
                y_validation,
                baseline_predictions,
            )
        )

        fold_metrics.append(
            {
                "fold":
                    fold_number,

                "train_rows":
                    int(
                        len(
                            fold_train
                        )
                    ),

                "validation_rows":
                    int(
                        len(
                            validation
                        )
                    ),

                "train_positive_pct":
                    percentage(
                        np.mean(
                            y_train
                        )
                    ),

                "validation_positive_pct":
                    percentage(
                        np.mean(
                            y_validation
                        )
                    ),

                "model_brier":
                    model_metrics[
                        "brier_score"
                    ],

                "model_log_loss":
                    model_metrics[
                        "log_loss"
                    ],

                "model_roc_auc":
                    model_metrics[
                        "roc_auc"
                    ],

                "model_pr_auc":
                    model_metrics[
                        "pr_auc"
                    ],

                "baseline_brier":
                    baseline_metrics[
                        "brier_score"
                    ],

                "baseline_log_loss":
                    baseline_metrics[
                        "log_loss"
                    ],
            }
        )

    if not fold_metrics:
        raise ValueError(
            f"No se pudo evaluar "
            f"{model_name}."
        )

    model_brier = mean_optional(
        [
            row["model_brier"]
            for row in fold_metrics
        ]
    )

    baseline_brier = mean_optional(
        [
            row["baseline_brier"]
            for row in fold_metrics
        ]
    )

    model_log = mean_optional(
        [
            row["model_log_loss"]
            for row in fold_metrics
        ]
    )

    baseline_log = mean_optional(
        [
            row["baseline_log_loss"]
            for row in fold_metrics
        ]
    )

    if (
        model_brier is not None
        and baseline_brier is not None
        and baseline_brier > 0
    ):

        brier_skill = (
            1
            - model_brier
            / baseline_brier
        )

    else:
        brier_skill = None

    return {
        "model":
            model_name,

        "params":
            params,

        "folds_used":
            len(
                fold_metrics
            ),

        "cv_brier_score":
            model_brier,

        "cv_log_loss":
            model_log,

        "cv_roc_auc":
            mean_optional(
                [
                    row[
                        "model_roc_auc"
                    ]
                    for row
                    in fold_metrics
                ]
            ),

        "cv_pr_auc":
            mean_optional(
                [
                    row[
                        "model_pr_auc"
                    ]
                    for row
                    in fold_metrics
                ]
            ),

        "cv_baseline_brier":
            baseline_brier,

        "cv_baseline_log_loss":
            baseline_log,

        "cv_brier_skill":
            brier_skill,

        "fold_details":
            fold_metrics,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print(
        "MODELOS DE CLASIFICACIÃ“N â€” "
        "RIESGO DE SUEÃ‘O CORTO"
    )
    print("=" * 78)

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_dataset()

    (
        train,
        test,
        purged_rows,
    ) = build_final_split(
        df
    )

    feature_info = (
        select_features(
            train,
            df,
        )
    )

    feature_columns = (
        feature_info[
            "feature_columns"
        ]
    )

    numeric_columns = (
        feature_info[
            "numeric_columns"
        ]
    )

    categorical_columns = (
        feature_info[
            "categorical_columns"
        ]
    )

    print()
    print("DATASET")
    print("-" * 78)

    print(
        f"Ejemplos totales:        "
        f"{len(df)}"
    )

    print(
        f"Train purgado:           "
        f"{len(train)}"
    )

    print(
        f"Test final:              "
        f"{len(test)}"
    )

    print(
        f"Filas purgadas:          "
        f"{purged_rows}"
    )

    print(
        f"Train:                   "
        f"{train['origin_date'].min().date()} "
        f"â†’ "
        f"{train['origin_date'].max().date()}"
    )

    print(
        f"Test:                    "
        f"{test['origin_date'].min().date()} "
        f"â†’ "
        f"{test['origin_date'].max().date()}"
    )

    print()
    print("FEATURES")
    print("-" * 78)

    print(
        f"Features utilizadas:     "
        f"{len(feature_columns)}"
    )

    print(
        f"NumÃ©ricas:               "
        f"{len(numeric_columns)}"
    )

    print(
        f"CategÃ³ricas:             "
        f"{len(categorical_columns)}"
    )

    print(
        f"Constantes eliminadas:   "
        f"{len(feature_info['constant_columns'])}"
    )

    print(
        f"Completamente nulas:     "
        f"{len(feature_info['all_null_columns'])}"
    )

    y_train = (
        train[
            "short_sleep_target"
        ]
        .to_numpy(
            dtype=int
        )
    )

    y_test = (
        test[
            "short_sleep_target"
        ]
        .to_numpy(
            dtype=int
        )
    )

    print()
    print("BALANCE")
    print("-" * 78)

    print(
        f"Train < 6 h:             "
        f"{y_train.sum()} / "
        f"{len(y_train)} "
        f"({percentage(y_train.mean()):.2f} %)"
    )

    print(
        f"Test < 6 h:              "
        f"{y_test.sum()} / "
        f"{len(y_test)} "
        f"({percentage(y_test.mean()):.2f} %)"
    )

    # ========================================================
    # GRID SEARCH TEMPORAL MANUAL
    # ========================================================

    all_results = []

    best_by_model = {}

    for model_name, grid in MODEL_GRIDS.items():

        print()
        print("=" * 78)
        print(
            f"VALIDANDO: "
            f"{model_name}"
        )
        print("=" * 78)

        model_results = []

        configurations = list(
            ParameterGrid(
                grid
            )
        )

        print(
            f"Configuraciones: "
            f"{len(configurations)}"
        )

        for number, params in enumerate(
            configurations,
            start=1,
        ):

            result = (
                evaluate_configuration_cv(
                    train,
                    feature_info,
                    model_name,
                    params,
                )
            )

            model_results.append(
                result
            )

            print(
                f"[{number:02d}/"
                f"{len(configurations):02d}] "
                f"Brier="
                f"{result['cv_brier_score']:.4f} "
                f"LogLoss="
                f"{result['cv_log_loss']:.4f} "
                f"Skill="
                f"{(
                    result['cv_brier_skill']
                    * 100
                    if result[
                        'cv_brier_skill'
                    ]
                    is not None
                    else float('nan')
                ):.2f}%"
            )

        model_results = sorted(
            model_results,
            key=lambda row: (
                row[
                    "cv_brier_score"
                ],
                row[
                    "cv_log_loss"
                ],
            ),
        )

        best = model_results[0]

        best_by_model[
            model_name
        ] = best

        all_results.extend(
            model_results
        )

        print()
        print(
            "Mejor configuraciÃ³n:"
        )

        print(
            json.dumps(
                best[
                    "params"
                ],
                ensure_ascii=False,
                indent=2,
            )
        )

        print(
            f"CV Brier: "
            f"{best['cv_brier_score']:.4f}"
        )

        print(
            f"CV baseline Brier: "
            f"{best['cv_baseline_brier']:.4f}"
        )

        print(
            f"CV Log Loss: "
            f"{best['cv_log_loss']:.4f}"
        )

        print(
            f"CV ROC-AUC: "
            f"{best['cv_roc_auc']:.4f}"
            if best[
                "cv_roc_auc"
            ]
            is not None
            else
            "CV ROC-AUC: n/a"
        )

        print(
            f"CV PR-AUC: "
            f"{best['cv_pr_auc']:.4f}"
            if best[
                "cv_pr_auc"
            ]
            is not None
            else
            "CV PR-AUC: n/a"
        )

    # ========================================================
    # SELECCIÃ“N DEL MEJOR CANDIDATO POR BRIER
    # ========================================================

    finalists = list(
        best_by_model.values()
    )

    finalists = sorted(
        finalists,
        key=lambda row: (
            row[
                "cv_brier_score"
            ],
            row[
                "cv_log_loss"
            ],
        ),
    )

    best_candidate = (
        finalists[0]
    )

    best_model_name = (
        best_candidate[
            "model"
        ]
    )

    best_params = (
        best_candidate[
            "params"
        ]
    )

    # ========================================================
    # BASELINE FINAL
    # ========================================================

    baseline_probability = float(
        np.mean(
            y_train
        )
    )

    baseline_predictions = np.full(
        len(test),
        baseline_probability,
        dtype=float,
    )

    baseline_metrics = (
        probability_metrics(
            y_test,
            baseline_predictions,
        )
    )

    # ========================================================
    # MODELO FINAL CANDIDATO
    # ========================================================

    final_pipeline = (
        build_pipeline(
            best_model_name,
            best_params,
            numeric_columns,
            categorical_columns,
        )
    )

    final_pipeline.fit(
        train[
            feature_columns
        ],
        y_train,
    )

    test_probabilities = (
        final_pipeline
        .predict_proba(
            test[
                feature_columns
            ]
        )[:, 1]
    )

    candidate_metrics = (
        probability_metrics(
            y_test,
            test_probabilities,
        )
    )

    if (
        baseline_metrics[
            "brier_score"
        ]
        > 0
    ):

        final_brier_skill = (
            1
            - candidate_metrics[
                "brier_score"
            ]
            / baseline_metrics[
                "brier_score"
            ]
        )

    else:
        final_brier_skill = None

    beats_baseline_brier = (
        candidate_metrics[
            "brier_score"
        ]
        < baseline_metrics[
            "brier_score"
        ]
    )

    beats_baseline_logloss = (
        candidate_metrics[
            "log_loss"
        ]
        < baseline_metrics[
            "log_loss"
        ]
    )

    # ========================================================
    # GUARDAR PREDICCIONES DEL TEST
    # ========================================================

    test_predictions = pd.DataFrame(
        {
            "origin_date":
                test[
                    "origin_date"
                ].dt.date,

            "target_date":
                test[
                    "target_date"
                ].dt.date,

            "actual_short_sleep":
                y_test,

            "actual_sleep_hours":
                test[
                    "target_sleep_hours"
                ].to_numpy(),

            "baseline_probability":
                baseline_predictions,

            "model_probability":
                test_probabilities,
        }
    )

    predictions_file = (
        REPORT_DIR
        / "short_sleep_test_predictions.csv"
    )

    test_predictions.to_csv(
        predictions_file,
        sep=";",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # GUARDAR GRID
    # ========================================================

    grid_rows = []

    for result in all_results:

        grid_rows.append(
            {
                "model":
                    result[
                        "model"
                    ],

                "params":
                    json.dumps(
                        result[
                            "params"
                        ],
                        ensure_ascii=False,
                        sort_keys=True,
                    ),

                "folds_used":
                    result[
                        "folds_used"
                    ],

                "cv_brier_score":
                    result[
                        "cv_brier_score"
                    ],

                "cv_log_loss":
                    result[
                        "cv_log_loss"
                    ],

                "cv_roc_auc":
                    result[
                        "cv_roc_auc"
                    ],

                "cv_pr_auc":
                    result[
                        "cv_pr_auc"
                    ],

                "cv_baseline_brier":
                    result[
                        "cv_baseline_brier"
                    ],

                "cv_baseline_log_loss":
                    result[
                        "cv_baseline_log_loss"
                    ],

                "cv_brier_skill":
                    result[
                        "cv_brier_skill"
                    ],
            }
        )

    grid_file = (
        REPORT_DIR
        / "short_sleep_model_cv_results.csv"
    )

    pd.DataFrame(
        grid_rows
    ).to_csv(
        grid_file,
        sep=";",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # GUARDAR MODELO CANDIDATO
    # ========================================================

    model_file = (
        MODEL_DIR
        / "candidate_short_sleep_model.joblib"
    )

    joblib.dump(
        {
            "pipeline":
                final_pipeline,

            "feature_columns":
                feature_columns,

            "model_name":
                best_model_name,

            "params":
                best_params,

            "short_sleep_threshold_hours":
                SHORT_SLEEP_THRESHOLD_HOURS,

            "training_end":
                str(
                    train[
                        "origin_date"
                    ]
                    .max()
                    .date()
                ),
        },
        model_file,
    )

    # ========================================================
    # INFORME JSON
    # ========================================================

    report = {

        "problem": {
            "target":
                "next-night sleep < 6 hours",

            "threshold_hours":
                SHORT_SLEEP_THRESHOLD_HOURS,
        },

        "dataset": {
            "total_examples":
                int(
                    len(df)
                ),

            "train_examples_after_purge":
                int(
                    len(train)
                ),

            "test_examples":
                int(
                    len(test)
                ),

            "purged_rows":
                int(
                    purged_rows
                ),

            "train_start":
                str(
                    train[
                        "origin_date"
                    ]
                    .min()
                    .date()
                ),

            "train_end":
                str(
                    train[
                        "origin_date"
                    ]
                    .max()
                    .date()
                ),

            "test_start":
                str(
                    test[
                        "origin_date"
                    ]
                    .min()
                    .date()
                ),

            "test_end":
                str(
                    test[
                        "origin_date"
                    ]
                    .max()
                    .date()
                ),

            "train_short_sleep_count":
                int(
                    y_train.sum()
                ),

            "train_short_sleep_pct":
                percentage(
                    y_train.mean()
                ),

            "test_short_sleep_count":
                int(
                    y_test.sum()
                ),

            "test_short_sleep_pct":
                percentage(
                    y_test.mean()
                ),
        },

        "features": {
            "feature_count":
                int(
                    len(
                        feature_columns
                    )
                ),

            "numeric_count":
                int(
                    len(
                        numeric_columns
                    )
                ),

            "categorical_count":
                int(
                    len(
                        categorical_columns
                    )
                ),

            "constant_columns_removed":
                feature_info[
                    "constant_columns"
                ],

            "all_null_columns_removed":
                feature_info[
                    "all_null_columns"
                ],
        },

        "best_cv_by_model":
            best_by_model,

        "selected_candidate": {
            "model":
                best_model_name,

            "params":
                best_params,

            "cv_brier_score":
                best_candidate[
                    "cv_brier_score"
                ],

            "cv_baseline_brier":
                best_candidate[
                    "cv_baseline_brier"
                ],

            "cv_log_loss":
                best_candidate[
                    "cv_log_loss"
                ],

            "cv_roc_auc":
                best_candidate[
                    "cv_roc_auc"
                ],

            "cv_pr_auc":
                best_candidate[
                    "cv_pr_auc"
                ],
        },

        "final_holdout": {

            "baseline": {
                "probability":
                    baseline_probability,

                "probability_pct":
                    percentage(
                        baseline_probability
                    ),

                **baseline_metrics,
            },

            "candidate": {
                "model":
                    best_model_name,

                **candidate_metrics,

                "brier_skill_vs_baseline":
                    final_brier_skill,

                "beats_baseline_brier":
                    bool(
                        beats_baseline_brier
                    ),

                "beats_baseline_log_loss":
                    bool(
                        beats_baseline_logloss
                    ),
            },
        },

        "status":
            (
                "candidate_only_not_production"
            ),
    }

    report_file = (
        REPORT_DIR
        / "short_sleep_model_evaluation.json"
    )

    with report_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    # ========================================================
    # TERMINAL
    # ========================================================

    print()
    print("=" * 78)
    print("MEJOR CANDIDATO")
    print("=" * 78)

    print(
        f"Modelo:                  "
        f"{best_model_name}"
    )

    print(
        f"ParÃ¡metros:              "
        f"{best_params}"
    )

    print()
    print("VALIDACIÃ“N TEMPORAL")
    print("-" * 78)

    print(
        f"CV Brier candidato:      "
        f"{best_candidate['cv_brier_score']:.4f}"
    )

    print(
        f"CV Brier baseline:       "
        f"{best_candidate['cv_baseline_brier']:.4f}"
    )

    print(
        f"CV Log Loss candidato:   "
        f"{best_candidate['cv_log_loss']:.4f}"
    )

    print(
        f"CV ROC-AUC:              "
        f"{best_candidate['cv_roc_auc']:.4f}"
        if best_candidate[
            "cv_roc_auc"
        ]
        is not None
        else
        "CV ROC-AUC:              n/a"
    )

    print(
        f"CV PR-AUC:               "
        f"{best_candidate['cv_pr_auc']:.4f}"
        if best_candidate[
            "cv_pr_auc"
        ]
        is not None
        else
        "CV PR-AUC:               n/a"
    )

    print()
    print("TEST FINAL")
    print("-" * 78)

    print(
        f"Baseline probabilidad:   "
        f"{baseline_probability * 100:.2f} %"
    )

    print()
    print("Baseline")
    print(
        f"  Brier Score:           "
        f"{baseline_metrics['brier_score']:.4f}"
    )

    print(
        f"  Log Loss:              "
        f"{baseline_metrics['log_loss']:.4f}"
    )

    print()
    print(best_model_name)

    print(
        f"  Brier Score:           "
        f"{candidate_metrics['brier_score']:.4f}"
    )

    print(
        f"  Log Loss:              "
        f"{candidate_metrics['log_loss']:.4f}"
    )

    print(
        f"  ROC-AUC:               "
        f"{candidate_metrics['roc_auc']:.4f}"
        if candidate_metrics[
            "roc_auc"
        ]
        is not None
        else
        "  ROC-AUC:               n/a"
    )

    print(
        f"  PR-AUC:                "
        f"{candidate_metrics['pr_auc']:.4f}"
        if candidate_metrics[
            "pr_auc"
        ]
        is not None
        else
        "  PR-AUC:                n/a"
    )

    if final_brier_skill is not None:

        print(
            f"  Brier Skill:           "
            f"{final_brier_skill * 100:+.2f} %"
        )

    print()
    print("COMPARACIÃ“N")
    print("-" * 78)

    print(
        "Supera baseline en Brier:"
        f" {'SÃ' if beats_baseline_brier else 'NO'}"
    )

    print(
        "Supera baseline en Log Loss:"
        f" {'SÃ' if beats_baseline_logloss else 'NO'}"
    )

    print()
    print("=" * 78)
    print("ARCHIVOS GENERADOS")
    print("=" * 78)

    print(
        report_file
    )

    print(
        grid_file
    )

    print(
        predictions_file
    )

    print(
        model_file
    )

    print()
    print(
        "IMPORTANTE:"
    )

    print(
        "El modelo guardado sigue siendo un "
        "CANDIDATO."
    )

    print(
        "TodavÃ­a NO debe utilizarse en producciÃ³n."
    )

    print(
        "El siguiente paso serÃ¡ estudiar la "
        "calibraciÃ³n de las probabilidades."
    )

    print("=" * 78)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
