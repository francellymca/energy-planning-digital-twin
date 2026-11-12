from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from xgboost import XGBRegressor


UF_ORDER = ["BA", "MG", "PR", "RJ", "RS", "SP"]

TARGET = "consumo_mwh"

TRAIN_START_YEAR = 2016
FIT_END_YEAR = 2022
VALIDATION_YEAR = 2023
VALIDATION_YEARS = (2021, 2022, 2023)
TEST_START_YEAR = 2024
TEST_END_YEAR = 2025

RANDOM_SEED = 42


TIME_FEATURES = [
    "mes_seno",
    "mes_cosseno",
    "tendencia_mensal",
]


TARGET_HISTORY_FEATURES = [
    "consumo_mwh_lag_1",
    "consumo_mwh_lag_3",
    "consumo_mwh_lag_6",
    "consumo_mwh_lag_12",
    "consumo_mwh_media_movel_3",
    "consumo_mwh_media_movel_6",
    "consumo_mwh_media_movel_12",
]


ENERGY_CLIMATE_FEATURES = [
    "carga_mwmed_media_mes_lag_1",
    "temperatura_media_c_lag_1",
    "temperatura_media_c_lag_12",
    "umidade_media_pct_lag_1",
    "umidade_media_pct_lag_12",
    "precipitacao_mediana_mm_lag_1",
    "precipitacao_mediana_mm_lag_12",
    "precipitacao_max_24h_mm_lag_1",
    "precipitacao_max_24h_mm_lag_12",
]


SOCIOECONOMIC_FEATURES = [
    "populacao_ultimo_disponivel",
    "pib_corrente_ultimo_disponivel",
    "participacao_pib_ultimo_disponivel",
    "indice_volume_pib_ultimo_disponivel",
]


FEATURE_GROUPS = {
    "historico": (
        TIME_FEATURES
        + TARGET_HISTORY_FEATURES
    ),
    "energia_clima": (
        TIME_FEATURES
        + TARGET_HISTORY_FEATURES
        + ENERGY_CLIMATE_FEATURES
    ),
    "completo": (
        TIME_FEATURES
        + TARGET_HISTORY_FEATURES
        + ENERGY_CLIMATE_FEATURES
        + SOCIOECONOMIC_FEATURES
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Treina e avalia modelos XGBoost multivariados "
            "para previsão mensal do consumo de energia."
        )
    )

    parser.add_argument(
        "--features-file",
        type=Path,
        default=Path(
            "data/processed/modelagem/"
            "features_modelagem_2016_2025.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/modelagem"),
    )

    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("docs/figures"),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
    )

    return parser.parse_args()
def load_and_validate_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo de atributos não encontrado: {path}"
        )

    data = pd.read_csv(path)

    required_columns = {
        "mes",
        "ano",
        "uf",
        TARGET,
        "amostra_socioeconomica_completa",
    }

    for features in FEATURE_GROUPS.values():
        required_columns.update(features)

    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        raise ValueError(
            "Colunas obrigatórias ausentes: "
            f"{sorted(missing_columns)}"
        )

    data["mes"] = pd.to_datetime(
        data["mes"],
        errors="raise",
    )

    data["ano"] = pd.to_numeric(
        data["ano"],
        errors="raise",
    ).astype(int)

    data = (
        data.loc[
            data["uf"].isin(UF_ORDER)
            & data["ano"].between(
                TRAIN_START_YEAR,
                TEST_END_YEAR,
            )
        ]
        .sort_values(["uf", "mes"])
        .reset_index(drop=True)
    )

    duplicated = data.duplicated(
        subset=["uf", "mes"],
        keep=False,
    )

    if duplicated.any():
        duplicated_rows = data.loc[
            duplicated,
            ["uf", "mes"],
        ]

        raise ValueError(
            "Foram encontrados registros duplicados:\n"
            f"{duplicated_rows.to_string(index=False)}"
        )

    expected_rows = (
        len(UF_ORDER)
        * (TEST_END_YEAR - TRAIN_START_YEAR + 1)
        * 12
    )

    if len(data) != expected_rows:
        raise ValueError(
            f"Eram esperados {expected_rows} registros, "
            f"mas foram encontrados {len(data)}."
        )

    rows_per_state = data.groupby("uf").size()

    if not rows_per_state.eq(120).all():
        raise ValueError(
            "Quantidade mensal inesperada por estado:\n"
            f"{rows_per_state.to_string()}"
        )

    sample = data.loc[
        data["amostra_socioeconomica_completa"].eq(1)
    ].copy()

    expected_model_rows = 570 + 144

    if len(sample) != expected_model_rows:
        raise ValueError(
            f"A amostra completa deveria conter "
            f"{expected_model_rows} registros, "
            f"mas contém {len(sample)}."
        )

    return sample.reset_index(drop=True)


def build_design_matrix(
    frame: pd.DataFrame,
    numeric_features: list[str],
) -> pd.DataFrame:
    numeric = (
        frame.loc[:, numeric_features]
        .apply(pd.to_numeric, errors="raise")
        .astype(np.float64)
        .reset_index(drop=True)
    )

    state_category = pd.Categorical(
        frame["uf"],
        categories=UF_ORDER,
        ordered=True,
    )

    state_dummies = pd.get_dummies(
        state_category,
        prefix="uf",
        dtype=np.float64,
    )

    expected_state_columns = [
        f"uf_{uf}"
        for uf in UF_ORDER
    ]

    state_dummies = (
        state_dummies
        .reindex(
            columns=expected_state_columns,
            fill_value=0.0,
        )
        .reset_index(drop=True)
    )

    design = pd.concat(
        [numeric, state_dummies],
        axis=1,
    )

    values = design.to_numpy(dtype=np.float64)

    if not np.isfinite(values).all():
        invalid_columns = design.columns[
            ~np.isfinite(values).all(axis=0)
        ].tolist()

        raise ValueError(
            "A matriz de atributos contém valores "
            f"não finitos nas colunas: {invalid_columns}"
        )

    return design

def split_temporal_sample(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fit = data.loc[
        data["ano"].between(
            TRAIN_START_YEAR,
            FIT_END_YEAR,
        )
    ].copy()

    validation = data.loc[
        data["ano"].eq(VALIDATION_YEAR)
    ].copy()

    test = data.loc[
        data["ano"].between(
            TEST_START_YEAR,
            TEST_END_YEAR,
        )
    ].copy()

    expected_sizes = {
        "ajuste": 498,
        "validação": 72,
        "teste": 144,
    }

    observed_sizes = {
        "ajuste": len(fit),
        "validação": len(validation),
        "teste": len(test),
    }

    if observed_sizes != expected_sizes:
        raise ValueError(
            "Tamanhos temporais inesperados. "
            f"Esperado: {expected_sizes}; "
            f"observado: {observed_sizes}."
        )

    return fit, validation, test

def calculate_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    observed = np.asarray(
        observed,
        dtype=np.float64,
    )

    predicted = np.asarray(
        predicted,
        dtype=np.float64,
    )

    mae = float(
        mean_absolute_error(
            observed,
            predicted,
        )
    )

    rmse = float(
        np.sqrt(
            mean_squared_error(
                observed,
                predicted,
            )
        )
    )

    nonzero = observed != 0

    mape = float(
        mean_absolute_percentage_error(
            observed[nonzero],
            predicted[nonzero],
        )
        * 100
    )

    return {
        "mae_mwh": mae,
        "rmse_mwh": rmse,
        "mape_pct": mape,
    }


HYPERPARAMETER_CANDIDATES = [
    {
        "configuracao": "conservadora",
        "max_depth": 2,
        "min_child_weight": 8,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.10,
        "reg_lambda": 8.0,
    },
    {
        "configuracao": "equilibrada",
        "max_depth": 3,
        "min_child_weight": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.05,
        "reg_lambda": 5.0,
    },
    {
        "configuracao": "flexivel",
        "max_depth": 3,
        "min_child_weight": 2,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "reg_alpha": 0.01,
        "reg_lambda": 3.0,
    },
]

def transform_target(
    frame: pd.DataFrame,
) -> np.ndarray:
    observed = frame[
        TARGET
    ].to_numpy(dtype=np.float64)

    seasonal_reference = frame[
        "consumo_mwh_lag_12"
    ].to_numpy(dtype=np.float64)

    if (observed <= 0).any():
        raise ValueError(
            "O consumo deve ser positivo para "
            "a transformação logarítmica."
        )

    if (seasonal_reference <= 0).any():
        raise ValueError(
            "A referência sazonal deve ser positiva."
        )

    return np.log(
        observed / seasonal_reference
    )


def restore_prediction(
    predicted_log_factor: np.ndarray,
    frame: pd.DataFrame,
) -> np.ndarray:
    seasonal_reference = frame[
        "consumo_mwh_lag_12"
    ].to_numpy(dtype=np.float64)

    predicted_factor = np.exp(
        np.asarray(
            predicted_log_factor,
            dtype=np.float64,
        )
    )

    forecast = (
        seasonal_reference
        * predicted_factor
    )

    return np.maximum(
        forecast,
        0.0,
    )


def build_model(
    parameters: dict[str, object],
    n_estimators: int,
    seed: int,
    use_early_stopping: bool,
) -> XGBRegressor:
    model_arguments = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": n_estimators,
        "learning_rate": 0.03,
        "max_depth": parameters["max_depth"],
        "min_child_weight": parameters["min_child_weight"],
        "subsample": parameters["subsample"],
        "colsample_bytree": parameters["colsample_bytree"],
        "reg_alpha": parameters["reg_alpha"],
        "reg_lambda": parameters["reg_lambda"],
        "random_state": seed,
        "n_jobs": -1,
        "tree_method": "hist",
    }

    if use_early_stopping:
        model_arguments["early_stopping_rounds"] = 50

    return XGBRegressor(**model_arguments)


def select_configuration(
    group_name: str,
    features: list[str],
    development: pd.DataFrame,
    seed: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    validation_rows: list[dict[str, object]] = []
    candidate_summaries: list[dict[str, object]] = []

    for candidate in HYPERPARAMETER_CANDIDATES:
        fold_metrics: list[dict[str, object]] = []

        for validation_year in VALIDATION_YEARS:
            fold_fit = development.loc[
                development["ano"].lt(validation_year)
            ].copy()

            fold_validation = development.loc[
                development["ano"].eq(validation_year)
            ].copy()

            if len(fold_validation) != 72:
                raise ValueError(
                    f"{group_name}: validação de "
                    f"{validation_year} deveria conter "
                    f"72 registros, mas contém "
                    f"{len(fold_validation)}."
                )

            x_fit = build_design_matrix(
                fold_fit,
                features,
            )

            x_validation = build_design_matrix(
                fold_validation,
                features,
            )

            y_fit = transform_target(
                fold_fit
            )

            y_validation = transform_target(
                fold_validation
            )

            model = build_model(
                parameters=candidate,
                n_estimators=1200,
                seed=seed,
                use_early_stopping=True,
            )

            model.fit(
                x_fit,
                y_fit,
                eval_set=[
                    (
                        x_validation,
                        y_validation,
                    )
                ],
                verbose=False,
            )

            best_iteration = int(
                model.best_iteration
            )

            prediction = restore_prediction(
                predicted_log_factor=model.predict(
                    x_validation
                ),
                frame=fold_validation,
            )

            metrics = calculate_metrics(
                fold_validation[TARGET].to_numpy(
                    dtype=np.float64
                ),
                prediction,
            )

            fold_record = {
                "tipo_registro": "dobra_temporal",
                "grupo_atributos": group_name,
                "configuracao": candidate[
                    "configuracao"
                ],
                "ano_validacao": validation_year,
                "inicio_ajuste": int(
                    fold_fit["ano"].min()
                ),
                "fim_ajuste": int(
                    fold_fit["ano"].max()
                ),
                "observacoes_ajuste": len(
                    fold_fit
                ),
                "observacoes_validacao": len(
                    fold_validation
                ),
                "melhor_iteracao": best_iteration,
                "numero_arvores": (
                    best_iteration + 1
                ),
                "numero_atributos": x_fit.shape[1],
                "mae_validacao_mwh": (
                    metrics["mae_mwh"]
                ),
                "rmse_validacao_mwh": (
                    metrics["rmse_mwh"]
                ),
                "mape_validacao_pct": (
                    metrics["mape_pct"]
                ),
                "parametros": json.dumps(
                    candidate,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }

            fold_metrics.append(fold_record)
            validation_rows.append(fold_record)

        mean_mae = float(
            np.mean(
                [
                    row["mae_validacao_mwh"]
                    for row in fold_metrics
                ]
            )
        )

        mean_rmse = float(
            np.mean(
                [
                    row["rmse_validacao_mwh"]
                    for row in fold_metrics
                ]
            )
        )

        mean_mape = float(
            np.mean(
                [
                    row["mape_validacao_pct"]
                    for row in fold_metrics
                ]
            )
        )

        worst_mape = float(
            np.max(
                [
                    row["mape_validacao_pct"]
                    for row in fold_metrics
                ]
            )
        )

        tree_counts = [
            row["numero_arvores"]
            for row in fold_metrics
        ]

        selected_trees = max(
            1,
            int(np.median(tree_counts)),
        )

        summary = {
            "tipo_registro": "resumo_configuracao",
            "grupo_atributos": group_name,
            "configuracao": candidate[
                "configuracao"
            ],
            "ano_validacao": "2021-2023",
            "inicio_ajuste": TRAIN_START_YEAR,
            "fim_ajuste": 2022,
            "observacoes_ajuste": np.nan,
            "observacoes_validacao": 216,
            "melhor_iteracao": (
                selected_trees - 1
            ),
            "numero_arvores": selected_trees,
            "numero_atributos": (
                fold_metrics[-1][
                    "numero_atributos"
                ]
            ),
            "mae_validacao_mwh": mean_mae,
            "rmse_validacao_mwh": mean_rmse,
            "mape_validacao_pct": mean_mape,
            "pior_mape_validacao_pct": worst_mape,
            "parametros": json.dumps(
                candidate,
                ensure_ascii=False,
                sort_keys=True,
            ),
        }

        validation_rows.append(summary)
        candidate_summaries.append(summary)

    ordered_summaries = sorted(
        candidate_summaries,
        key=lambda row: (
            row["mape_validacao_pct"],
            row["pior_mape_validacao_pct"],
            row["rmse_validacao_mwh"],
        ),
    )

    best_result = ordered_summaries[0].copy()

    best_parameters = next(
        candidate
        for candidate in HYPERPARAMETER_CANDIDATES
        if candidate["configuracao"]
        == best_result["configuracao"]
    )

    selection = {
        **best_result,
        "parametros_modelo": (
            best_parameters.copy()
        ),
    }

    return selection, validation_rows

def train_final_model(
    group_name: str,
    features: list[str],
    selection: dict[str, object],
    fit: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    seed: int,
) -> tuple[
    XGBRegressor,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    train = (
        pd.concat(
            [fit, validation],
            ignore_index=True,
        )
        .sort_values(["uf", "mes"])
        .reset_index(drop=True)
    )

    x_train = build_design_matrix(
        train,
        features,
    )

    x_test = build_design_matrix(
        test,
        features,
    )

    y_train = transform_target(train)

    number_of_trees = int(
        selection["numero_arvores"]
    )

    model = build_model(
        parameters=selection["parametros_modelo"],
        n_estimators=number_of_trees,
        seed=seed,
        use_early_stopping=False,
    )

    model.fit(
        x_train,
        y_train,
        verbose=False,
    )

    predicted_log_factor = model.predict(
        x_test
    )

    predicted_factor = np.exp(
        predicted_log_factor
    )

    seasonal_reference = test[
        "consumo_mwh_lag_12"
    ].to_numpy(dtype=np.float64)

    forecast = restore_prediction(
        predicted_log_factor=predicted_log_factor,
        frame=test,
    )

    observed = test[
        TARGET
    ].to_numpy(dtype=np.float64)

    error = observed - forecast
    absolute_error = np.abs(error)

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        percentage_error = np.where(
            observed != 0,
            absolute_error / np.abs(observed) * 100,
            np.nan,
        )

    predictions = pd.DataFrame(
        {
            "mes": test["mes"].to_numpy(),
            "uf": test["uf"].to_numpy(),
            "observado_mwh": observed,
            "referencia_sazonal_mwh": seasonal_reference,
            "fator_correcao_previsto": predicted_factor,
            "previsto_xgboost_mwh": forecast,
            "erro_mwh": error,
            "erro_absoluto_mwh": absolute_error,
            "erro_percentual_absoluto_pct": percentage_error,
            "grupo_atributos": group_name,
            "configuracao": selection["configuracao"],
            "numero_arvores": number_of_trees,
            "estrategia_teste": (
                "origem_movel_um_passo_a_frente"
            ),
        }
    )

    return (
        model,
        x_train,
        x_test,
        predictions,
    )


def build_metric_rows(
    predictions: pd.DataFrame,
) -> list[dict[str, object]]:
    metric_rows: list[dict[str, object]] = []

    group_name = str(
        predictions["grupo_atributos"].iloc[0]
    )

    configuration = str(
        predictions["configuracao"].iloc[0]
    )

    for uf in [*UF_ORDER, "GERAL"]:
        if uf == "GERAL":
            state = predictions
        else:
            state = predictions.loc[
                predictions["uf"].eq(uf)
            ]

        observed = state[
            "observado_mwh"
        ].to_numpy(dtype=np.float64)

        predicted = state[
            "previsto_xgboost_mwh"
        ].to_numpy(dtype=np.float64)

        metrics = calculate_metrics(
            observed,
            predicted,
        )

        metric_rows.append(
            {
                "uf": uf,
                "modelo": "XGBoost",
                "grupo_atributos": group_name,
                "configuracao": configuration,
                "observacoes": len(state),
                "mae_mwh": metrics["mae_mwh"],
                "rmse_mwh": metrics["rmse_mwh"],
                "mape_pct": metrics["mape_pct"],
            }
        )

    return metric_rows


def build_model_parameters(
    group_name: str,
    features: list[str],
    selection: dict[str, object],
    x_train: pd.DataFrame,
    model: XGBRegressor,
    seed: int,
) -> dict[str, object]:
    booster = model.get_booster()

    return {
        "grupo_atributos": group_name,
        "configuracao": selection["configuracao"],
        "seed": seed,
        "transformacao_alvo": (
            "log_consumo_sobre_lag_sazonal_12"
        ),
        "criterio_selecao": (
            "menor_mape_medio_validacao_temporal_2021_2023"
        ),
        "periodo_ajuste_selecao": (
            "origens_moveis_ate_2020_2021_2022"
        ),
        "periodo_validacao": "2021-2023",
        "periodo_treinamento_final": "2016-2023",
        "periodo_teste": "2024-2025",
        "estrategia_teste": (
            "origem_movel_um_passo_a_frente"
        ),
        "observacoes_treinamento_final": (
            x_train.shape[0]
        ),
        "numero_atributos": x_train.shape[1],
        "numero_arvores": int(
            selection["numero_arvores"]
        ),
        "melhor_iteracao_validacao": int(
            selection["melhor_iteracao"]
        ),
        "mae_validacao_mwh": float(
            selection["mae_validacao_mwh"]
        ),
        "rmse_validacao_mwh": float(
            selection["rmse_validacao_mwh"]
        ),
        "mape_validacao_pct": float(
            selection["mape_validacao_pct"]
        ),
        "atributos": json.dumps(
            list(x_train.columns),
            ensure_ascii=False,
        ),
        "parametros": json.dumps(
            selection["parametros_modelo"],
            ensure_ascii=False,
            sort_keys=True,
        ),
        "numero_nos_modelo": int(
            booster.trees_to_dataframe().shape[0]
        ),
    }
    
def calculate_shap_results(
    model: XGBRegressor,
    x_test: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    explainer = shap.TreeExplainer(model)

    explanation = explainer(x_test)

    shap_values = np.asarray(
        explanation.values,
        dtype=np.float64,
    )

    if shap_values.shape != x_test.shape:
        raise ValueError(
            "Dimensão inesperada dos valores SHAP. "
            f"SHAP: {shap_values.shape}; "
            f"atributos: {x_test.shape}."
        )

    identity = (
        test.loc[:, ["mes", "uf"]]
        .reset_index(drop=True)
    )

    shap_wide = pd.concat(
        [
            identity,
            pd.DataFrame(
                shap_values,
                columns=x_test.columns,
            ),
        ],
        axis=1,
    )

    feature_wide = pd.concat(
        [
            identity,
            x_test.reset_index(drop=True),
        ],
        axis=1,
    )

    shap_long = shap_wide.melt(
        id_vars=["mes", "uf"],
        var_name="atributo",
        value_name="valor_shap",
    )

    feature_long = feature_wide.melt(
        id_vars=["mes", "uf"],
        var_name="atributo",
        value_name="valor_atributo",
    )

    shap_long = shap_long.merge(
        feature_long,
        on=["mes", "uf", "atributo"],
        how="left",
        validate="one_to_one",
    )

    shap_long["valor_shap_absoluto"] = (
        shap_long["valor_shap"].abs()
    )

    global_importance = (
        shap_long.groupby(
            "atributo",
            as_index=False,
        )
        .agg(
            media_shap_absoluto=(
                "valor_shap_absoluto",
                "mean",
            ),
            media_shap=(
                "valor_shap",
                "mean",
            ),
        )
        .sort_values(
            "media_shap_absoluto",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    state_importance = (
        shap_long.groupby(
            ["uf", "atributo"],
            as_index=False,
        )
        .agg(
            media_shap_absoluto=(
                "valor_shap_absoluto",
                "mean",
            ),
            media_shap=(
                "valor_shap",
                "mean",
            ),
        )
        .sort_values(
            ["uf", "media_shap_absoluto"],
            ascending=[True, False],
        )
        .reset_index(drop=True)
    )

    return (
        shap_long,
        global_importance,
        state_importance,
    )


def plot_global_shap_importance(
    importance: pd.DataFrame,
    output: Path,
    top_n: int = 15,
) -> None:
    feature_labels = {
        "tendencia_mensal": "Tendência mensal",
        "mes_seno": "Sazonalidade (seno)",
        "mes_cosseno": "Sazonalidade (cosseno)",
        "consumo_mwh_lag_1": (
            "Consumo – defasagem de 1 mês"
        ),
        "consumo_mwh_lag_3": (
            "Consumo – defasagem de 3 meses"
        ),
        "consumo_mwh_lag_6": (
            "Consumo – defasagem de 6 meses"
        ),
        "consumo_mwh_lag_12": (
            "Consumo – defasagem de 12 meses"
        ),
        "umidade_media_pct_lag_12": (
            "Umidade média – defasagem de 12 meses"
        ),
        "carga_mwmed_media_mes_lag_1": (
            "Carga média – defasagem de 1 mês"
        ),
        "uf_RJ": "Indicador estadual – RJ",
    }

    selected = (
        importance.loc[
            importance[
                "media_shap_absoluto"
            ].gt(0)
        ]
        .head(top_n)
        .sort_values(
            "media_shap_absoluto",
            ascending=True,
        )
        .copy()
    )

    if selected.empty:
        raise ValueError(
            "Nenhum atributo apresentou importância "
            "SHAP diferente de zero."
        )

    selected["rotulo"] = (
        selected["atributo"]
        .map(feature_labels)
        .fillna(
            selected["atributo"].str.replace(
                "_",
                " ",
                regex=False,
            )
        )
    )

    figure_height = max(
        4.8,
        0.62 * len(selected) + 1.8,
    )

    fig, axis = plt.subplots(
        figsize=(10.5, figure_height)
    )

    bars = axis.barh(
        selected["rotulo"],
        selected["media_shap_absoluto"],
        color="#2A9D8F",
    )

    axis.bar_label(
        bars,
        labels=[
            f"{value:.6f}"
            for value in selected[
                "media_shap_absoluto"
            ]
        ],
        padding=4,
        fontsize=9,
    )

    axis.set_xlabel(
        (
            "Média de |SHAP| no log do "
            "fator de correção sazonal"
        ),
        fontsize=11,
    )

    axis.set_ylabel(
        "Atributo",
        fontsize=11,
    )

    axis.set_title(
        (
            "Importância global dos atributos no "
            "XGBoost de energia e clima"
        ),
        fontsize=14,
        fontweight="bold",
    )

    axis.grid(
        axis="x",
        alpha=0.25,
    )

    axis.set_axisbelow(True)

    maximum = float(
        selected["media_shap_absoluto"].max()
    )

    axis.set_xlim(
        0,
        maximum * 1.18,
    )

    fig.tight_layout()

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)
    
def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = load_and_validate_data(
        args.features_file
    )

    fit, validation, test = split_temporal_sample(
        data
    )

    all_validation_results: list[
        dict[str, object]
    ] = []

    all_predictions: list[pd.DataFrame] = []
    all_metric_rows: list[dict[str, object]] = []
    all_parameter_rows: list[dict[str, object]] = []

    selections: dict[
        str,
        dict[str, object],
    ] = {}

    trained_models: dict[
        str,
        XGBRegressor,
    ] = {}

    test_matrices: dict[
        str,
        pd.DataFrame,
    ] = {}
    
    development = (
        pd.concat(
            [fit, validation],
            ignore_index=True,
        )
        .sort_values(["uf", "mes"])
        .reset_index(drop=True)
    )

    for group_name, features in FEATURE_GROUPS.items():
        print(
            f"{group_name}: selecionando configuração..."
        )

        selection, validation_results = (
            select_configuration(
                group_name=group_name,
                features=features,
                development=development,
                seed=args.seed,
            )
        )

        selections[group_name] = selection

        all_validation_results.extend(
            validation_results
        )

        print(
            f"{group_name}: "
            f"configuração={selection['configuracao']}; "
            f"árvores={selection['numero_arvores']}; "
            f"MAPE validação="
            f"{selection['mape_validacao_pct']:.3f}%"
        )

        (
            model,
            x_train,
            x_test,
            predictions,
        ) = train_final_model(
            group_name=group_name,
            features=features,
            selection=selection,
            fit=fit,
            validation=validation,
            test=test,
            seed=args.seed,
        )

        trained_models[group_name] = model
        test_matrices[group_name] = x_test

        all_predictions.append(predictions)

        all_metric_rows.extend(
            build_metric_rows(predictions)
        )

        all_parameter_rows.append(
            build_model_parameters(
                group_name=group_name,
                features=features,
                selection=selection,
                x_train=x_train,
                model=model,
                seed=args.seed,
            )
        )

    best_group = min(
        selections,
        key=lambda name: (
            selections[name][
                "mape_validacao_pct"
            ],
            selections[name][
                "rmse_validacao_mwh"
            ],
        ),
    )

    print(
        "\nModelo selecionado para XAI: "
        f"{best_group}"
    )

    predictions_table = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    metrics_table = pd.DataFrame(
        all_metric_rows
    )

    validation_table = pd.DataFrame(
        all_validation_results
    )

    parameters_table = pd.DataFrame(
        all_parameter_rows
    )

    (
        shap_values,
        global_importance,
        state_importance,
    ) = calculate_shap_results(
        model=trained_models[best_group],
        x_test=test_matrices[best_group],
        test=test,
    )

    predictions_path = (
        args.output_dir
        / "xgboost_multivariado_previsoes_2024_2025.csv"
    )

    metrics_path = (
        args.output_dir
        / "metricas_xgboost_multivariado_2024_2025.csv"
    )

    validation_path = (
        args.output_dir
        / "validacao_xgboost_multivariado_2021_2023.csv"
    )

    parameters_path = (
        args.output_dir
        / "parametros_xgboost_multivariado.csv"
    )

    shap_values_path = (
        args.output_dir
        / "shap_valores_xgboost_2024_2025.csv"
    )

    global_importance_path = (
        args.output_dir
        / "shap_importancia_global_xgboost.csv"
    )

    state_importance_path = (
        args.output_dir
        / "shap_importancia_por_uf_xgboost.csv"
    )

    shap_figure_path = (
        args.figures_dir
        / "xgboost_shap_importancia_global.png"
    )

    predictions_table.to_csv(
        predictions_path,
        index=False,
        date_format="%Y-%m-%d",
    )

    metrics_table.to_csv(
        metrics_path,
        index=False,
    )

    validation_table.to_csv(
        validation_path,
        index=False,
    )

    parameters_table.to_csv(
        parameters_path,
        index=False,
    )

    shap_values.to_csv(
        shap_values_path,
        index=False,
        date_format="%Y-%m-%d",
    )

    global_importance.to_csv(
        global_importance_path,
        index=False,
    )

    state_importance.to_csv(
        state_importance_path,
        index=False,
    )

    plot_global_shap_importance(
        importance=global_importance,
        output=shap_figure_path,
    )

    print(
        "\nMAPE por estado e grupo de atributos:\n"
    )

    print(
        metrics_table.pivot(
            index="uf",
            columns="grupo_atributos",
            values="mape_pct",
        )
        .reindex([*UF_ORDER, "GERAL"])
        .round(3)
        .to_string()
    )

    print(
        "\nArquivos gerados:"
        f"\nPrevisões: {predictions_path}"
        f"\nMétricas: {metrics_path}"
        f"\nValidação: {validation_path}"
        f"\nParâmetros: {parameters_path}"
        f"\nSHAP detalhado: {shap_values_path}"
        f"\nSHAP global: {global_importance_path}"
        f"\nSHAP por estado: {state_importance_path}"
        f"\nFigura SHAP: {shap_figure_path}"
    )


if __name__ == "__main__":
    main()