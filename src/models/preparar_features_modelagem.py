"""Prepara atributos temporalmente seguros para os modelos de previsão.

A base integrada permanece inalterada. Este script cria defasagens, médias
móveis e indicadores socioeconômicos conhecidos antes de cada ano previsto.
Variáveis climáticas contemporâneas são mantidas separadamente para análise de
cenários, enquanto as versões defasadas podem ser usadas em previsões ordinárias.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TARGET = "consumo_mwh"
CLIMATE_COLUMNS = (
    "temperatura_media_c",
    "umidade_media_pct",
    "precipitacao_mediana_mm",
    "precipitacao_max_24h_mm",
)
ANNUAL_COLUMNS = {
    "populacao_pessoas": "populacao",
    "pib_corrente_milhoes_rs": "pib_corrente",
    "participacao_pib_pct": "participacao_pib",
    "indice_volume_pib_2010_100": "indice_volume_pib",
    "vab_corrente_milhoes_rs": "vab_corrente",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cria atributos temporais para a modelagem do consumo mensal."
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("data/processed/modelagem/base_modelagem_mensal_2015_2025.csv"),
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/processed/modelagem/features_modelagem_2016_2025.csv"),
    )
    parser.add_argument(
        "--audit-file",
        type=Path,
        default=Path("data/processed/auditoria/auditoria_features_modelagem_2016_2025.csv"),
    )
    parser.add_argument("--train-end-year", type=int, default=2023)
    parser.add_argument("--test-start-year", type=int, default=2024)
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: set[str]) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {missing}")


def assert_unique(frame: pd.DataFrame) -> None:
    duplicated = frame.duplicated(["mes", "uf"], keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, ["mes", "uf"]].head().to_dict("records")
        raise ValueError(f"Chaves mes + uf duplicadas: {examples}")


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["mes_seno"] = np.sin(2 * np.pi * result["mes_numero"] / 12)
    result["mes_cosseno"] = np.cos(2 * np.pi * result["mes_numero"] / 12)
    result["tendencia_mensal"] = (
        (result["mes"].dt.year - result["mes"].dt.year.min()) * 12
        + result["mes"].dt.month
        - 1
    )
    return result


def add_group_lags(frame: pd.DataFrame, column: str, lags: tuple[int, ...]) -> None:
    grouped = frame.groupby("uf", sort=False)[column]
    for lag in lags:
        frame[f"{column}_lag_{lag}"] = grouped.shift(lag)


def add_target_history(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.sort_values(["uf", "mes"]).copy()
    add_group_lags(result, TARGET, (1, 3, 6, 12))
    shifted = result.groupby("uf", sort=False)[TARGET].shift(1)
    for window in (3, 6, 12):
        result[f"{TARGET}_media_movel_{window}"] = (
            shifted.groupby(result["uf"], sort=False)
            .rolling(window=window, min_periods=window)
            .mean()
            .reset_index(level=0, drop=True)
        )
    return result


def add_exogenous_lags(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    add_group_lags(result, "carga_mwmed_media_mes", (1, 12))
    for column in CLIMATE_COLUMNS:
        add_group_lags(result, column, (1, 12))
    return result


def prior_available_by_year(
    frame: pd.DataFrame, source_column: str, prefix: str
) -> pd.DataFrame:
    """Obtém o último valor anual não nulo de ano estritamente anterior."""
    annual = (
        frame[["uf", "ano", source_column]]
        .drop_duplicates(["uf", "ano"])
        .sort_values(["uf", "ano"])
    )
    records: list[dict[str, object]] = []
    for uf, group in annual.groupby("uf", sort=False):
        known = group.dropna(subset=[source_column])
        for year in sorted(group["ano"].unique()):
            candidates = known.loc[known["ano"] < year]
            if candidates.empty:
                reference_year = pd.NA
                value = np.nan
                age = pd.NA
            else:
                row = candidates.iloc[-1]
                reference_year = int(row["ano"])
                value = row[source_column]
                age = int(year - reference_year)
            records.append(
                {
                    "uf": uf,
                    "ano": int(year),
                    f"{prefix}_ultimo_disponivel": value,
                    f"{prefix}_ano_referencia": reference_year,
                    f"{prefix}_defasagem_anos": age,
                }
            )
    result = pd.DataFrame(records)
    result[f"{prefix}_ano_referencia"] = result[
        f"{prefix}_ano_referencia"
    ].astype("Int64")
    result[f"{prefix}_defasagem_anos"] = result[
        f"{prefix}_defasagem_anos"
    ].astype("Int64")
    return result


def add_prior_annual_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for source_column, prefix in ANNUAL_COLUMNS.items():
        annual_features = prior_available_by_year(result, source_column, prefix)
        result = result.merge(
            annual_features,
            on=["uf", "ano"],
            how="left",
            validate="many_to_one",
        )
    return result


def add_sample_flags(
    frame: pd.DataFrame, train_end_year: int, test_start_year: int
) -> pd.DataFrame:
    result = frame.copy()
    result["particao"] = np.select(
        [
            result["ano"].le(train_end_year),
            result["ano"].ge(test_start_year),
        ],
        ["treino", "teste"],
        default="intervalo",
    )

    univariate_required = [
        "consumo_mwh_lag_1",
        "consumo_mwh_lag_3",
        "consumo_mwh_lag_6",
        "consumo_mwh_lag_12",
        "consumo_mwh_media_movel_3",
        "consumo_mwh_media_movel_6",
        "consumo_mwh_media_movel_12",
    ]
    energy_climate_required = univariate_required + [
        "carga_mwmed_media_mes_lag_1",
        "temperatura_media_c_lag_1",
        "umidade_media_pct_lag_1",
        "precipitacao_mediana_mm_lag_1",
    ]
    socioeconomic_required = energy_climate_required + [
        "populacao_ultimo_disponivel",
        "pib_corrente_ultimo_disponivel",
    ]

    result["amostra_univariada_completa"] = (
        result[univariate_required].notna().all(axis=1).astype("int8")
    )
    result["amostra_energia_clima_completa"] = (
        result[energy_climate_required].notna().all(axis=1).astype("int8")
    )
    result["amostra_socioeconomica_completa"] = (
        result[socioeconomic_required].notna().all(axis=1).astype("int8")
    )
    return result


def select_output_columns(frame: pd.DataFrame) -> pd.DataFrame:
    identity = [
        "mes",
        "ano",
        "mes_numero",
        "trimestre",
        "uf",
        "regiao",
        "id_subsistema",
        TARGET,
        "particao",
    ]
    time_features = ["mes_seno", "mes_cosseno", "tendencia_mensal"]
    target_features = [
        f"{TARGET}_lag_{lag}" for lag in (1, 3, 6, 12)
    ] + [f"{TARGET}_media_movel_{window}" for window in (3, 6, 12)]
    energy_features = [
        "carga_mwmed_media_mes_lag_1",
        "carga_mwmed_media_mes_lag_12",
    ]
    climate_scenario = list(CLIMATE_COLUMNS)
    climate_lags = [
        f"{column}_lag_{lag}" for column in CLIMATE_COLUMNS for lag in (1, 12)
    ]
    annual_features = [
        f"{prefix}_{suffix}"
        for prefix in ANNUAL_COLUMNS.values()
        for suffix in ("ultimo_disponivel", "ano_referencia", "defasagem_anos")
    ]
    flags = [
        "amostra_univariada_completa",
        "amostra_energia_clima_completa",
        "amostra_socioeconomica_completa",
    ]
    columns = (
        identity
        + time_features
        + target_features
        + energy_features
        + climate_scenario
        + climate_lags
        + annual_features
        + flags
    )
    return frame[columns].copy()


def build_audit(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for partition, group in frame.groupby("particao", sort=False):
        for flag in (
            "amostra_univariada_completa",
            "amostra_energia_clima_completa",
            "amostra_socioeconomica_completa",
        ):
            available = int(group[flag].sum())
            rows.append(
                {
                    "tipo": "amostra",
                    "particao": partition,
                    "item": flag,
                    "registros_totais": len(group),
                    "registros_disponiveis": available,
                    "registros_ausentes": len(group) - available,
                    "cobertura_pct": round(available / len(group) * 100, 3),
                }
            )

    audited_features = [
        "populacao_ultimo_disponivel",
        "populacao_defasagem_anos",
        "pib_corrente_ultimo_disponivel",
        "pib_corrente_defasagem_anos",
        "carga_mwmed_media_mes_lag_1",
        "temperatura_media_c_lag_1",
        "precipitacao_mediana_mm_lag_1",
    ]
    for feature in audited_features:
        available = int(frame[feature].notna().sum())
        rows.append(
            {
                "tipo": "variavel",
                "particao": "TOTAL",
                "item": feature,
                "registros_totais": len(frame),
                "registros_disponiveis": available,
                "registros_ausentes": len(frame) - available,
                "cobertura_pct": round(available / len(frame) * 100, 3),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.train_end_year >= args.test_start_year:
        raise ValueError("O fim do treino deve ser anterior ao início do teste.")

    base = pd.read_csv(args.input_file)
    required = {
        "mes",
        "ano",
        "mes_numero",
        "trimestre",
        "uf",
        "regiao",
        "id_subsistema",
        TARGET,
        "carga_mwmed_media_mes",
        *CLIMATE_COLUMNS,
        *ANNUAL_COLUMNS.keys(),
    }
    require_columns(base, required)
    base["mes"] = pd.to_datetime(base["mes"], errors="coerce")
    if base["mes"].isna().any():
        raise ValueError("A base possui datas mensais inválidas.")
    assert_unique(base)
    base = base.sort_values(["uf", "mes"]).reset_index(drop=True)

    features = add_time_features(base)
    features = add_target_history(features)
    features = add_exogenous_lags(features)
    features = add_prior_annual_features(features)
    features = add_sample_flags(
        features, args.train_end_year, args.test_start_year
    )
    # 2015 fornece o histórico de 12 meses; a modelagem começa em 2016.
    features = features.loc[features["ano"].ge(2016)].copy()
    features = select_output_columns(features)
    features = features.sort_values(["mes", "uf"]).reset_index(drop=True)
    audit = build_audit(features)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.audit_file.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output_file, index=False, date_format="%Y-%m-%d")
    audit.to_csv(args.audit_file, index=False)

    print("Preparação de atributos concluída.")
    print(f"Linhas de 2016 a 2025: {len(features)}")
    print(f"Treino: {int((features['particao'] == 'treino').sum())}")
    print(f"Teste: {int((features['particao'] == 'teste').sum())}")
    for flag in (
        "amostra_univariada_completa",
        "amostra_energia_clima_completa",
        "amostra_socioeconomica_completa",
    ):
        print(f"{flag}: {int(features[flag].sum())}/{len(features)}")
    print(f"Base de atributos: {args.output_file}")
    print(f"Auditoria: {args.audit_file}")


if __name__ == "__main__":
    main()
