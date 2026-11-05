"""Integra EPE/ONS, IBGE e INMET em uma base mensal para modelagem.

O script preserva valores ausentes, não interpola indicadores anuais e exclui do
recorte analítico as medidas do ONS de 2015, cuja incompletude foi identificada
na auditoria da fonte.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


UFS_PADRAO = ("BA", "MG", "PR", "RJ", "RS", "SP")
COLUNAS_ONS = (
    "carga_mwmed_media_mes",
    "energia_estimada_mwh_mes",
    "valores_imputados",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Integra as bases mensais de EPE/ONS, IBGE e INMET."
    )
    parser.add_argument(
        "--energia-file",
        type=Path,
        default=Path("data/processed/integrado/base_energetica_mensal_2015_2025.csv"),
    )
    parser.add_argument(
        "--populacao-file",
        type=Path,
        default=Path("data/processed/ibge/populacao_uf_2015_2025.csv"),
    )
    parser.add_argument(
        "--pib-file",
        type=Path,
        default=Path("data/processed/ibge/pib_estadual_2015_2023.csv"),
    )
    parser.add_argument(
        "--clima-file",
        type=Path,
        default=Path("data/processed/inmet/clima_mensal_uf_2015_2025.csv"),
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/processed/modelagem/base_modelagem_mensal_2015_2025.csv"),
    )
    parser.add_argument(
        "--audit-file",
        type=Path,
        default=Path("data/processed/auditoria/auditoria_base_modelagem_2015_2025.csv"),
    )
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--ufs", nargs="+", default=list(UFS_PADRAO))
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: set[str], source: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{source}: colunas obrigatórias ausentes: {missing}")


def normalize_uf(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["uf"] = result["uf"].astype("string").str.strip().str.upper()
    return result


def assert_unique(frame: pd.DataFrame, keys: list[str], source: str) -> None:
    duplicated = frame.duplicated(keys, keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, keys].head(10).to_dict("records")
        raise ValueError(f"{source}: chaves duplicadas em {keys}: {examples}")


def load_energy(path: Path, ufs: list[str], start: int, end: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    require_columns(frame, {"mes", "uf", "consumo_mwh"}, str(path))
    frame = normalize_uf(frame)
    frame["mes"] = pd.to_datetime(frame["mes"], errors="coerce")
    if frame["mes"].isna().any():
        raise ValueError(f"{path}: existem datas mensais inválidas.")
    frame["mes"] = frame["mes"].dt.to_period("M").dt.to_timestamp()
    frame = frame.loc[
        frame["uf"].isin(ufs) & frame["mes"].dt.year.between(start, end)
    ].copy()
    assert_unique(frame, ["mes", "uf"], str(path))
    return frame


def load_annual(
    path: Path,
    ufs: list[str],
    start: int,
    end: int,
    source_name: str,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    require_columns(frame, {"ano", "uf"}, str(path))
    frame = normalize_uf(frame)
    frame["ano"] = pd.to_numeric(frame["ano"], errors="coerce").astype("Int64")
    frame = frame.loc[
        frame["uf"].isin(ufs) & frame["ano"].between(start, end)
    ].copy()
    assert_unique(frame, ["ano", "uf"], source_name)
    return frame


def load_climate(path: Path, ufs: list[str], start: int, end: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    require_columns(frame, {"mes", "uf"}, str(path))
    frame = normalize_uf(frame)
    frame["mes"] = pd.to_datetime(frame["mes"], errors="coerce")
    if frame["mes"].isna().any():
        raise ValueError(f"{path}: existem datas mensais inválidas.")
    frame["mes"] = frame["mes"].dt.to_period("M").dt.to_timestamp()
    frame = frame.loc[
        frame["uf"].isin(ufs) & frame["mes"].dt.year.between(start, end)
    ].copy()
    assert_unique(frame, ["mes", "uf"], str(path))
    return frame


def expected_calendar(ufs: list[str], start: int, end: int) -> pd.DataFrame:
    months = pd.date_range(f"{start}-01-01", f"{end}-12-01", freq="MS")
    return pd.MultiIndex.from_product(
        [months, ufs], names=["mes", "uf"]
    ).to_frame(index=False)


def rename_source_columns(
    population: pd.DataFrame, pib: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    population = population.rename(
        columns={"status_dado": "status_populacao", "fonte": "fonte_populacao"}
    )
    pib = pib.rename(columns={"fonte": "fonte_pib"})
    return population, pib


def add_annual_lags(base: pd.DataFrame) -> pd.DataFrame:
    """Cria defasagens anuais sem preencher lacunas ou extrapolar valores."""
    result = base.sort_values(["uf", "mes"]).copy()
    annual_columns = [
        "populacao_pessoas",
        "pib_corrente_milhoes_rs",
        "participacao_pib_pct",
        "indice_volume_pib_2010_100",
        "vab_corrente_milhoes_rs",
    ]
    for column in annual_columns:
        if column not in result.columns:
            continue
        year_values = (
            result[["uf", "ano", column]]
            .drop_duplicates(["uf", "ano"])
            .sort_values(["uf", "ano"])
        )
        year_values[f"{column}_lag1_ano"] = year_values.groupby("uf")[column].shift(1)
        result = result.merge(
            year_values[["uf", "ano", f"{column}_lag1_ano"]],
            on=["uf", "ano"],
            how="left",
            validate="many_to_one",
        )
    return result


def build_audit(base: pd.DataFrame, expected_rows: int) -> pd.DataFrame:
    ignored = {
        "mes",
        "ano",
        "mes_numero",
        "trimestre",
        "uf",
        "regiao",
        "sistema_epe",
        "id_subsistema",
        "nom_subsistema",
        "papel_epe",
        "papel_ons",
        "fonte_populacao",
        "fonte_pib",
        "status_populacao",
    }
    variables = [column for column in base.columns if column not in ignored]
    rows: list[dict[str, object]] = []
    for year, group in base.groupby("ano", sort=True):
        for variable in variables:
            observed = int(group[variable].notna().sum())
            rows.append(
                {
                    "ano": int(year),
                    "variavel": variable,
                    "registros_esperados": int(len(group)),
                    "registros_observados": observed,
                    "registros_ausentes": int(len(group) - observed),
                    "cobertura_pct": round(observed / len(group) * 100, 3),
                }
            )

    for variable in variables:
        observed = int(base[variable].notna().sum())
        rows.append(
            {
                "ano": "TOTAL",
                "variavel": variable,
                "registros_esperados": expected_rows,
                "registros_observados": observed,
                "registros_ausentes": expected_rows - observed,
                "cobertura_pct": round(observed / expected_rows * 100, 3),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.start_year > args.end_year:
        raise ValueError("--start-year não pode ser maior que --end-year.")

    ufs = sorted({uf.strip().upper() for uf in args.ufs})
    calendar = expected_calendar(ufs, args.start_year, args.end_year)
    energy = load_energy(args.energia_file, ufs, args.start_year, args.end_year)
    population = load_annual(
        args.populacao_file,
        ufs,
        args.start_year,
        args.end_year,
        "população IBGE",
    )
    pib = load_annual(
        args.pib_file, ufs, args.start_year, args.end_year, "PIB IBGE"
    )
    climate = load_climate(args.clima_file, ufs, args.start_year, args.end_year)
    population, pib = rename_source_columns(population, pib)

    base = calendar.merge(energy, on=["mes", "uf"], how="left", validate="one_to_one")
    base = base.merge(climate, on=["mes", "uf"], how="left", validate="one_to_one")
    base.insert(1, "ano", base["mes"].dt.year)
    base.insert(2, "mes_numero", base["mes"].dt.month)
    base.insert(3, "trimestre", base["mes"].dt.quarter)
    base = base.merge(population, on=["ano", "uf"], how="left", validate="many_to_one")
    base = base.merge(pib, on=["ano", "uf"], how="left", validate="many_to_one")

    # ONS: 2015 foi mantido nos dados de origem, mas retirado do recorte analítico.
    mask_ons_2015 = base["ano"].eq(2015)
    for column in COLUNAS_ONS:
        if column in base.columns:
            base.loc[mask_ons_2015, column] = np.nan
    base["ons_no_recorte"] = (
        base["ano"].ge(2016)
        & base.get("carga_mwmed_media_mes", pd.Series(False, index=base.index)).notna()
    ).astype("int8")

    base = add_annual_lags(base)
    base = base.sort_values(["mes", "uf"]).reset_index(drop=True)
    assert_unique(base, ["mes", "uf"], "base final")

    expected_rows = len(calendar)
    if len(base) != expected_rows:
        raise ValueError(
            f"Base final com {len(base)} linhas; eram esperadas {expected_rows}."
        )

    audit = build_audit(base, expected_rows)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.audit_file.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(args.output_file, index=False, date_format="%Y-%m-%d")
    audit.to_csv(args.audit_file, index=False)

    climate_columns = [
        "temperatura_media_c",
        "umidade_media_pct",
        "precipitacao_mediana_mm",
    ]
    climate_missing = int(base[climate_columns].isna().any(axis=1).sum())
    print("Integração concluída.")
    print(f"Linhas da base final: {len(base)}")
    print(f"Chaves duplicadas: {int(base.duplicated(['mes', 'uf']).sum())}")
    print(f"Consumo EPE ausente: {int(base['consumo_mwh'].isna().sum())}")
    print(
        "Carga ONS ausente no recorte 2016-2025: "
        f"{int(base.loc[base['ano'].ge(2016), 'carga_mwmed_media_mes'].isna().sum())}"
    )
    print(f"UF-meses com clima principal ausente: {climate_missing}")
    print(
        "UF-meses com população ausente: "
        f"{int(base['populacao_pessoas'].isna().sum())}"
    )
    print(
        "UF-meses com PIB corrente ausente: "
        f"{int(base['pib_corrente_milhoes_rs'].isna().sum())}"
    )
    print(f"Base: {args.output_file}")
    print(f"Auditoria: {args.audit_file}")


if __name__ == "__main__":
    main()
