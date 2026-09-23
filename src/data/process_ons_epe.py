#!/usr/bin/env python3
"""Processa e integra as bases mensais de consumo da EPE e carga do ONS.

O consumo da EPE por unidade da federação é a variável principal. A carga do
ONS, disponível por subsistema, é mantida como variável operacional
complementar e não é tratada como equivalente ao consumo estadual.
"""

from __future__ import annotations

import argparse
import calendar
import re
from pathlib import Path

import numpy as np
import pandas as pd


START_YEAR = 2015
END_YEAR = 2025
SELECTED_UFS = ("SP", "MG", "RJ", "PR", "BA", "RS")
ONS_COLUMNS = (
    "id_subsistema",
    "nom_subsistema",
    "din_instante",
    "val_cargaenergiamwmed",
)
SYSTEM_TO_SUBSYSTEM = {
    "NORTE": "N",
    "NORDESTE": "NE",
    "SUL": "S",
    "SUDESTE / CENTRO - OESTE": "SE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ons-dir",
        type=Path,
        default=Path("data/ons/carga_energia"),
        help="Diretório com os CSVs anuais CARGA_ENERGIA_*.csv.",
    )
    parser.add_argument(
        "--epe-file",
        type=Path,
        default=Path("data/epe/consumo_mensal/Dados_abertos_Consumo_Mensal.xlsx"),
        help="Arquivo XLSX de consumo mensal da EPE.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Diretório raiz para os arquivos processados.",
    )
    return parser.parse_args()


def extract_year(path: Path) -> int:
    match = re.search(r"(20\d{2})", path.name)
    if not match:
        raise ValueError(f"Ano não identificado no arquivo: {path}")
    return int(match.group(1))


def audit_ons_year(df: pd.DataFrame, year: int) -> dict[str, object]:
    expected_days = 366 if calendar.isleap(year) else 365
    expected_rows = expected_days * 4
    return {
        "fonte": "ONS",
        "ano": year,
        "registros": len(df),
        "registros_esperados": expected_rows,
        "periodos_observados": df["data"].nunique(),
        "periodos_esperados": expected_days,
        "valores_nulos_originais": int(df["carga_mwmed_original"].isna().sum()),
        "valores_imputados": int(df["valor_imputado"].sum()),
        "chaves_duplicadas": int(df.duplicated(["id_subsistema", "data"]).sum()),
        "status": "OK",
    }


def load_and_process_ons(ons_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    files = sorted(ons_dir.glob("CARGA_ENERGIA_*.csv"))
    selected_files = [p for p in files if START_YEAR <= extract_year(p) <= END_YEAR]
    years = [extract_year(p) for p in selected_files]
    expected_years = list(range(START_YEAR, END_YEAR + 1))
    if years != expected_years:
        raise ValueError(f"Arquivos anuais ONS incompletos. Encontrados: {years}")

    frames: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    for path in selected_files:
        year = extract_year(path)
        df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
        missing_columns = set(ONS_COLUMNS) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Colunas ausentes em {path.name}: {sorted(missing_columns)}")

        df = df.loc[:, ONS_COLUMNS].copy()
        df["data"] = pd.to_datetime(
            df.pop("din_instante"), format="%Y-%m-%d", errors="raise"
        )
        df["carga_mwmed_original"] = pd.to_numeric(
            df.pop("val_cargaenergiamwmed"), errors="coerce"
        )
        df["valor_imputado"] = df["carga_mwmed_original"].isna()
        duplicate_count = int(df.duplicated(["id_subsistema", "data"]).sum())
        if duplicate_count:
            raise ValueError(f"{path.name} contém {duplicate_count} chaves duplicadas")

        expected_dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
        if set(df["data"]) != set(expected_dates):
            raise ValueError(f"{path.name} não contém todos os dias do ano")

        df = df.sort_values(["id_subsistema", "data"])
        df["carga_mwmed"] = df.groupby("id_subsistema")[
            "carga_mwmed_original"
        ].transform(lambda series: series.interpolate(method="linear", limit_area="inside"))
        if df["carga_mwmed"].isna().any():
            raise ValueError(f"Ainda existem nulos após interpolação em {path.name}")

        audits.append(audit_ons_year(df, year))
        frames.append(df)

    daily = pd.concat(frames, ignore_index=True)
    daily["mes"] = daily["data"].dt.to_period("M").dt.to_timestamp()
    daily["energia_estimada_mwh_dia"] = daily["carga_mwmed"] * 24.0

    monthly = (
        daily.groupby(["mes", "id_subsistema", "nom_subsistema"], as_index=False)
        .agg(
            carga_mwmed_media_mes=("carga_mwmed", "mean"),
            energia_estimada_mwh_mes=("energia_estimada_mwh_dia", "sum"),
            dias_no_mes=("data", "nunique"),
            valores_imputados=("valor_imputado", "sum"),
        )
        .sort_values(["mes", "id_subsistema"])
    )
    monthly["valores_imputados"] = monthly["valores_imputados"].astype(int)
    return monthly, pd.DataFrame(audits)


def load_and_process_epe(epe_file: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_excel(epe_file, sheet_name="CONSUMO E NUMCONS SAM UF")
    required = {
        "Data",
        "UF",
        "Regiao",
        "Sistema",
        "Classe",
        "TipoConsumidor",
        "Consumo",
        "Consumidores",
    }
    missing_columns = required - set(df.columns)
    if missing_columns:
        raise ValueError(f"Colunas ausentes na planilha EPE: {sorted(missing_columns)}")

    df["mes"] = pd.to_datetime(
        df["Data"].astype("Int64").astype(str), format="%Y%m%d", errors="raise"
    )
    df = df[
        df["UF"].isin(SELECTED_UFS)
        & df["mes"].dt.year.between(START_YEAR, END_YEAR)
    ].copy()
    df["consumo_mwh"] = pd.to_numeric(df["Consumo"], errors="coerce")
    df["numero_consumidores"] = pd.to_numeric(df["Consumidores"], errors="coerce")

    source_keys = ["mes", "UF", "Sistema", "Classe", "TipoConsumidor"]
    duplicate_count = int(df.duplicated(source_keys).sum())
    if duplicate_count:
        raise ValueError(f"A planilha EPE contém {duplicate_count} chaves duplicadas")
    if df[["consumo_mwh", "numero_consumidores"]].isna().any().any():
        raise ValueError("A planilha EPE contém valores nulos nas medidas selecionadas")

    monthly = (
        df.groupby(["mes", "UF", "Regiao", "Sistema"], as_index=False)
        .agg(
            consumo_mwh=("consumo_mwh", "sum"),
            numero_consumidores=("numero_consumidores", "sum"),
        )
        .rename(columns={"UF": "uf", "Regiao": "regiao", "Sistema": "sistema_epe"})
        .sort_values(["mes", "uf"])
    )
    monthly["id_subsistema"] = monthly["sistema_epe"].map(SYSTEM_TO_SUBSYSTEM)
    if monthly["id_subsistema"].isna().any():
        systems = sorted(monthly.loc[monthly["id_subsistema"].isna(), "sistema_epe"].unique())
        raise ValueError(f"Sistemas EPE sem correspondência com ONS: {systems}")
    monthly["consumo_mwh_por_consumidor"] = np.where(
        monthly["numero_consumidores"] > 0,
        monthly["consumo_mwh"] / monthly["numero_consumidores"],
        np.nan,
    )

    complete_months = pd.period_range(
        f"{START_YEAR}-01", f"{END_YEAR}-12", freq="M"
    ).to_timestamp()
    audits: list[dict[str, object]] = []
    for year in range(START_YEAR, END_YEAR + 1):
        part = monthly[monthly["mes"].dt.year == year]
        expected_rows = 12 * len(SELECTED_UFS)
        audits.append(
            {
                "fonte": "EPE",
                "ano": year,
                "registros": len(part),
                "registros_esperados": expected_rows,
                "periodos_observados": part["mes"].nunique(),
                "periodos_esperados": 12,
                "valores_nulos_originais": 0,
                "valores_imputados": 0,
                "chaves_duplicadas": int(part.duplicated(["mes", "uf"]).sum()),
                "status": "OK" if len(part) == expected_rows else "REVISAR",
            }
        )

    expected_index = pd.MultiIndex.from_product(
        [complete_months, SELECTED_UFS], names=["mes", "uf"]
    )
    actual_index = pd.MultiIndex.from_frame(monthly[["mes", "uf"]])
    missing = expected_index.difference(actual_index)
    if len(missing):
        raise ValueError(f"Faltam {len(missing)} combinações mensais UF na EPE")
    return monthly, pd.DataFrame(audits)


def integrate(epe: pd.DataFrame, ons: pd.DataFrame) -> pd.DataFrame:
    integrated = epe.merge(
        ons,
        on=["mes", "id_subsistema"],
        how="left",
        validate="many_to_one",
    )
    ons_fields = ["carga_mwmed_media_mes", "energia_estimada_mwh_mes"]
    if integrated[ons_fields].isna().any().any():
        raise ValueError("A integração EPE–ONS deixou medidas operacionais sem correspondência")
    integrated["papel_epe"] = "variavel_principal_consumo_estadual"
    integrated["papel_ons"] = "variavel_complementar_carga_subsistema"
    return integrated.sort_values(["mes", "uf"]).reset_index(drop=True)


def save_outputs(
    output_dir: Path,
    epe: pd.DataFrame,
    ons: pd.DataFrame,
    integrated: pd.DataFrame,
    audit: pd.DataFrame,
) -> None:
    paths = {
        "epe": output_dir / "epe" / "consumo_mensal_uf_2015_2025.csv",
        "ons": output_dir / "ons" / "carga_energia_mensal_2015_2025.csv",
        "integrated": output_dir
        / "integrado"
        / "base_energetica_mensal_2015_2025.csv",
        "audit": output_dir / "auditoria" / "auditoria_fontes_2015_2025.csv",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    epe.to_csv(paths["epe"], index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    ons.to_csv(paths["ons"], index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    integrated.to_csv(
        paths["integrated"], index=False, encoding="utf-8-sig", date_format="%Y-%m-%d"
    )
    audit.to_csv(paths["audit"], index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    ons, audit_ons = load_and_process_ons(args.ons_dir)
    epe, audit_epe = load_and_process_epe(args.epe_file)
    integrated = integrate(epe, ons)
    audit = pd.concat([audit_ons, audit_epe], ignore_index=True)
    save_outputs(args.output_dir, epe, ons, integrated, audit)
    print(
        "Processamento concluído: "
        f"EPE={len(epe)} linhas; ONS={len(ons)} linhas; "
        f"integrada={len(integrated)} linhas."
    )


if __name__ == "__main__":
    main()
