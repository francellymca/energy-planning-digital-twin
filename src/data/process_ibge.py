from __future__ import annotations

import argparse
import csv
import re
import tempfile
import zipfile
from pathlib import Path

import pandas as pd


UF_POR_NOME = {
    "Bahia": "BA",
    "Minas Gerais": "MG",
    "Rio de Janeiro": "RJ",
    "São Paulo": "SP",
    "Paraná": "PR",
    "Rio Grande do Sul": "RS",
}

ANOS_POPULACAO = list(range(2015, 2026))
ANOS_PIB = list(range(2015, 2024))

TABELAS_PIB = {
    "tab01.xls": "pib_corrente_milhoes_rs",
    "tab02.xls": "participacao_pib_pct",
    "tab03.xls": "indice_volume_pib_2010_100",
    "tab04.xls": "vab_corrente_milhoes_rs",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Processa dados anuais de população e PIB estadual do IBGE."
    )
    parser.add_argument(
        "--population-file",
        type=Path,
        required=True,
        help="CSV da Tabela SIDRA 6579 com população por UF.",
    )
    parser.add_argument(
        "--pib-zip",
        type=Path,
        required=True,
        help="ZIP Especiais 2010-2023 das Contas Regionais do IBGE.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Diretório raiz das saídas processadas.",
    )
    return parser.parse_args()


def normalize_year(value: object) -> int | None:
    if pd.isna(value):
        return None
    match = re.fullmatch(r"\s*(\d{4})(?:\.0)?\s*", str(value))
    return int(match.group(1)) if match else None


def load_population(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.reader(file, delimiter=";"))

    header_index = next(
        (
            index
            for index, row in enumerate(rows)
            if len(row) >= 4
            and row[0].strip() == "Nível"
            and row[2].strip() == "Unidade da Federação"
            and any(normalize_year(value) for value in row[3:])
        ),
        None,
    )
    if header_index is None:
        raise ValueError("Cabeçalho anual da população não foi encontrado.")

    header = rows[header_index]
    year_columns = {
        column: year
        for column, value in enumerate(header)
        if (year := normalize_year(value)) in ANOS_POPULACAO
    }

    records: list[dict[str, object]] = []
    for row in rows[header_index + 1 :]:
        if len(row) < 3 or row[0].strip() != "UF":
            continue
        state_name = row[2].strip()
        uf = UF_POR_NOME.get(state_name)
        if uf is None:
            continue
        for column, year in year_columns.items():
            raw_value = row[column].strip() if column < len(row) else ""
            population = pd.to_numeric(raw_value, errors="coerce")
            records.append(
                {
                    "ano": year,
                    "uf": uf,
                    "populacao_pessoas": population,
                    "status_dado": (
                        "observado" if pd.notna(population) else "ausente_na_fonte"
                    ),
                }
            )

    observed = pd.DataFrame(records)
    if observed.empty:
        raise ValueError("Nenhum registro das seis UFs foi encontrado no CSV.")

    if observed.duplicated(["ano", "uf"]).any():
        raise ValueError("Foram encontradas duplicidades de ano e UF na população.")

    grid = pd.MultiIndex.from_product(
        [ANOS_POPULACAO, sorted(UF_POR_NOME.values())], names=["ano", "uf"]
    ).to_frame(index=False)
    result = grid.merge(observed, on=["ano", "uf"], how="left")
    result["status_dado"] = result["status_dado"].fillna("ausente_na_fonte")
    result["fonte"] = "IBGE/SIDRA - Tabela 6579"
    result["populacao_pessoas"] = result["populacao_pessoas"].astype("Int64")
    return result.sort_values(["ano", "uf"]).reset_index(drop=True)


def extract_pib_metric(path: Path, metric: str) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=0, header=None, engine="xlrd")
    years = {
        column: year
        for column, value in enumerate(raw.iloc[3])
        if (year := normalize_year(value)) in ANOS_PIB
    }

    records: list[dict[str, object]] = []
    for row_index in range(4, len(raw)):
        state_name = str(raw.iat[row_index, 0]).strip()
        uf = UF_POR_NOME.get(state_name)
        if uf is None:
            continue
        for column, year in years.items():
            value = pd.to_numeric(raw.iat[row_index, column], errors="coerce")
            records.append({"ano": year, "uf": uf, metric: value})

    result = pd.DataFrame(records)
    expected = len(UF_POR_NOME) * len(ANOS_PIB)
    if len(result) != expected:
        raise ValueError(
            f"{path.name}: esperados {expected} registros, encontrados {len(result)}."
        )
    if result[metric].isna().any():
        raise ValueError(f"{path.name}: existem valores nulos em {metric}.")
    if result.duplicated(["ano", "uf"]).any():
        raise ValueError(f"{path.name}: existem duplicidades de ano e UF.")
    return result


def load_pib(zip_path: Path) -> pd.DataFrame:
    with tempfile.TemporaryDirectory(prefix="ibge_pib_") as temp_dir:
        temp_path = Path(temp_dir)
        with zipfile.ZipFile(zip_path) as archive:
            archive_names = {Path(name).name: name for name in archive.namelist()}
            missing_files = set(TABELAS_PIB) - set(archive_names)
            if missing_files:
                raise ValueError(
                    "Tabelas ausentes no ZIP do PIB: " + ", ".join(sorted(missing_files))
                )
            for filename in TABELAS_PIB:
                archive.extract(archive_names[filename], temp_path)

        metrics: list[pd.DataFrame] = []
        for filename, metric in TABELAS_PIB.items():
            extracted_path = temp_path / archive_names[filename]
            metrics.append(extract_pib_metric(extracted_path, metric))

    result = metrics[0]
    for metric_frame in metrics[1:]:
        result = result.merge(
            metric_frame, on=["ano", "uf"], how="outer", validate="one_to_one"
        )

    metric_columns = list(TABELAS_PIB.values())
    if result[metric_columns].isna().any().any():
        raise ValueError("A integração das tabelas de PIB produziu valores nulos.")

    result["fonte"] = "IBGE - Contas Regionais do Brasil 2023"
    return result.sort_values(["ano", "uf"]).reset_index(drop=True)


def build_audit(population: pd.DataFrame, pib: pd.DataFrame) -> pd.DataFrame:
    population_missing = population.loc[
        population["populacao_pessoas"].isna(), ["ano", "uf"]
    ]
    missing_description = ";".join(
        f"{row.ano}-{row.uf}" for row in population_missing.itertuples(index=False)
    )

    pib_metrics = list(TABELAS_PIB.values())
    return pd.DataFrame(
        [
            {
                "conjunto": "populacao_uf",
                "periodo": "2015-2025",
                "linhas_esperadas": len(ANOS_POPULACAO) * len(UF_POR_NOME),
                "linhas_geradas": len(population),
                "duplicidades_ano_uf": int(
                    population.duplicated(["ano", "uf"]).sum()
                ),
                "valores_ausentes": int(population["populacao_pessoas"].isna().sum()),
                "detalhes_ausencia": missing_description,
                "status": "ATENCAO" if not population_missing.empty else "OK",
            },
            {
                "conjunto": "pib_estadual",
                "periodo": "2015-2023",
                "linhas_esperadas": len(ANOS_PIB) * len(UF_POR_NOME),
                "linhas_geradas": len(pib),
                "duplicidades_ano_uf": int(pib.duplicated(["ano", "uf"]).sum()),
                "valores_ausentes": int(pib[pib_metrics].isna().sum().sum()),
                "detalhes_ausencia": "",
                "status": "OK",
            },
        ]
    )


def save_outputs(
    population: pd.DataFrame, pib: pd.DataFrame, audit: pd.DataFrame, output_dir: Path
) -> None:
    ibge_dir = output_dir / "ibge"
    audit_dir = output_dir / "auditoria"
    ibge_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    population.to_csv(
        ibge_dir / "populacao_uf_2015_2025.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pib.to_csv(
        ibge_dir / "pib_estadual_2015_2023.csv",
        index=False,
        encoding="utf-8-sig",
    )
    audit.to_csv(
        audit_dir / "auditoria_ibge.csv",
        index=False,
        encoding="utf-8-sig",
    )


def main() -> None:
    args = parse_args()
    if not args.population_file.is_file():
        raise FileNotFoundError(f"CSV de população não encontrado: {args.population_file}")
    if not args.pib_zip.is_file():
        raise FileNotFoundError(f"ZIP do PIB não encontrado: {args.pib_zip}")

    population = load_population(args.population_file)
    pib = load_pib(args.pib_zip)
    audit = build_audit(population, pib)
    save_outputs(population, pib, audit, args.output_dir)

    print(
        "Processamento concluído: "
        f"população={len(population)} linhas; "
        f"PIB={len(pib)} linhas; "
        f"ausências_população={population['populacao_pessoas'].isna().sum()}."
    )


if __name__ == "__main__":
    main()