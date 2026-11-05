from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_UFS = ("BA", "MG", "PR", "RJ", "RS", "SP")


def normalize_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.upper().strip()
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def detect_encoding_and_header(path: Path) -> tuple[str, int, dict[str, str]]:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "cp1252", "latin1"):
        try:
            with path.open("r", encoding=encoding) as stream:
                lines = [stream.readline() for _ in range(20)]
            header_index = next(
                index
                for index, line in enumerate(lines)
                if ";" in line
                and normalize_text(line.split(";", 1)[0])
                in {"DATA", "DATA YYYY MM DD"}
            )
            metadata: dict[str, str] = {}
            for line in lines[:header_index]:
                if ";" not in line:
                    continue
                key, value = line.rstrip("\r\n").split(";", 1)
                metadata[normalize_text(key)] = value.strip().strip(";")
            return encoding, header_index, metadata
        except (UnicodeDecodeError, StopIteration) as error:
            last_error = error
    raise ValueError(f"Cabecalho nao reconhecido: {last_error}")


def find_column(columns: list[str], required_terms: tuple[str, ...]) -> str:
    for original in columns:
        normalized = normalize_text(original)
        if all(term in normalized for term in required_terms):
            return original
    raise KeyError(f"Coluna nao encontrada para termos: {required_terms}")


def numeric_series(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype("string")
        .str.strip()
        .str.replace(",", ".", regex=False)
        .replace(
            {
                "": pd.NA,
                "-9999": pd.NA,
                "-9999.0": pd.NA,
                "-9999.00": pd.NA,
                "NULL": pd.NA,
                "null": pd.NA,
            }
        )
    )
    return pd.to_numeric(cleaned, errors="coerce")


def metadata_number(metadata: dict[str, str], key: str) -> float:
    value = metadata.get(key, "").replace(",", ".")
    return float(value) if value else np.nan


def uf_from_path(path: Path, ufs: tuple[str, ...]) -> str | None:
    match = re.search(r"_(%s)_" % "|".join(ufs), path.name.upper())
    return match.group(1) if match else None


def process_station_file(
    path: Path,
    ufs: tuple[str, ...],
    start_year: int,
    end_year: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    encoding, header_index, metadata = detect_encoding_and_header(path)
    uf = normalize_text(metadata.get("UF", "")) or uf_from_path(path, ufs)
    station = metadata.get("ESTACAO", path.stem)
    station_code = metadata.get("CODIGO WMO", "")

    audit: dict[str, object] = {
        "arquivo": str(path),
        "uf": uf,
        "codigo_estacao": station_code,
        "estacao": station,
        "encoding": encoding,
        "status": "ok",
        "erro": "",
        "linhas_lidas": 0,
        "linhas_validas": 0,
        "data_inicial": "",
        "data_final": "",
    }

    frame = pd.read_csv(
        path,
        sep=";",
        skiprows=header_index,
        dtype="string",
        encoding=encoding,
        low_memory=False,
        on_bad_lines="skip",
    )
    frame = frame.loc[:, ~frame.columns.astype(str).str.startswith("Unnamed")]
    audit["linhas_lidas"] = len(frame)

    columns = list(frame.columns)
    date_col = find_column(columns, ("DATA",))
    hour_col = find_column(columns, ("HORA",))
    precipitation_col = find_column(columns, ("PRECIPITACAO", "TOTAL", "HORARIO"))
    temperature_col = find_column(columns, ("TEMPERATURA", "AR", "BULBO", "SECO"))
    humidity_col = find_column(columns, ("UMIDADE", "RELATIVA", "AR", "HORARIA"))

    date_text = frame[date_col].astype("string").str.strip()
    dates = pd.to_datetime(date_text, errors="coerce", yearfirst=True)
    valid_period = dates.dt.year.between(start_year, end_year, inclusive="both")
    frame = frame.loc[valid_period].copy()
    dates = dates.loc[valid_period]

    frame["data"] = dates.dt.normalize()
    frame["hora_original"] = frame[hour_col].astype("string").str.strip()
    frame = frame.drop_duplicates(subset=["data", "hora_original"], keep="first")

    frame["precipitacao_mm"] = numeric_series(frame[precipitation_col])
    frame["temperatura_c"] = numeric_series(frame[temperature_col])
    frame["umidade_pct"] = numeric_series(frame[humidity_col])

    frame.loc[frame["precipitacao_mm"] < 0, "precipitacao_mm"] = np.nan
    frame.loc[~frame["temperatura_c"].between(-50, 60), "temperatura_c"] = np.nan
    frame.loc[~frame["umidade_pct"].between(0, 100), "umidade_pct"] = np.nan

    frame["mes"] = frame["data"].dt.to_period("M").dt.to_timestamp()
    audit["linhas_validas"] = len(frame)
    if not frame.empty:
        audit["data_inicial"] = frame["data"].min().date().isoformat()
        audit["data_final"] = frame["data"].max().date().isoformat()

    daily = (
        frame.groupby(["mes", "data"], as_index=False)
        .agg(
            temperatura_max_dia_c=("temperatura_c", "max"),
            temperatura_min_dia_c=("temperatura_c", "min"),
            precipitacao_dia_mm=("precipitacao_mm", lambda values: values.sum(min_count=1)),
        )
    )

    monthly = (
        frame.groupby("mes", as_index=False)
        .agg(
            temperatura_media_c=("temperatura_c", "mean"),
            umidade_media_pct=("umidade_pct", "mean"),
            precipitacao_mensal_mm=(
                "precipitacao_mm",
                lambda values: values.sum(min_count=1),
            ),
            registros_temperatura=("temperatura_c", "count"),
            registros_umidade=("umidade_pct", "count"),
            registros_precipitacao=("precipitacao_mm", "count"),
            registros_observados=("data", "size"),
        )
    )

    daily_monthly = (
        daily.groupby("mes", as_index=False)
        .agg(
            temperatura_max_media_diaria_c=("temperatura_max_dia_c", "mean"),
            temperatura_min_media_diaria_c=("temperatura_min_dia_c", "mean"),
            precipitacao_max_24h_mm=("precipitacao_dia_mm", "max"),
            dias_chuva_intensa=(
                "precipitacao_dia_mm",
                lambda values: int((values >= 50.0).sum()),
            ),
        )
    )

    monthly = monthly.merge(daily_monthly, on="mes", how="left")
    monthly["horas_esperadas"] = monthly["mes"].dt.days_in_month * 24
    monthly["cobertura_temperatura"] = (
        monthly["registros_temperatura"] / monthly["horas_esperadas"]
    ).clip(upper=1)
    monthly["cobertura_umidade"] = (
        monthly["registros_umidade"] / monthly["horas_esperadas"]
    ).clip(upper=1)
    monthly["cobertura_precipitacao"] = (
        monthly["registros_precipitacao"] / monthly["horas_esperadas"]
    ).clip(upper=1)

    monthly.insert(1, "uf", uf)
    monthly.insert(2, "codigo_estacao", station_code)
    monthly.insert(3, "estacao", station)
    monthly.insert(4, "latitude", metadata_number(metadata, "LATITUDE"))
    monthly.insert(5, "longitude", metadata_number(metadata, "LONGITUDE"))
    monthly.insert(6, "altitude_m", metadata_number(metadata, "ALTITUDE"))
    monthly.insert(7, "arquivo", str(path))
    return monthly, audit


def eligible_median(group: pd.DataFrame, value: str, coverage: str, threshold: float) -> float:
    values = group.loc[group[coverage] >= threshold, value].dropna()
    return values.median() if not values.empty else np.nan


def eligible_mean(group: pd.DataFrame, value: str, coverage: str, threshold: float) -> float:
    values = group.loc[group[coverage] >= threshold, value].dropna()
    return values.mean() if not values.empty else np.nan


def eligible_max(group: pd.DataFrame, value: str, coverage: str, threshold: float) -> float:
    values = group.loc[group[coverage] >= threshold, value].dropna()
    return values.max() if not values.empty else np.nan


def eligible_count(group: pd.DataFrame, value: str, coverage: str, threshold: float) -> int:
    return int(group.loc[group[coverage] >= threshold, value].notna().sum())


def aggregate_states(
    station_month: pd.DataFrame,
    ufs: tuple[str, ...],
    start_year: int,
    end_year: int,
    threshold: float,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for (uf, month), group in station_month.groupby(["uf", "mes"], sort=True):
        records.append(
            {
                "mes": month,
                "uf": uf,
                "temperatura_media_c": eligible_median(
                    group, "temperatura_media_c", "cobertura_temperatura", threshold
                ),
                "temperatura_max_media_diaria_c": eligible_median(
                    group,
                    "temperatura_max_media_diaria_c",
                    "cobertura_temperatura",
                    threshold,
                ),
                "temperatura_min_media_diaria_c": eligible_median(
                    group,
                    "temperatura_min_media_diaria_c",
                    "cobertura_temperatura",
                    threshold,
                ),
                "umidade_media_pct": eligible_median(
                    group, "umidade_media_pct", "cobertura_umidade", threshold
                ),
                "precipitacao_mediana_mm": eligible_median(
                    group,
                    "precipitacao_mensal_mm",
                    "cobertura_precipitacao",
                    threshold,
                ),
                "precipitacao_media_mm": eligible_mean(
                    group,
                    "precipitacao_mensal_mm",
                    "cobertura_precipitacao",
                    threshold,
                ),
                "precipitacao_max_24h_mm": eligible_max(
                    group,
                    "precipitacao_max_24h_mm",
                    "cobertura_precipitacao",
                    threshold,
                ),
                "dias_chuva_intensa_max_estacao": eligible_max(
                    group,
                    "dias_chuva_intensa",
                    "cobertura_precipitacao",
                    threshold,
                ),
                "estacoes_temperatura_validas": eligible_count(
                    group, "temperatura_media_c", "cobertura_temperatura", threshold
                ),
                "estacoes_umidade_validas": eligible_count(
                    group, "umidade_media_pct", "cobertura_umidade", threshold
                ),
                "estacoes_precipitacao_validas": eligible_count(
                    group,
                    "precipitacao_mensal_mm",
                    "cobertura_precipitacao",
                    threshold,
                ),
            }
        )

    state_month = pd.DataFrame.from_records(records)
    calendar = pd.date_range(
        f"{start_year}-01-01", f"{end_year}-12-01", freq="MS"
    )
    expected = pd.MultiIndex.from_product(
        [calendar, ufs], names=["mes", "uf"]
    ).to_frame(index=False)
    state_month = expected.merge(state_month, on=["mes", "uf"], how="left")

    count_columns = [column for column in state_month.columns if column.startswith("estacoes_")]
    state_month[count_columns] = state_month[count_columns].fillna(0).astype(int)
    return state_month.sort_values(["mes", "uf"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Processa dados horarios das estacoes automaticas do INMET."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--coverage-threshold", type=float, default=0.80)
    parser.add_argument("--ufs", nargs="+", default=list(DEFAULT_UFS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ufs = tuple(uf.upper() for uf in args.ufs)
    if not 0 < args.coverage_threshold <= 1:
        raise ValueError("--coverage-threshold deve estar entre 0 e 1.")

    all_files = sorted(
        path
        for path in args.input_dir.rglob("*")
        if path.is_file() and path.suffix.upper() == ".CSV"
    )
    selected_files = [path for path in all_files if uf_from_path(path, ufs)]
    if not selected_files:
        raise FileNotFoundError("Nenhum CSV dos estados selecionados foi encontrado.")

    print(f"CSVs encontrados: {len(all_files)}")
    print(f"CSVs selecionados ({', '.join(ufs)}): {len(selected_files)}")

    station_frames: list[pd.DataFrame] = []
    file_audits: list[dict[str, object]] = []
    for index, path in enumerate(selected_files, start=1):
        try:
            monthly, audit = process_station_file(
                path, ufs, args.start_year, args.end_year
            )
            station_frames.append(monthly)
            file_audits.append(audit)
        except Exception as error:  # registra o arquivo problemático e segue o lote
            file_audits.append(
                {
                    "arquivo": str(path),
                    "uf": uf_from_path(path, ufs) or "",
                    "codigo_estacao": "",
                    "estacao": "",
                    "encoding": "",
                    "status": "erro",
                    "erro": f"{type(error).__name__}: {error}",
                    "linhas_lidas": 0,
                    "linhas_validas": 0,
                    "data_inicial": "",
                    "data_final": "",
                }
            )
        if index % 100 == 0 or index == len(selected_files):
            print(f"Processados: {index}/{len(selected_files)}")

    if not station_frames:
        raise RuntimeError("Nenhum arquivo do INMET foi processado com sucesso.")

    station_month = pd.concat(station_frames, ignore_index=True)
    state_month = aggregate_states(
        station_month,
        ufs,
        args.start_year,
        args.end_year,
        args.coverage_threshold,
    )
    file_audit = pd.DataFrame.from_records(file_audits)

    inmet_dir = args.output_dir / "inmet"
    audit_dir = args.output_dir / "auditoria"
    inmet_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    period = f"{args.start_year}_{args.end_year}"
    state_path = inmet_dir / f"clima_mensal_uf_{period}.csv"
    station_path = audit_dir / f"auditoria_inmet_estacao_mes_{period}.csv"
    file_path = audit_dir / f"auditoria_inmet_arquivos_{period}.csv"

    state_month.to_csv(state_path, index=False, date_format="%Y-%m-%d")
    station_month.to_csv(station_path, index=False, date_format="%Y-%m-%d")
    file_audit.to_csv(file_path, index=False)

    errors = int((file_audit["status"] == "erro").sum())
    missing_climate = int(
        state_month[
            ["temperatura_media_c", "umidade_media_pct", "precipitacao_mediana_mm"]
        ]
        .isna()
        .any(axis=1)
        .sum()
    )
    print("\nProcessamento do INMET concluido.")
    print(f"Arquivos com erro: {errors}")
    print(f"Linhas mensais por estacao: {len(station_month)}")
    print(f"Linhas mensais por UF: {len(state_month)}")
    print(f"UF-meses com alguma variavel climatica ausente: {missing_climate}")
    print(f"Base estadual: {state_path}")
    print(f"Auditoria por estacao/mes: {station_path}")
    print(f"Auditoria de arquivos: {file_path}")


if __name__ == "__main__":
    main()
