from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


UFS_ESPERADAS = {"BA", "MG", "PR", "RJ", "RS", "SP"}
INICIO_TESTE = pd.Timestamp("2024-01-01")
FIM_TESTE = pd.Timestamp("2025-12-01")
DEFASAGEM_SAZONAL = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Avalia o baseline sazonal do consumo mensal de energia da EPE."
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="CSV processado de consumo mensal por UF.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/modelagem"),
        help="Diretório de saída das previsões e métricas.",
    )
    return parser.parse_args()


def load_and_validate(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {path}")

    data = pd.read_csv(path, encoding="utf-8-sig")
    required = {"mes", "uf", "consumo_mwh"}
    missing_columns = required - set(data.columns)
    if missing_columns:
        raise ValueError(
            "Colunas obrigatórias ausentes: " + ", ".join(sorted(missing_columns))
        )

    data = data.copy()
    data["mes"] = pd.to_datetime(data["mes"], errors="coerce")
    data["consumo_mwh"] = pd.to_numeric(data["consumo_mwh"], errors="coerce")
    data["uf"] = data["uf"].astype(str).str.strip().str.upper()

    if data[["mes", "uf", "consumo_mwh"]].isna().any().any():
        null_counts = data[["mes", "uf", "consumo_mwh"]].isna().sum()
        raise ValueError(f"Existem valores nulos nas colunas principais:\n{null_counts}")

    duplicate_count = int(data.duplicated(["mes", "uf"]).sum())
    if duplicate_count:
        raise ValueError(f"Existem {duplicate_count} duplicidades de mês e UF.")

    found_ufs = set(data["uf"].unique())
    if found_ufs != UFS_ESPERADAS:
        raise ValueError(
            f"UFs esperadas: {sorted(UFS_ESPERADAS)}; encontradas: {sorted(found_ufs)}."
        )

    for uf, group in data.groupby("uf"):
        observed = pd.DatetimeIndex(group["mes"].sort_values().unique())
        expected = pd.date_range(observed.min(), observed.max(), freq="MS")
        missing_months = expected.difference(observed)
        if not missing_months.empty:
            formatted = ", ".join(date.strftime("%Y-%m") for date in missing_months)
            raise ValueError(f"Meses ausentes para {uf}: {formatted}")

    return data.sort_values(["uf", "mes"]).reset_index(drop=True)


def calculate_metrics(frame: pd.DataFrame, uf: str) -> dict[str, object]:
    actual = frame["consumo_mwh"].to_numpy(dtype=float)
    predicted = frame["previsao_consumo_mwh"].to_numpy(dtype=float)
    error = actual - predicted

    nonzero = actual != 0
    mape = (
        float(np.mean(np.abs(error[nonzero] / actual[nonzero])) * 100)
        if nonzero.any()
        else np.nan
    )

    return {
        "uf": uf,
        "inicio_teste": frame["mes"].min().date().isoformat(),
        "fim_teste": frame["mes"].max().date().isoformat(),
        "observacoes": len(frame),
        "mae_mwh": float(np.mean(np.abs(error))),
        "rmse_mwh": float(np.sqrt(np.mean(np.square(error)))),
        "mape_pct": mape,
    }


def run_baseline(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = data.copy()
    result["previsao_consumo_mwh"] = result.groupby("uf")["consumo_mwh"].shift(
        DEFASAGEM_SAZONAL
    )

    test = result.loc[
        result["mes"].between(INICIO_TESTE, FIM_TESTE),
        ["mes", "uf", "regiao", "consumo_mwh", "previsao_consumo_mwh"],
    ].copy()

    expected_rows = len(UFS_ESPERADAS) * 24
    if len(test) != expected_rows:
        raise ValueError(
            f"Teste deveria conter {expected_rows} linhas, mas contém {len(test)}."
        )
    if test["previsao_consumo_mwh"].isna().any():
        raise ValueError("O baseline produziu previsões nulas no período de teste.")

    test["erro_mwh"] = test["consumo_mwh"] - test["previsao_consumo_mwh"]
    test["erro_absoluto_mwh"] = test["erro_mwh"].abs()
    test["erro_percentual_absoluto_pct"] = np.where(
        test["consumo_mwh"] != 0,
        test["erro_absoluto_mwh"] / test["consumo_mwh"] * 100,
        np.nan,
    )
    test["modelo"] = "sazonal_ingenuo_12_meses"

    metrics = [
        calculate_metrics(group, uf)
        for uf, group in test.groupby("uf", sort=True)
    ]
    metrics.append(calculate_metrics(test, "GERAL"))
    metrics_frame = pd.DataFrame(metrics)
    metrics_frame["modelo"] = "sazonal_ingenuo_12_meses"

    return test.sort_values(["uf", "mes"]), metrics_frame


def save_outputs(
    predictions: pd.DataFrame, metrics: pd.DataFrame, output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(
        output_dir / "baseline_sazonal_previsoes_2024_2025.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metrics.to_csv(
        output_dir / "metricas_baseline_sazonal_2024_2025.csv",
        index=False,
        encoding="utf-8-sig",
    )


def main() -> None:
    args = parse_args()
    data = load_and_validate(args.input_file)
    predictions, metrics = run_baseline(data)
    save_outputs(predictions, metrics, args.output_dir)

    print("Baseline sazonal concluído.")
    print(f"Previsões geradas: {len(predictions)}")
    print("\nMétricas por UF e geral:")
    print(
        metrics[["uf", "observacoes", "mae_mwh", "rmse_mwh", "mape_pct"]]
        .round(3)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
