"""Compara a linha de base sazonal com o SARIMA no teste de 2024-2025."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


UF_ORDER = ("BA", "MG", "PR", "RJ", "RS", "SP")
BASELINE_COLUMN = "previsao_consumo_mwh"
SARIMA_COLUMN = "previsto_sarima_mwh"
OBSERVED_COLUMN = "consumo_mwh"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compara previsões da linha de base e do SARIMA."
    )
    parser.add_argument(
        "--baseline-file",
        type=Path,
        default=Path(
            "data/processed/modelagem/baseline_sazonal_previsoes_2024_2025.csv"
        ),
    )
    parser.add_argument(
        "--sarima-file",
        type=Path,
        default=Path("data/processed/modelagem/sarima_previsoes_2024_2025.csv"),
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("data/processed/modelagem"),
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=Path("docs/figures")
    )
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, required: set[str], source: Path) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source}: colunas obrigatórias ausentes: {missing}")


def load_and_merge(baseline_path: Path, sarima_path: Path) -> pd.DataFrame:
    baseline = pd.read_csv(baseline_path)
    sarima = pd.read_csv(sarima_path)
    require_columns(
        baseline,
        {"mes", "uf", "regiao", OBSERVED_COLUMN, BASELINE_COLUMN},
        baseline_path,
    )
    require_columns(
        sarima,
        {"mes", "uf", "observado_mwh", SARIMA_COLUMN},
        sarima_path,
    )
    for frame, source in ((baseline, baseline_path), (sarima, sarima_path)):
        frame["mes"] = pd.to_datetime(frame["mes"], errors="coerce")
        if frame["mes"].isna().any():
            raise ValueError(f"{source}: existem datas inválidas.")
        if frame.duplicated(["mes", "uf"]).any():
            raise ValueError(f"{source}: existem chaves mes + uf duplicadas.")

    merged = baseline[
        ["mes", "uf", "regiao", OBSERVED_COLUMN, BASELINE_COLUMN]
    ].merge(
        sarima[["mes", "uf", "observado_mwh", SARIMA_COLUMN, "ordem", "ordem_sazonal"]],
        on=["mes", "uf"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(baseline) or len(merged) != len(sarima):
        raise ValueError(
            "As bases não possuem exatamente as mesmas combinações mes + uf."
        )
    difference = np.abs(merged[OBSERVED_COLUMN] - merged["observado_mwh"])
    if not np.allclose(difference, 0, atol=0.01, rtol=0):
        raise ValueError("Os valores observados divergem entre as duas bases.")
    merged = merged.drop(columns="observado_mwh")

    for name, forecast_column in (
        ("baseline", BASELINE_COLUMN),
        ("sarima", SARIMA_COLUMN),
    ):
        error = merged[OBSERVED_COLUMN] - merged[forecast_column]
        merged[f"erro_{name}_mwh"] = error
        merged[f"erro_absoluto_{name}_mwh"] = error.abs()
        merged[f"erro_percentual_absoluto_{name}_pct"] = (
            error.abs() / merged[OBSERVED_COLUMN].abs() * 100
        )
    return merged.sort_values(["mes", "uf"]).reset_index(drop=True)


def calculate_metrics(
    observed: pd.Series, predicted: pd.Series
) -> dict[str, float]:
    error = observed.to_numpy(dtype=float) - predicted.to_numpy(dtype=float)
    absolute = np.abs(error)
    return {
        "mae_mwh": float(absolute.mean()),
        "rmse_mwh": float(np.sqrt(np.mean(error**2))),
        "mape_pct": float(np.mean(absolute / np.abs(observed)) * 100),
    }


def metric_comparison(merged: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups: list[tuple[str, pd.DataFrame]] = [
        (uf, merged.loc[merged["uf"].eq(uf)]) for uf in UF_ORDER
    ]
    groups.append(("GERAL", merged))

    for uf, group in groups:
        baseline = calculate_metrics(group[OBSERVED_COLUMN], group[BASELINE_COLUMN])
        sarima = calculate_metrics(group[OBSERVED_COLUMN], group[SARIMA_COLUMN])
        rows.append(
            {
                "uf": uf,
                "observacoes": len(group),
                "mae_baseline_mwh": baseline["mae_mwh"],
                "mae_sarima_mwh": sarima["mae_mwh"],
                "reducao_mae_pct": (
                    (baseline["mae_mwh"] - sarima["mae_mwh"])
                    / baseline["mae_mwh"]
                    * 100
                ),
                "rmse_baseline_mwh": baseline["rmse_mwh"],
                "rmse_sarima_mwh": sarima["rmse_mwh"],
                "reducao_rmse_pct": (
                    (baseline["rmse_mwh"] - sarima["rmse_mwh"])
                    / baseline["rmse_mwh"]
                    * 100
                ),
                "mape_baseline_pct": baseline["mape_pct"],
                "mape_sarima_pct": sarima["mape_pct"],
                "reducao_mape_pct": (
                    (baseline["mape_pct"] - sarima["mape_pct"])
                    / baseline["mape_pct"]
                    * 100
                ),
                "modelo_melhor_mape": (
                    "SARIMA"
                    if sarima["mape_pct"] < baseline["mape_pct"]
                    else "Linha de base"
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_forecasts(merged: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(15, 13), sharex=True)
    axes = axes.ravel()
    for axis, uf in zip(axes, UF_ORDER):
        state = merged.loc[merged["uf"].eq(uf)].sort_values("mes")
        axis.plot(
            state["mes"],
            state[OBSERVED_COLUMN] / 1_000_000,
            color="#184E77",
            marker="o",
            markersize=3.5,
            linewidth=2.2,
            label="Observado",
        )
        axis.plot(
            state["mes"],
            state[BASELINE_COLUMN] / 1_000_000,
            color="#E07A00",
            linestyle="--",
            linewidth=1.8,
            label="Linha de base sazonal",
        )
        axis.plot(
            state["mes"],
            state[SARIMA_COLUMN] / 1_000_000,
            color="#2A9D8F",
            linestyle="-.",
            linewidth=1.8,
            label="SARIMA",
        )
        if uf == "RS":
            axis.axvspan(
                pd.Timestamp("2024-04-01"),
                pd.Timestamp("2024-05-31"),
                color="#D1495B",
                alpha=0.14,
                label="Evento extremo (RS)",
            )
        axis.set_title(uf, fontsize=13, fontweight="bold")
        axis.grid(alpha=0.25)
        axis.tick_params(axis="x", rotation=35, labelsize=9)
        axis.tick_params(axis="y", labelsize=10)

    fig.suptitle(
        "Consumo observado, linha de base sazonal e SARIMA",
        fontsize=17,
        fontweight="bold",
        y=0.995,
    )
    fig.supxlabel("Mês", fontsize=13, y=0.050)

    fig.supylabel(
        "Consumo de energia elétrica (milhões de MWh)",
        fontsize=13,
        x=0.015,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    rs_handles, rs_labels = axes[4].get_legend_handles_labels()

    for handle, label in zip(rs_handles, rs_labels):
        if label not in labels:
            handles.append(handle)
            labels.append(label)

    fig.legend(
        handles=handles,
        labels=labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=13,
        bbox_to_anchor=(0.5, 0.008),
    )

    fig.tight_layout(
        rect=(0.035, 0.09, 1.0, 0.96),
        h_pad=1.2,
        w_pad=1.0,
    )

    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_mape(comparison: pd.DataFrame, output: Path) -> None:
    state = comparison.loc[comparison["uf"].isin(UF_ORDER)].copy()
    state["uf"] = pd.Categorical(
        state["uf"],
        categories=UF_ORDER,
        ordered=True,
    )
    state = state.sort_values("uf").reset_index(drop=True)

    positions = np.arange(len(state))
    width = 0.36

    mape_geral_baseline = 3.889
    mape_geral_sarima = 2.202

    fig, axis = plt.subplots(figsize=(11.5, 6.5))

    bars_baseline = axis.bar(
        positions - width / 2,
        state["mape_baseline_pct"],
        width,
        label="Linha de base sazonal",
        color="#D98C3F",
    )

    bars_sarima = axis.bar(
        positions + width / 2,
        state["mape_sarima_pct"],
        width,
        label="SARIMA",
        color="#2A9D8F",
    )

    # MAPE geral de cada método
    axis.axhline(
        y=mape_geral_baseline,
        color="#B45309",
        linestyle="--",
        linewidth=1.8,
        label="MAPE geral da linha de base: 3,89%",
    )

    axis.axhline(
        y=mape_geral_sarima,
        color="#184E77",
        linestyle=":",
        linewidth=2.2,
        label="MAPE geral do SARIMA: 2,20%",
    )

    axis.bar_label(
        bars_baseline,
        fmt="%.2f%%",
        padding=3,
        fontsize=10,
    )

    axis.bar_label(
        bars_sarima,
        fmt="%.2f%%",
        padding=3,
        fontsize=10,
    )

    # Redução relativa do MAPE
    for position, reduction in zip(
        positions,
        state["reducao_mape_pct"],
    ):
        height = max(
            state.iloc[position]["mape_baseline_pct"],
            state.iloc[position]["mape_sarima_pct"],
        )

        axis.text(
            position,
            height + 0.5,
            f"Redução: {reduction:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color="#2F4F4F",
        )

    axis.set_xticks(positions, state["uf"].astype(str))
    axis.set_ylabel("MAPE (%)", fontsize=12)
    axis.set_xlabel("Unidade da Federação", fontsize=12)

    axis.set_title(
        "Comparação do erro percentual: linha de base e SARIMA",
        fontsize=15,
        fontweight="bold",
    )

    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)

    axis.legend(
        frameon=False,
        fontsize=9.5,
        loc="upper left",
    )

    axis.set_ylim(
        0,
        max(state["mape_baseline_pct"].max() + 1.3, 6.2),
    )

    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    merged = load_and_merge(args.baseline_file, args.sarima_file)
    comparison = metric_comparison(merged)

    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = (
        args.analysis_dir / "comparacao_baseline_sarima_previsoes_2024_2025.csv"
    )
    metrics_path = (
        args.analysis_dir / "comparacao_baseline_sarima_metricas_2024_2025.csv"
    )
    forecast_figure = (
        args.figures_dir / "comparacao_observado_baseline_sarima_2024_2025.png"
    )
    mape_figure = args.figures_dir / "comparacao_mape_baseline_sarima_2024_2025.png"

    merged.to_csv(predictions_path, index=False, date_format="%Y-%m-%d")
    comparison.to_csv(metrics_path, index=False)
    plot_forecasts(merged, forecast_figure)
    plot_mape(comparison, mape_figure)

    display_columns = [
        "uf",
        "mape_baseline_pct",
        "mape_sarima_pct",
        "reducao_mape_pct",
        "modelo_melhor_mape",
    ]
    print("Comparação concluída.\n")
    print(
        comparison[display_columns].to_string(
            index=False, float_format=lambda value: f"{value:.3f}"
        )
    )
    print(f"\nPrevisões comparadas: {predictions_path}")
    print(f"Métricas comparadas: {metrics_path}")
    print(f"Figura das previsões: {forecast_figure}")
    print(f"Figura de MAPE: {mape_figure}")


if __name__ == "__main__":
    main()
