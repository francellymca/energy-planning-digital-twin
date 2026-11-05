from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


UFS_ORDEM = ["BA", "MG", "PR", "RJ", "RS", "SP"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera figuras e análise dos erros da linha de base sazonal."
    )
    parser.add_argument(
        "--predictions-file",
        type=Path,
        required=True,
        help="CSV de previsões produzido pela linha de base sazonal.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("docs/figures"),
        help="Diretório de saída das figuras.",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("data/processed/modelagem"),
        help="Diretório de saída das tabelas de análise.",
    )
    return parser.parse_args()


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Arquivo de previsões não encontrado: {path}")

    data = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "mes",
        "uf",
        "consumo_mwh",
        "previsao_consumo_mwh",
        "erro_absoluto_mwh",
        "erro_percentual_absoluto_pct",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError("Colunas ausentes: " + ", ".join(sorted(missing)))

    data = data.copy()
    data["mes"] = pd.to_datetime(data["mes"], errors="coerce")
    numeric_columns = [
        "consumo_mwh",
        "previsao_consumo_mwh",
        "erro_absoluto_mwh",
        "erro_percentual_absoluto_pct",
    ]
    data[numeric_columns] = data[numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    if data[["mes", "uf", *numeric_columns]].isna().any().any():
        raise ValueError("Existem valores nulos ou inválidos no arquivo de previsões.")
    if len(data) != 144:
        raise ValueError(f"Esperadas 144 previsões; encontradas {len(data)}.")
    if set(data["uf"].unique()) != set(UFS_ORDEM):
        raise ValueError("O conjunto de UFs não corresponde às seis UFs esperadas.")

    return data.sort_values(["uf", "mes"]).reset_index(drop=True)


def millions_formatter(value: float, _: int) -> str:
    return f"{value / 1_000_000:.1f}"


def plot_observed_vs_predicted(data: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(15, 13), sharex=True)
    axes = axes.flatten()

    for axis, uf in zip(axes, UFS_ORDEM):
        state = (
            data.loc[data["uf"].eq(uf)]
            .sort_values("mes")
        )

        axis.plot(
            state["mes"],
            state["consumo_mwh"],
            color="#184E77",
            linewidth=2.2,
            marker="o",
            markersize=3.5,
            label="Observado",
        )

        axis.plot(
            state["mes"],
            state["previsao_consumo_mwh"],
            color="#E07A00",
            linewidth=1.8,
            linestyle="--",
            label="Linha de base sazonal",
        )

        if uf == "RS":
            axis.axvspan(
                pd.Timestamp("2024-04-01"),
                pd.Timestamp("2024-05-31"),
                color="#D1495B",
                alpha=0.14,
                label="Evento extremo (RS)",
            )

        axis.set_title(
            uf,
            fontsize=13,
            fontweight="bold",
        )

        axis.grid(alpha=0.25)

        axis.yaxis.set_major_formatter(
            FuncFormatter(millions_formatter)
        )

        axis.xaxis.set_major_locator(
            mdates.MonthLocator(interval=3)
        )

        axis.xaxis.set_major_formatter(
            mdates.DateFormatter("%Y-%m")
        )

        axis.tick_params(
            axis="x",
            rotation=35,
            labelsize=9,
        )

        axis.tick_params(
            axis="y",
            labelsize=10,
        )

    fig.suptitle(
        "Consumo observado e previsto pela linha de base sazonal",
        fontsize=17,
        fontweight="bold",
        y=0.995,
    )

    fig.supxlabel(
        "Mês",
        fontsize=13,
        y=0.050,
    )

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
        ncol=3,
        frameon=False,
        fontsize=13,
        bbox_to_anchor=(0.5, 0.008),
    )

    fig.tight_layout(
        rect=(0.035, 0.09, 1.0, 0.96),
        h_pad=1.2,
        w_pad=1.0,
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_mape(data: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    metrics = (
        data.groupby("uf", as_index=False)["erro_percentual_absoluto_pct"]
        .mean()
        .rename(columns={"erro_percentual_absoluto_pct": "mape_pct"})
    )
    metrics["uf"] = pd.Categorical(metrics["uf"], categories=UFS_ORDEM, ordered=True)
    metrics = metrics.sort_values("uf")
    overall = float(data["erro_percentual_absoluto_pct"].mean())

    fig, axis = plt.subplots(figsize=(9, 5.4))
    bars = axis.bar(
        metrics["uf"].astype(str),
        metrics["mape_pct"],
        color="#E07A00",
        width=0.68,
    )
    axis.axhline(
        overall,
        color="#b91c1c",
        linewidth=1.6,
        linestyle="--",
        label=f"MAPE geral: {overall:.2f}%",
    )
    axis.bar_label(bars, fmt="%.2f%%", padding=4, fontsize=9)
    axis.set_title(
        "Erro percentual da linha de base sazonal por estado",
        fontsize=14,
        fontweight="bold",
    )
    axis.set_xlabel("Unidade da Federação")
    axis.set_ylabel("MAPE (%)")
    axis.set_ylim(0, max(metrics["mape_pct"].max(), overall) * 1.22)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, loc="upper left")
    fig.text(
        0.99,
        0.01,
        "Período de teste: jan./2024 a dez./2025.",
        ha="right",
        fontsize=8,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    metrics = metrics.copy()
    metrics["uf"] = metrics["uf"].astype(str)
    return metrics


def build_largest_errors(data: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "uf",
        "mes",
        "consumo_mwh",
        "previsao_consumo_mwh",
        "erro_absoluto_mwh",
        "erro_percentual_absoluto_pct",
    ]
    return (
        data.sort_values(
            ["uf", "erro_percentual_absoluto_pct"], ascending=[True, False]
        )
        .groupby("uf", as_index=False, group_keys=False)
        .head(3)[columns]
        .sort_values(["uf", "erro_percentual_absoluto_pct"], ascending=[True, False])
        .reset_index(drop=True)
    )


def main() -> None:
    args = parse_args()
    data = load_predictions(args.predictions_file)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.analysis_dir.mkdir(parents=True, exist_ok=True)

    observed_path = args.figures_dir / "baseline_observado_previsto_2024_2025.png"
    mape_path = args.figures_dir / "baseline_mape_por_uf_2024_2025.png"

    plot_observed_vs_predicted(data, observed_path)
    metrics = plot_mape(data, mape_path)
    largest_errors = build_largest_errors(data)

    metrics.to_csv(
        args.analysis_dir / "mape_baseline_por_uf_2024_2025.csv",
        index=False,
        encoding="utf-8-sig",
    )
    largest_errors.to_csv(
        args.analysis_dir / "maiores_erros_baseline_2024_2025.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("Análise visual concluída.")
    print(f"Figura observado × previsto: {observed_path}")
    print(f"Figura MAPE por UF: {mape_path}")
    print("\nMaiores erros percentuais por UF:")
    print(
        largest_errors[["uf", "mes", "erro_percentual_absoluto_pct"]]
        .assign(mes=lambda frame: frame["mes"].dt.strftime("%Y-%m"))
        .round({"erro_percentual_absoluto_pct": 3})
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
