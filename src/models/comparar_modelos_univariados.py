"""Compara linha de base sazonal, SARIMA, Prophet e LSTM no teste 2024-2025."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


UF_ORDER = ["BA", "MG", "PR", "RJ", "RS", "SP"]
OBSERVED = "observado_mwh"
MODEL_COLUMNS = {
    "Linha de base sazonal": "previsto_baseline_mwh",
    "SARIMA": "previsto_sarima_mwh",
    "Prophet": "previsto_prophet_mwh",
    "LSTM": "previsto_lstm_mwh",
}
COLORS = {
    "Observado": "#184E77",
    "Linha de base sazonal": "#E07A00",
    "SARIMA": "#2A9D8F",
    "Prophet": "#7B2CBF",
    "LSTM": "#C44536",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--prophet-file",
        type=Path,
        default=Path("data/processed/modelagem/prophet_previsoes_2024_2025.csv"),
    )
    
    parser.add_argument(
        "--lstm-file",
        type=Path,
        default=Path("data/processed/modelagem/lstm_previsoes_2024_2025.csv"),
    )
    
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("data/processed/modelagem"),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("docs/figures"),
    )
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: set[str], source: Path) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{source}: colunas ausentes: {sorted(missing)}")


def load_predictions(
    path: Path,
    observed_column: str,
    predicted_column: str,
    output_prediction: str,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    require_columns(frame, {"mes", "uf", observed_column, predicted_column}, path)
    frame["mes"] = pd.to_datetime(frame["mes"], errors="coerce")
    frame[observed_column] = pd.to_numeric(frame[observed_column], errors="coerce")
    frame[predicted_column] = pd.to_numeric(frame[predicted_column], errors="coerce")
    if frame[["mes", "uf", observed_column, predicted_column]].isna().any().any():
        raise ValueError(f"{path}: datas, UFs ou valores ausentes.")
    if frame.duplicated(["mes", "uf"]).any():
        raise ValueError(f"{path}: chaves mes + uf duplicadas.")
    return frame[["mes", "uf", observed_column, predicted_column]].rename(
        columns={observed_column: OBSERVED, predicted_column: output_prediction}
    )


def merge_predictions(args: argparse.Namespace) -> pd.DataFrame:
    baseline = load_predictions(
        args.baseline_file,
        "consumo_mwh",
        "previsao_consumo_mwh",
        MODEL_COLUMNS["Linha de base sazonal"],
    )
    sarima = load_predictions(
        args.sarima_file,
        "observado_mwh",
        "previsto_sarima_mwh",
        MODEL_COLUMNS["SARIMA"],
    )
    prophet = load_predictions(
        args.prophet_file,
        "observado_mwh",
        "previsto_prophet_mwh",
        MODEL_COLUMNS["Prophet"],
    )
    lstm = load_predictions(
        args.lstm_file,
        "observado_mwh",
        "previsto_lstm_mwh",
        MODEL_COLUMNS["LSTM"],
    )

    merged = baseline.merge(
        sarima,
        on=["mes", "uf"],
        how="inner",
        suffixes=("_baseline", "_sarima"),
        validate="one_to_one",
    )
    if not np.allclose(
        merged[f"{OBSERVED}_baseline"],
        merged[f"{OBSERVED}_sarima"],
        rtol=1e-10,
        atol=1e-6,
    ):
        raise ValueError("Baseline e SARIMA possuem valores observados divergentes.")
    merged = merged.rename(columns={f"{OBSERVED}_baseline": OBSERVED}).drop(
        columns=[f"{OBSERVED}_sarima"]
    )

    merged = merged.merge(
        prophet,
        on=["mes", "uf"],
        how="inner",
        suffixes=("", "_prophet"),
        validate="one_to_one",
    )
    if not np.allclose(
        merged[OBSERVED],
        merged[f"{OBSERVED}_prophet"],
        rtol=1e-10,
        atol=1e-6,
    ):
        raise ValueError("Prophet possui valores observados divergentes.")
    merged = merged.drop(columns=[f"{OBSERVED}_prophet"])
    
    merged = merged.merge(
        lstm,
        on=["mes", "uf"],
        how="inner",
        suffixes=("", "_lstm"),
        validate="one_to_one",
    )

    if not np.allclose(
        merged[OBSERVED],
        merged[f"{OBSERVED}_lstm"],
        rtol=1e-10,
        atol=1e-6,
    ):
        raise ValueError("LSTM possui valores observados divergentes.")

    merged = merged.drop(columns=[f"{OBSERVED}_lstm"])

    expected = len(UF_ORDER) * 24
    if len(merged) != expected:
        raise ValueError(f"Esperadas {expected} observacoes; encontradas {len(merged)}.")
    return merged.sort_values(["mes", "uf"]).reset_index(drop=True)


def calculate_metrics(observed: pd.Series, predicted: pd.Series) -> dict[str, float]:
    error = observed.to_numpy(dtype=float) - predicted.to_numpy(dtype=float)
    absolute = np.abs(error)
    actual = observed.to_numpy(dtype=float)
    valid = actual != 0
    return {
        "mae_mwh": float(absolute.mean()),
        "rmse_mwh": float(np.sqrt(np.mean(error**2))),
        "mape_pct": float(np.mean(absolute[valid] / np.abs(actual[valid])) * 100),
    }


def build_metrics(merged: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for uf in [*UF_ORDER, "GERAL"]:
        state = merged if uf == "GERAL" else merged.loc[merged["uf"].eq(uf)]
        for model, column in MODEL_COLUMNS.items():
            rows.append(
                {
                    "uf": uf,
                    "modelo": model,
                    "observacoes": len(state),
                    **calculate_metrics(state[OBSERVED], state[column]),
                }
            )
    return pd.DataFrame(rows)


def plot_forecasts(merged: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(15, 13), sharex=True)
    axes = axes.ravel()
    styles = {
        "Linha de base sazonal": "--",
        "SARIMA": "-.",
        "Prophet": ":",
        "LSTM": (0, (5, 1)),
    }

    for axis, uf in zip(axes, UF_ORDER):
        state = merged.loc[merged["uf"].eq(uf)].sort_values("mes")
        axis.plot(
            state["mes"],
            state[OBSERVED] / 1_000_000,
            color=COLORS["Observado"],
            marker="o",
            markersize=3.5,
            linewidth=2.2,
            label="Observado",
        )
        for model, column in MODEL_COLUMNS.items():
            axis.plot(
                state["mes"],
                state[column] / 1_000_000,
                color=COLORS[model],
                linestyle=styles[model],
                linewidth=1.8,
                label=model,
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
        axis.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    fig.suptitle(
        "Consumo observado e previsões dos modelos univariados",
        fontsize=17,
        fontweight="bold",
        y=0.995,
    )
    fig.supxlabel("Mês", fontsize=13, y=0.060)
    fig.supylabel("Consumo de energia elétrica (milhões de MWh)", fontsize=13)
    handles, labels = axes[0].get_legend_handles_labels()
    rs_handles, rs_labels = axes[4].get_legend_handles_labels()
    for handle, label in zip(rs_handles, rs_labels):
        if label not in labels:
            handles.append(handle)
            labels.append(label)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=6,
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(0.5, 0.012),
    )
    fig.subplots_adjust(bottom=0.13, top=0.91, hspace=0.13, wspace=0.12)
    fig.tight_layout(rect=(0.025, 0.045, 1, 0.98))
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_mape(metrics: pd.DataFrame, output: Path) -> None:
    state = metrics.loc[metrics["uf"].isin(UF_ORDER)].copy()

    pivot = (
        state.pivot(index="uf", columns="modelo", values="mape_pct")
        .reindex(UF_ORDER)
    )

    positions = np.arange(len(UF_ORDER))

    # Quatro barras centralizadas em cada estado
    width = 0.18
    offsets = (
        -1.5 * width,
        -0.5 * width,
         0.5 * width,
         1.5 * width,
    )

    fig, axis = plt.subplots(figsize=(1.5, 7.5))

    bar_handles = []

    for offset, model in zip(offsets, MODEL_COLUMNS):
        bars = axis.bar(
            positions + offset,
            pivot[model],
            width,
            label=model,
            color=COLORS[model],
            zorder=3,
        )

        bar_handles.append(bars)

        for bar in bars:
            value = bar.get_height()

            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.11,
                f"{value:.2f}%",
                ha="center",
                va="bottom",
                fontsize=8.5,
                zorder=6,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.88,
                    "pad": 0.6,
                },
            )

    general = (
        metrics.loc[metrics["uf"].eq("GERAL")]
        .set_index("modelo")
    )

    line_styles = {
        "Linha de base sazonal": "--",
        "SARIMA": ":",
        "Prophet": "-.",
        "LSTM": (0, (5, 1)),
    }

    line_colors = {
        "Linha de base sazonal": "#B45309",
        "SARIMA": "#184E77",
        "Prophet": "#5A189A",
        "LSTM": "#8B1E1E",
    }

    line_handles = []

    for model in MODEL_COLUMNS:
        value = float(general.loc[model, "mape_pct"])

        line = axis.axhline(
            value,
            color=line_colors[model],
            linestyle=line_styles[model],
            linewidth=1.8,
            label=f"MAPE geral {model}: {value:.2f}%",
            zorder=1,
        )

        line_handles.append(line)

    axis.set_xticks(positions, UF_ORDER)
    axis.set_ylabel("MAPE (%)", fontsize=12)
    axis.set_xlabel("Unidade da Federação", fontsize=12)

    fig.suptitle(
        "Comparação do erro percentual dos modelos univariados",
        fontsize=15,
        fontweight="bold",
        y=0.975,
    )

    axis.grid(axis="y", alpha=0.25, zorder=0)
    axis.set_axisbelow(True)

    # Espaço calculado pelo maior MAPE estadual
    maximum_mape = float(pivot.max().max())
    axis.set_ylim(0, maximum_mape * 1.16)

    # Uma legenda para as médias gerais
    legend_lines = fig.legend(
        handles=line_handles,
        labels=[
            handle.get_label()
            for handle in line_handles
        ],
        frameon=False,
        fontsize=9,
        ncol=2,
        loc="upper left",
        bbox_to_anchor=(0.07, 0.925),        
    )

    # Outra legenda para as barras
    fig.legend(
        handles=[bars[0] for bars in bar_handles],
        labels=list(MODEL_COLUMNS.keys()),
        frameon=False,
        fontsize=9,
        ncol=2,
        loc="upper right",
        bbox_to_anchor=(0.94, 0.925),       
    )

    fig.subplots_adjust(
        top=0.78,
        bottom=0.13,
        left=0.08,
        right=0.98,
    )
    
    fig.set_size_inches(14.5, 7.5, forward=True)

    fig.savefig(
        output,
        dpi=300,
        facecolor="white",
    )

    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    merged = merge_predictions(args)
    metrics = build_metrics(merged)

    predictions_path = (
        args.analysis_dir / "comparacao_modelos_univariados_previsoes_2024_2025.csv"
    )
    metrics_path = (
        args.analysis_dir / "comparacao_modelos_univariados_metricas_2024_2025.csv"
    )
    forecast_figure = (
        args.figures_dir / "comparacao_modelos_univariados_2024_2025.png"
    )
    mape_figure = (
        args.figures_dir / "comparacao_mape_modelos_univariados_2024_2025.png"
    )

    merged.to_csv(predictions_path, index=False, date_format="%Y-%m-%d")
    metrics.to_csv(metrics_path, index=False)
    plot_forecasts(merged, forecast_figure)
    plot_mape(metrics, mape_figure)

    print("Comparação dos modelos univariados concluída.\n")
    print(
        metrics.pivot(index="uf", columns="modelo", values="mape_pct")
        .reindex([*UF_ORDER, "GERAL"])
        .to_string(float_format=lambda value: f"{value:.3f}")
    )
    print(f"\nPrevisões: {predictions_path}")
    print(f"Métricas: {metrics_path}")
    print(f"Figura temporal: {forecast_figure}")
    print(f"Figura MAPE: {mape_figure}")


if __name__ == "__main__":
    main()
