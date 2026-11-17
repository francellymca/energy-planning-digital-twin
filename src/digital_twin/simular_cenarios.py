"""Gera a projeção de referência e os cenários do gêmeo digital energético.

O SARIMA estadual validado é reajustado com os dados disponíveis até 2025 e
projeta 2026. Os cenários P25 e P75 são centralizados na mediana histórica para
evitar dupla contagem da tendência já representada pelo modelo de referência.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.modelo_sarima import SCALE, fit_model  # noqa: E402


UF_ORDER = ["BA", "MG", "PR", "RJ", "RS", "SP"]
SCENARIOS = {
    "crescimento_reduzido": "p25",
    "tendencia_historica": "mediana",
    "crescimento_elevado": "p75",
}
SCENARIO_LABELS = {
    "crescimento_reduzido": "Menor demanda (P25)",
    "tendencia_historica": "Cenário de referência",
    "crescimento_elevado": "Maior demanda (P75)",
}
COLORS = {
    "referencia_sarima": "#174F78",
    "crescimento_reduzido": "#7F8C8D",
    "tendencia_historica": "#2A9D8F",
    "crescimento_elevado": "#E67E22",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Projeta o consumo entre 2026 e 2030 e simula "
            "cenários estaduais de demanda."
        )
    )
    parser.add_argument(
        "--features-file",
        type=Path,
        default=Path(
            "data/processed/modelagem/features_modelagem_2016_2025.csv"
        ),
    )
    parser.add_argument(
        "--parameters-file",
        type=Path,
        default=Path(
            "data/processed/modelagem/parametros_sarima_por_uf.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/cenarios"),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("docs/figures"),
    )
    parser.add_argument(
        "--forecast-start-year",
        type=int,
        default=2026,
    )
    parser.add_argument(
        "--forecast-end-year",
        type=int,
        default=2030,
    )
    parser.add_argument("--history-start-year", type=int, default=2016)
    parser.add_argument("--history-end-year", type=int, default=2023)
    parser.add_argument("--maxiter", type=int, default=300)
    return parser.parse_args()


def parse_order(value: str, expected_length: int) -> tuple[int, ...]:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, tuple) or len(parsed) != expected_length:
        raise ValueError(f"Ordem SARIMA inválida: {value}")
    return tuple(int(item) for item in parsed)


def load_inputs(
    features_path: Path,
    parameters_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not features_path.exists():
        raise FileNotFoundError(f"Base de atributos não encontrada: {features_path}")
    if not parameters_path.exists():
        raise FileNotFoundError(f"Parâmetros SARIMA não encontrados: {parameters_path}")

    data = pd.read_csv(features_path)
    parameters = pd.read_csv(parameters_path)

    required_data = {"mes", "ano", "uf", "consumo_mwh"}
    required_parameters = {"uf", "ordem", "ordem_sazonal"}
    missing_data = required_data.difference(data.columns)
    missing_parameters = required_parameters.difference(parameters.columns)

    if missing_data:
        raise ValueError(f"Colunas ausentes na base: {sorted(missing_data)}")
    if missing_parameters:
        raise ValueError(
            f"Colunas ausentes nos parâmetros: {sorted(missing_parameters)}"
        )

    data["mes"] = pd.to_datetime(data["mes"], errors="raise")
    data["ano"] = pd.to_numeric(data["ano"], errors="raise").astype(int)
    data["consumo_mwh"] = pd.to_numeric(
        data["consumo_mwh"], errors="raise"
    ).astype(float)
    data = (
        data.loc[data["uf"].isin(UF_ORDER)]
        .sort_values(["uf", "mes"])
        .reset_index(drop=True)
    )

    if data.duplicated(["uf", "mes"]).any():
        raise ValueError("A base contém chaves duplicadas de UF e mês.")
    if set(parameters["uf"]) != set(UF_ORDER):
        raise ValueError("Os parâmetros SARIMA não contemplam exatamente as seis UFs.")

    return data, parameters


def calculate_historical_rates(
    data: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    history = data.loc[data["ano"].between(start_year, end_year)].copy()
    annual = (
        history.groupby(["uf", "ano"], as_index=False)["consumo_mwh"]
        .sum()
        .sort_values(["uf", "ano"])
    )
    annual["variacao_anual_pct"] = (
        annual.groupby("uf")["consumo_mwh"].pct_change() * 100.0
    )

    rates = (
        annual.dropna(subset=["variacao_anual_pct"])
        .groupby("uf")["variacao_anual_pct"]
        .quantile([0.25, 0.50, 0.75])
        .unstack()
        .rename(columns={0.25: "p25", 0.50: "mediana", 0.75: "p75"})
        .reindex(UF_ORDER)
        .reset_index()
    )
    rates["ajuste_p25_pct"] = rates["p25"] - rates["mediana"]
    rates["ajuste_mediana_pct"] = 0.0
    rates["ajuste_p75_pct"] = rates["p75"] - rates["mediana"]
    rates["periodo_historico"] = f"{start_year}-{end_year}"
    return rates


def forecast_reference(
    data: pd.DataFrame,
    parameters: pd.DataFrame,
    forecast_start_year: int,
    forecast_end_year: int,
    maxiter: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    expected_last_month = pd.Timestamp(
        forecast_start_year - 1,
        12,
        1,
    )

    forecast_steps = (
        forecast_end_year
        - forecast_start_year
        + 1
    ) * 12

    future_months = pd.date_range(
        start=f"{forecast_start_year}-01-01",
        periods=forecast_steps,
        freq="MS",
    )

    for uf in UF_ORDER:
        state = (
            data.loc[
                data["uf"].eq(uf)
                & data["mes"].lt(
                    f"{forecast_start_year}-01-01"
                ),
                ["mes", "consumo_mwh"],
            ]
            .sort_values("mes")
            .set_index("mes")["consumo_mwh"]
        )

        if (
            state.empty
            or state.index.max() != expected_last_month
        ):
            raise ValueError(
                f"{uf}: o histórico deve terminar em "
                f"{expected_last_month:%Y-%m}."
            )

        parameter = parameters.loc[
            parameters["uf"].eq(uf)
        ].iloc[0]

        order = parse_order(
            str(parameter["ordem"]),
            3,
        )

        seasonal_order = parse_order(
            str(parameter["ordem_sazonal"]),
            4,
        )

        print(
            f"{uf}: ajustando SARIMA{order}x"
            f"{seasonal_order} e projetando "
            f"{forecast_start_year}-{forecast_end_year}..."
        )

        result = fit_model(
            state,
            order,
            seasonal_order,
            maxiter,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            forecast_result = result.get_forecast(
                steps=forecast_steps
            )

        mean = (
            np.asarray(
                forecast_result.predicted_mean,
                dtype=float,
            )
            * SCALE
        )

        confidence = (
            np.asarray(
                forecast_result.conf_int(alpha=0.05),
                dtype=float,
            )
            * SCALE
        )

        rows.append(
            pd.DataFrame(
                {
                    "mes": future_months,
                    "ano": future_months.year,
                    "uf": uf,
                    "referencia_sarima_mwh": np.maximum(
                        mean,
                        0.0,
                    ),
                    "limite_inferior_95_mwh": np.maximum(
                        confidence[:, 0],
                        0.0,
                    ),
                    "limite_superior_95_mwh": np.maximum(
                        confidence[:, 1],
                        0.0,
                    ),
                    "ordem": str(order),
                    "ordem_sazonal": str(
                        seasonal_order
                    ),
                }
            )
        )

    return pd.concat(
        rows,
        ignore_index=True,
    )


def build_scenarios(
    reference: pd.DataFrame,
    rates: pd.DataFrame,
) -> pd.DataFrame:
    merged = reference.merge(
        rates,
        on="uf",
        how="left",
        validate="many_to_one",
    )

    forecast_start_year = int(
        merged["ano"].min()
    )

    merged["ano_horizonte"] = (
        merged["ano"]
        - forecast_start_year
        + 1
    )

    scenario_rows: list[pd.DataFrame] = []

    adjustment_columns = {
        "p25": "ajuste_p25_pct",
        "mediana": "ajuste_mediana_pct",
        "p75": "ajuste_p75_pct",
    }

    for scenario, percentile_column in SCENARIOS.items():
        current = merged.copy()

        adjustment_column = adjustment_columns[
            percentile_column
        ]

        current["cenario"] = scenario
        current["cenario_rotulo"] = (
            SCENARIO_LABELS[scenario]
        )

        current["percentil_historico_pct"] = (
            current[percentile_column]
        )

        current["ajuste_anual_referencia_pct"] = (
            current[adjustment_column]
        )

        annual_multiplier = (
            1.0
            + current[
                "ajuste_anual_referencia_pct"
            ]
            / 100.0
        )

        cumulative_multiplier = (
            annual_multiplier
            ** current["ano_horizonte"]
        )

        current[
            "ajuste_relativo_referencia_pct"
        ] = (
            cumulative_multiplier - 1.0
        ) * 100.0

        current["consumo_cenario_mwh"] = (
            current["referencia_sarima_mwh"]
            * cumulative_multiplier
        )

        current[
            "diferenca_referencia_mwh"
        ] = (
            current["consumo_cenario_mwh"]
            - current["referencia_sarima_mwh"]
        )

        scenario_rows.append(current)

    columns = [
        "mes",
        "ano",
        "ano_horizonte",
        "uf",
        "cenario",
        "cenario_rotulo",
        "referencia_sarima_mwh",
        "consumo_cenario_mwh",
        "diferenca_referencia_mwh",
        "percentil_historico_pct",
        "ajuste_anual_referencia_pct",
        "ajuste_relativo_referencia_pct",
        "limite_inferior_95_mwh",
        "limite_superior_95_mwh",
        "periodo_historico",
        "ordem",
        "ordem_sazonal",
    ]

    return (
        pd.concat(
            scenario_rows,
            ignore_index=True,
        )[columns]
        .sort_values(
            ["uf", "cenario", "mes"]
        )
        .reset_index(drop=True)
    )


def build_annual_summary(scenarios: pd.DataFrame) -> pd.DataFrame:
    summary = (
        scenarios.groupby(
            ["ano", "uf", "cenario", "cenario_rotulo"], as_index=False
        )
        .agg(
            referencia_sarima_mwh=("referencia_sarima_mwh", "sum"),
            consumo_cenario_mwh=("consumo_cenario_mwh", "sum"),
            diferenca_referencia_mwh=("diferenca_referencia_mwh", "sum"),
            percentil_historico_pct=("percentil_historico_pct", "first"),
            ajuste_anual_referencia_pct=(
                "ajuste_anual_referencia_pct",
                "first",
            ),
            ajuste_relativo_referencia_pct=(
                "ajuste_relativo_referencia_pct",
                "first",
            ),
        )
    )
    summary["diferenca_referencia_pct"] = np.where(
        summary["referencia_sarima_mwh"].ne(0),
        summary["diferenca_referencia_mwh"]
        / summary["referencia_sarima_mwh"]
        * 100.0,
        np.nan,
    )
    return (
        summary.sort_values(
            ["uf", "ano", "cenario"]
        )
        .reset_index(drop=True)
    )


def plot_scenarios(scenarios: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 10.5), sharex=True)
    axes = axes.ravel()

    for axis, uf in zip(axes, UF_ORDER):
        state = scenarios.loc[scenarios["uf"].eq(uf)]
        reference = state.loc[
            state["cenario"].eq("tendencia_historica")
        ].sort_values("mes")

        axis.fill_between(
            reference["mes"],
            reference["limite_inferior_95_mwh"] / SCALE,
            reference["limite_superior_95_mwh"] / SCALE,
            color=COLORS["referencia_sarima"],
            alpha=0.10,
            label="IC 95% SARIMA",
        )
        axis.plot(
            reference["mes"],
            reference["referencia_sarima_mwh"] / SCALE,
            color=COLORS["referencia_sarima"],
            linewidth=2.4,
            marker="o",
            markersize=3.5,
            label="Referência SARIMA",
        )

        for scenario in ("crescimento_reduzido", "crescimento_elevado"):
            current = state.loc[state["cenario"].eq(scenario)].sort_values("mes")
            axis.plot(
                current["mes"],
                current["consumo_cenario_mwh"] / SCALE,
                color=COLORS[scenario],
                linewidth=1.8,
                linestyle="--",
                label=SCENARIO_LABELS[scenario],
            )

        axis.set_title(uf, fontsize=12, fontweight="bold")
        axis.grid(alpha=0.25)
        axis.tick_params(axis="x", rotation=35)

    fig.suptitle(
        "Projeção de referência e cenários de consumo: 2026–2030",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    fig.supylabel("Consumo mensal (milhões de MWh)", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.subplots_adjust(top=0.92, bottom=0.10, hspace=0.32, wspace=0.18)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if (
        args.history_start_year
        >= args.history_end_year
    ):
        raise ValueError(
            "O período histórico dos cenários "
            "é inválido."
        )

    if (
        args.forecast_start_year
        > args.forecast_end_year
    ):
        raise ValueError(
            "O início da projeção não pode ser "
            "posterior ao final."
        )

    data, parameters = load_inputs(
        args.features_file,
        args.parameters_file,
    )

    rates = calculate_historical_rates(
        data,
        args.history_start_year,
        args.history_end_year,
    )

    reference = forecast_reference(
        data=data,
        parameters=parameters,
        forecast_start_year=(
            args.forecast_start_year
        ),
        forecast_end_year=(
            args.forecast_end_year
        ),
        maxiter=args.maxiter,
    )

    scenarios = build_scenarios(
        reference,
        rates,
    )

    summary = build_annual_summary(
        scenarios
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    horizon_suffix = (
        f"{args.forecast_start_year}_"
        f"{args.forecast_end_year}"
    )

    rates_path = (
        args.output_dir
        / "taxas_historicas_cenarios_por_uf.csv"
    )

    reference_path = (
        args.output_dir
        / f"projecao_sarima_{horizon_suffix}.csv"
    )

    scenarios_path = (
        args.output_dir
        / f"cenarios_consumo_{horizon_suffix}.csv"
    )

    summary_path = (
        args.output_dir
        / f"resumo_cenarios_{horizon_suffix}.csv"
    )

    metadata_path = (
        args.output_dir
        / f"metadados_cenarios_{horizon_suffix}.json"
    )

    figure_path = (
        args.figures_dir
        / f"gemeo_digital_cenarios_{horizon_suffix}.png"
    )

    rates.to_csv(
        rates_path,
        index=False,
    )

    reference.to_csv(
        reference_path,
        index=False,
        date_format="%Y-%m-%d",
    )

    scenarios.to_csv(
        scenarios_path,
        index=False,
        date_format="%Y-%m-%d",
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    plot_scenarios(
        scenarios,
        figure_path,
    )

    metadata = {
        "inicio_projecao": (
            args.forecast_start_year
        ),
        "fim_projecao": (
            args.forecast_end_year
        ),
        "meses_projetados": int(
            (
                args.forecast_end_year
                - args.forecast_start_year
                + 1
            )
            * 12
        ),
        "modelo_referencia": (
            "SARIMA estadual reajustado "
            "com observações até 2025"
        ),
        "intervalo_confianca": 0.95,
        "periodo_taxas_historicas": (
            f"{args.history_start_year}-"
            f"{args.history_end_year}"
        ),
        "cenarios": SCENARIO_LABELS,
        "formula": (
            "cenario = referencia_sarima * "
            "(1 + (percentil - mediana) / 100) "
            "** ano_horizonte"
        ),
        "interpretacao": (
            "Cenários contrafactuais de demanda; "
            "não representam probabilidades nem "
            "efeitos climáticos causais."
        ),
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nCenários do gêmeo digital concluídos."
    )

    print(
        "\nTaxas históricas por UF (%):"
    )

    print(
        rates[
            ["uf", "p25", "mediana", "p75"]
        ]
        .round(3)
        .to_string(index=False)
    )

    print(
        "\nResumo anual dos cenários:"
    )

    print(
        summary[
            [
                "ano",
                "uf",
                "cenario",
                "consumo_cenario_mwh",
                "diferenca_referencia_pct",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    print(
        "\nArquivos gerados:"
        f"\nTaxas: {rates_path}"
        f"\nReferência: {reference_path}"
        f"\nCenários mensais: {scenarios_path}"
        f"\nResumo anual: {summary_path}"
        f"\nMetadados: {metadata_path}"
        f"\nFigura: {figure_path}"
    )

if __name__ == "__main__":
    main()