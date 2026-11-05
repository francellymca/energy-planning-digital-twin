"""Seleciona, treina e avalia modelos Prophet por unidade da Federacao.

Protocolo temporal:
  - ajuste inicial: 2016-2021;
  - selecao de hiperparametros: previsao de 2022-2023;
  - teste progressivo final: 2024-2025.

No teste final, cada mes e previsto com os dados disponiveis ate o mes
anterior. O valor observado e incorporado ao historico somente depois da
previsao, evitando vazamento de informacoes futuras.
"""

from __future__ import annotations

import argparse
import itertools
import logging
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from prophet import Prophet
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "A biblioteca prophet nao esta instalada. Execute: "
        "python -m pip install prophet"
    ) from exc


TARGET = "consumo_mwh"
SCALE = 1_000_000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seleciona, treina e avalia modelos Prophet estaduais."
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("data/processed/modelagem/features_modelagem_2016_2025.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/modelagem"),
    )
    parser.add_argument("--selection-end-year", type=int, default=2021)
    parser.add_argument("--validation-start-year", type=int, default=2022)
    parser.add_argument("--validation-end-year", type=int, default=2023)
    parser.add_argument("--test-start-year", type=int, default=2024)
    parser.add_argument("--test-end-year", type=int, default=2025)
    return parser.parse_args()


def require_columns(frame: pd.DataFrame) -> None:
    missing = {"mes", "uf", TARGET}.difference(frame.columns)
    if missing:
        raise ValueError(f"Colunas obrigatorias ausentes: {sorted(missing)}")


def metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = observed - predicted
    absolute = np.abs(error)
    valid_percentage = observed != 0
    mape = (
        np.mean(absolute[valid_percentage] / np.abs(observed[valid_percentage])) * 100
        if valid_percentage.any()
        else np.nan
    )
    return {
        "mae_mwh": float(np.mean(absolute)),
        "rmse_mwh": float(np.sqrt(np.mean(error**2))),
        "mape_pct": float(mape),
    }


def candidate_parameters() -> list[dict[str, object]]:
    grid = itertools.product(
        (0.01, 0.05, 0.10, 0.50),
        (1.0, 10.0),
        ("additive", "multiplicative"),
        (3, 5),
    )
    return [
        {
            "changepoint_prior_scale": changepoint,
            "seasonality_prior_scale": seasonality,
            "seasonality_mode": mode,
            "fourier_order": fourier,
        }
        for changepoint, seasonality, mode, fourier in grid
    ]


def fit_prophet(history: pd.Series, parameters: dict[str, object]) -> Prophet:
    training = pd.DataFrame(
        {
            "ds": history.index,
            "y": history.to_numpy(dtype=float) / SCALE,
        }
    )
    model = Prophet(
        growth="linear",
        changepoint_prior_scale=float(parameters["changepoint_prior_scale"]),
        seasonality_prior_scale=float(parameters["seasonality_prior_scale"]),
        seasonality_mode=str(parameters["seasonality_mode"]),
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        n_changepoints=min(15, max(0, len(training) - 2)),
        uncertainty_samples=0,
    )
    model.add_seasonality(
        name="annual",
        period=365.25,
        fourier_order=int(parameters["fourier_order"]),
    )
    model.fit(training)
    return model


def forecast_dates(
    history: pd.Series,
    dates: pd.DatetimeIndex,
    parameters: dict[str, object],
) -> np.ndarray:
    model = fit_prophet(history, parameters)
    future = pd.DataFrame({"ds": pd.DatetimeIndex(dates)})
    forecast = model.predict(future)
    return forecast["yhat"].to_numpy(dtype=float) * SCALE


def progressive_forecast(
    train: pd.Series,
    evaluation: pd.Series,
    parameters: dict[str, object],
) -> np.ndarray:
    history = train.copy()
    predictions: list[float] = []

    for date, observed in evaluation.items():
        prediction = forecast_dates(
            history,
            pd.DatetimeIndex([date]),
            parameters,
        )[0]
        predictions.append(float(prediction))
        history.loc[date] = float(observed)
        history = history.sort_index()

    return np.asarray(predictions, dtype=float)


def select_model(
    series: pd.Series,
    args: argparse.Namespace,
) -> tuple[dict[str, object], dict[str, float], int]:
    selection = series.loc[series.index.year <= args.selection_end_year]
    years = series.index.year
    validation = series.loc[
        (years >= args.validation_start_year)
        & (years <= args.validation_end_year)
    ]
    if len(selection) < 36 or validation.empty:
        raise ValueError("Periodos insuficientes para selecao e validacao.")

    candidates: list[dict[str, object]] = []
    failures = 0
    for parameters in candidate_parameters():
        try:
            predicted = forecast_dates(selection, validation.index, parameters)
            score = metrics(validation.to_numpy(dtype=float), predicted)
            if np.isfinite(score["rmse_mwh"]):
                candidates.append({**parameters, **score})
            else:
                failures += 1
        except (ValueError, RuntimeError, OverflowError):
            failures += 1

    if not candidates:
        raise RuntimeError("Nenhum candidato Prophet concluiu a validacao.")

    best = min(candidates, key=lambda item: (item["rmse_mwh"], item["mae_mwh"]))
    parameters = {
        "changepoint_prior_scale": best["changepoint_prior_scale"],
        "seasonality_prior_scale": best["seasonality_prior_scale"],
        "seasonality_mode": best["seasonality_mode"],
        "fourier_order": best["fourier_order"],
    }
    validation_metrics = {
        "validacao_mae_mwh": float(best["mae_mwh"]),
        "validacao_rmse_mwh": float(best["rmse_mwh"]),
        "validacao_mape_pct": float(best["mape_pct"]),
    }
    return parameters, validation_metrics, failures


def main() -> None:
    args = parse_args()
    if not (
        args.selection_end_year < args.validation_start_year
        <= args.validation_end_year
        < args.test_start_year
        <= args.test_end_year
    ):
        raise ValueError("A ordem cronologica dos periodos e invalida.")

    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    logging.getLogger("prophet").setLevel(logging.WARNING)

    frame = pd.read_csv(args.input_file)
    require_columns(frame)
    frame["mes"] = pd.to_datetime(frame["mes"], errors="coerce")
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce")
    if frame[["mes", TARGET]].isna().any().any():
        raise ValueError("A base possui datas ou valores de consumo ausentes.")
    if frame.duplicated(["mes", "uf"]).any():
        raise ValueError("A base possui chaves mes + uf duplicadas.")

    prediction_rows: list[pd.DataFrame] = []
    parameter_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []

    for uf in sorted(frame["uf"].unique()):
        state = (
            frame.loc[frame["uf"].eq(uf), ["mes", TARGET]]
            .sort_values("mes")
            .set_index("mes")[TARGET]
            .astype(float)
        )
        train = state.loc[state.index.year < args.test_start_year]
        years = state.index.year
        test = state.loc[
            (years >= args.test_start_year) & (years <= args.test_end_year)
        ]
        expected = (args.test_end_year - args.test_start_year + 1) * 12
        if len(test) != expected:
            raise ValueError(f"{uf}: periodo de teste incompleto ({len(test)} meses).")

        print(f"{uf}: selecionando hiperparametros Prophet...")
        parameters, validation_scores, failures = select_model(state, args)
        print(
            f"{uf}: modo={parameters['seasonality_mode']}; "
            f"changepoint={parameters['changepoint_prior_scale']}; "
            f"fourier={parameters['fourier_order']}; validacao RMSE="
            f"{validation_scores['validacao_rmse_mwh']:.3f} MWh"
        )

        predicted = progressive_forecast(train, test, parameters)
        observed = test.to_numpy(dtype=float)
        score = metrics(observed, predicted)
        metric_rows.append({"uf": uf, "observacoes": len(test), **score})
        parameter_rows.append(
            {
                "uf": uf,
                **parameters,
                "ajustes_falhos": failures,
                **validation_scores,
            }
        )
        prediction_rows.append(
            pd.DataFrame(
                {
                    "mes": test.index,
                    "uf": uf,
                    "observado_mwh": observed,
                    "previsto_prophet_mwh": predicted,
                    "erro_mwh": observed - predicted,
                    "erro_absoluto_mwh": np.abs(observed - predicted),
                    "erro_percentual_absoluto_pct": (
                        np.abs(observed - predicted) / np.abs(observed) * 100
                    ),
                }
            )
        )

    predictions = pd.concat(prediction_rows, ignore_index=True).sort_values(
        ["mes", "uf"]
    )
    overall = metrics(
        predictions["observado_mwh"].to_numpy(dtype=float),
        predictions["previsto_prophet_mwh"].to_numpy(dtype=float),
    )
    metric_rows.append({"uf": "GERAL", "observacoes": len(predictions), **overall})
    metrics_frame = pd.DataFrame(metric_rows)
    parameters_frame = pd.DataFrame(parameter_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "prophet_previsoes_2024_2025.csv"
    metrics_path = args.output_dir / "metricas_prophet_2024_2025.csv"
    parameters_path = args.output_dir / "parametros_prophet_por_uf.csv"
    predictions.to_csv(predictions_path, index=False, date_format="%Y-%m-%d")
    metrics_frame.to_csv(metrics_path, index=False)
    parameters_frame.to_csv(parameters_path, index=False)

    print("\nProphet concluido.")
    print(metrics_frame.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"Previsoes: {predictions_path}")
    print(f"Metricas: {metrics_path}")
    print(f"Parametros: {parameters_path}")


if __name__ == "__main__":
    main()
