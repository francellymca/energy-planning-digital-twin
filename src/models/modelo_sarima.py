"""Treina e avalia modelos SARIMA por unidade da Federação.

Seleção temporal:
  - ajuste inicial e triagem por AIC: 2016-2021;
  - validação progressiva: 2022-2023;
  - teste progressivo final: 2024-2025.

As observações são incorporadas ao estado do modelo somente depois da previsão
de cada mês, evitando vazamento de informações futuras.
"""

from __future__ import annotations

import argparse
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "A biblioteca statsmodels não está instalada. Execute: "
        "python -m pip install statsmodels"
    ) from exc


TARGET = "consumo_mwh"
SEASONAL_PERIOD = 12
SCALE = 1_000_000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seleciona, treina e avalia modelos SARIMA estaduais."
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
    parser.add_argument(
        "--top-aic",
        type=int,
        default=8,
        help="Quantidade de candidatos por UF levados à validação temporal.",
    )
    parser.add_argument("--maxiter", type=int, default=300)
    return parser.parse_args()


def require_columns(frame: pd.DataFrame) -> None:
    missing = {"mes", "uf", TARGET}.difference(frame.columns)
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {sorted(missing)}")


def candidate_orders() -> list[tuple[tuple[int, int, int], tuple[int, int, int, int]]]:
    orders = list(itertools.product((0, 1), (0, 1), (0, 1)))
    seasonal = list(itertools.product((0, 1), (0, 1), (0, 1)))
    return [
        (order, (*seasonal_order, SEASONAL_PERIOD))
        for order in orders
        for seasonal_order in seasonal
        if order != (0, 0, 0) or seasonal_order != (0, 0, 0)
    ]


def fit_model(
    values: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    maxiter: int,
):
    endog = pd.Series(values, dtype="float64").reset_index(drop=True) / SCALE
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            endog,
            order=order,
            seasonal_order=seasonal_order,
            trend=None,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit(disp=False, maxiter=maxiter)
    return result


def converged(result) -> bool:
    details = getattr(result, "mle_retvals", {}) or {}
    return bool(details.get("converged", True))


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


def progressive_forecast(
    train: pd.Series,
    evaluation: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    maxiter: int,
) -> np.ndarray:
    result = fit_model(train, order, seasonal_order, maxiter)
    predictions: list[float] = []
    history = list(train.to_numpy(dtype=float))

    for observed in evaluation.to_numpy(dtype=float):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prediction = float(np.asarray(result.forecast(steps=1))[0] * SCALE)
        predictions.append(prediction)
        history.append(float(observed))

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = result.append([float(observed) / SCALE], refit=False)
        except (ValueError, TypeError):
            result = fit_model(pd.Series(history), order, seasonal_order, maxiter)

    return np.asarray(predictions)


def select_model(
    series: pd.Series,
    args: argparse.Namespace,
) -> tuple[
    tuple[int, int, int],
    tuple[int, int, int, int],
    dict[str, float],
    int,
    int,
]:
    selection = series.loc[series.index.year <= args.selection_end_year]
    validation_years = series.index.year
    validation = series.loc[
        (validation_years >= args.validation_start_year)
        & (validation_years <= args.validation_end_year)
    ]
    if len(selection) < 36 or validation.empty:
        raise ValueError("Períodos insuficientes para seleção e validação.")

    aic_candidates: list[dict[str, object]] = []
    failures = 0
    for order, seasonal_order in candidate_orders():
        try:
            result = fit_model(selection, order, seasonal_order, args.maxiter)
            if converged(result) and np.isfinite(result.aic):
                aic_candidates.append(
                    {
                        "order": order,
                        "seasonal_order": seasonal_order,
                        "aic": float(result.aic),
                    }
                )
            else:
                failures += 1
        except (ValueError, np.linalg.LinAlgError, OverflowError):
            failures += 1

    if not aic_candidates:
        raise RuntimeError("Nenhum candidato SARIMA convergiu.")

    top = sorted(aic_candidates, key=lambda item: item["aic"])[: args.top_aic]
    validated: list[dict[str, object]] = []
    for candidate in top:
        try:
            prediction = progressive_forecast(
                selection,
                validation,
                candidate["order"],
                candidate["seasonal_order"],
                args.maxiter,
            )
            score = metrics(validation.to_numpy(dtype=float), prediction)
            validated.append({**candidate, **score})
        except (ValueError, np.linalg.LinAlgError, OverflowError):
            failures += 1

    if not validated:
        raise RuntimeError("Nenhum candidato sobreviveu à validação temporal.")

    best = min(validated, key=lambda item: (item["rmse_mwh"], item["mae_mwh"]))
    validation_metrics = {
        "validacao_mae_mwh": float(best["mae_mwh"]),
        "validacao_rmse_mwh": float(best["rmse_mwh"]),
        "validacao_mape_pct": float(best["mape_pct"]),
        "aic_selecao": float(best["aic"]),
    }
    return (
        best["order"],
        best["seasonal_order"],
        validation_metrics,
        len(aic_candidates),
        failures,
    )


def tuple_text(values: tuple[int, ...]) -> str:
    return "(" + ",".join(str(value) for value in values) + ")"


def main() -> None:
    args = parse_args()
    if not (
        args.selection_end_year < args.validation_start_year
        <= args.validation_end_year
        < args.test_start_year
        <= args.test_end_year
    ):
        raise ValueError("A ordem cronológica dos períodos é inválida.")
    if args.top_aic < 1:
        raise ValueError("--top-aic deve ser maior ou igual a 1.")

    frame = pd.read_csv(args.input_file)
    require_columns(frame)
    frame["mes"] = pd.to_datetime(frame["mes"], errors="coerce")
    if frame["mes"].isna().any() or frame[TARGET].isna().any():
        raise ValueError("A base possui datas ou valores de consumo ausentes.")
    if frame.duplicated(["mes", "uf"]).any():
        raise ValueError("A base possui chaves mes + uf duplicadas.")

    predictions_rows: list[pd.DataFrame] = []
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
        test_years = state.index.year
        test = state.loc[
            (test_years >= args.test_start_year)
            & (test_years <= args.test_end_year)
        ]
        if len(test) != (args.test_end_year - args.test_start_year + 1) * 12:
            raise ValueError(f"{uf}: período de teste incompleto ({len(test)} meses).")

        print(f"{uf}: selecionando parâmetros SARIMA...")
        order, seasonal_order, validation_scores, candidates, failures = select_model(
            state, args
        )
        print(
            f"{uf}: SARIMA{tuple_text(order)}x"
            f"{tuple_text(seasonal_order)}; validacao RMSE="
            f"{validation_scores['validacao_rmse_mwh']:.3f} MWh"
        )

        predicted = progressive_forecast(
            train, test, order, seasonal_order, args.maxiter
        )
        observed = test.to_numpy(dtype=float)
        state_metrics = metrics(observed, predicted)
        metric_rows.append(
            {"uf": uf, "observacoes": len(test), **state_metrics}
        )
        parameter_rows.append(
            {
                "uf": uf,
                "ordem": tuple_text(order),
                "ordem_sazonal": tuple_text(seasonal_order),
                "candidatos_convergentes": candidates,
                "ajustes_falhos": failures,
                **validation_scores,
            }
        )
        predictions_rows.append(
            pd.DataFrame(
                {
                    "mes": test.index,
                    "uf": uf,
                    "observado_mwh": observed,
                    "previsto_sarima_mwh": predicted,
                    "erro_mwh": observed - predicted,
                    "erro_absoluto_mwh": np.abs(observed - predicted),
                    "erro_percentual_absoluto_pct": (
                        np.abs(observed - predicted) / np.abs(observed) * 100
                    ),
                    "ordem": tuple_text(order),
                    "ordem_sazonal": tuple_text(seasonal_order),
                }
            )
        )

    predictions = pd.concat(predictions_rows, ignore_index=True).sort_values(
        ["mes", "uf"]
    )
    all_metrics = metrics(
        predictions["observado_mwh"].to_numpy(dtype=float),
        predictions["previsto_sarima_mwh"].to_numpy(dtype=float),
    )
    metric_rows.append(
        {"uf": "GERAL", "observacoes": len(predictions), **all_metrics}
    )
    metrics_frame = pd.DataFrame(metric_rows)
    parameters = pd.DataFrame(parameter_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "sarima_previsoes_2024_2025.csv"
    metrics_path = args.output_dir / "metricas_sarima_2024_2025.csv"
    parameters_path = args.output_dir / "parametros_sarima_por_uf.csv"
    predictions.to_csv(predictions_path, index=False, date_format="%Y-%m-%d")
    metrics_frame.to_csv(metrics_path, index=False)
    parameters.to_csv(parameters_path, index=False)

    print("\nSARIMA concluído.")
    print(metrics_frame.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"Previsões: {predictions_path}")
    print(f"Métricas: {metrics_path}")
    print(f"Parâmetros: {parameters_path}")


if __name__ == "__main__":
    main()
