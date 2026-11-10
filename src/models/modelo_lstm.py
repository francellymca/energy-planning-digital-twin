"""Modelo LSTM univariado para previsão mensal do consumo de energia elétrica.

O modelo utiliza somente o histórico do consumo de cada unidade da Federação.
O período de 2016 a 2023 é destinado ao treinamento, enquanto 2024 e 2025
constituem o conjunto de teste.

A previsão do teste é recursiva: após dezembro de 2023, cada valor previsto
passa a integrar a janela usada na previsão do mês seguinte. Dessa forma, os
valores observados no período de teste não são utilizados como entrada.
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


UF_ORDER = ["BA", "MG", "PR", "RJ", "RS", "SP"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina e avalia uma LSTM univariada por estado."
    )
    parser.add_argument(
        "--features-file",
        type=Path,
        default=Path(
            "data/processed/modelagem/features_modelagem_2016_2025.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/modelagem"),
    )
    parser.add_argument(
        "--window",
        type=int,
        default=12,
        help="Quantidade de meses utilizada como janela de entrada.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Número máximo de épocas na seleção do treinamento.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=35,
        help="Paciência do early stopping.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return parser.parse_args()


def configure_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)

    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


class MinMaxScaler1D:
    """Normalização min-max simples para evitar dependência do scikit-learn."""

    def __init__(self) -> None:
        self.minimum: float | None = None
        self.maximum: float | None = None

    def fit(self, values: np.ndarray) -> "MinMaxScaler1D":
        array = np.asarray(values, dtype=np.float64)
        self.minimum = float(np.min(array))
        self.maximum = float(np.max(array))

        if np.isclose(self.minimum, self.maximum):
            raise ValueError(
                "Não é possível normalizar uma série sem variação."
            )

        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.minimum is None or self.maximum is None:
            raise RuntimeError("O normalizador ainda não foi ajustado.")

        array = np.asarray(values, dtype=np.float64)
        return (array - self.minimum) / (self.maximum - self.minimum)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        if self.minimum is None or self.maximum is None:
            raise RuntimeError("O normalizador ainda não foi ajustado.")

        array = np.asarray(values, dtype=np.float64)
        return array * (self.maximum - self.minimum) + self.minimum


def create_sequences(
    values: np.ndarray,
    dates: pd.Series,
    window: int,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    inputs: list[np.ndarray] = []
    targets: list[float] = []
    target_dates: list[pd.Timestamp] = []

    for index in range(window, len(values)):
        inputs.append(values[index - window:index])
        targets.append(float(values[index]))
        target_dates.append(pd.Timestamp(dates.iloc[index]))

    x_array = np.asarray(inputs, dtype=np.float32).reshape(-1, window, 1)
    y_array = np.asarray(targets, dtype=np.float32).reshape(-1, 1)

    return x_array, y_array, pd.DatetimeIndex(target_dates)


def build_model(window: int) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(window, 1)),
            layers.LSTM(
                units=32,
                activation="tanh",
                recurrent_activation="sigmoid",
            ),
            layers.Dropout(0.10),
            layers.Dense(16, activation="relu"),
            layers.Dense(1),
        ],
        name="lstm_univariada",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )

    return model


def recursive_forecast(
    model: keras.Model,
    initial_window: np.ndarray,
    horizon: int,
) -> np.ndarray:
    history = list(
        np.asarray(initial_window, dtype=np.float32).reshape(-1)
    )
    predictions: list[float] = []

    for _ in range(horizon):
        model_input = np.asarray(
            history[-len(initial_window):],
            dtype=np.float32,
        ).reshape(1, len(initial_window), 1)

        prediction = float(
            model(model_input, training=False).numpy().reshape(-1)[0]
        )

        predictions.append(prediction)
        history.append(prediction)

    return np.asarray(predictions, dtype=np.float64)

class RecursiveValidationCallback(keras.callbacks.Callback):
    """Seleciona a época com uma previsão recursiva do período de validação."""

    def __init__(
        self,
        initial_window: np.ndarray,
        observed_scaled: np.ndarray,
        patience: int,
        min_delta: float = 1e-5,
    ) -> None:
        super().__init__()

        self.initial_window = np.asarray(
            initial_window,
            dtype=np.float32,
        )
        self.observed_scaled = np.asarray(
            observed_scaled,
            dtype=np.float64,
        ).reshape(-1)

        self.patience = patience
        self.min_delta = min_delta

        self.best_rmse = np.inf
        self.best_epoch = 1
        self.best_weights = None
        self.wait = 0
        self.rmse_history: list[float] = []

    def on_epoch_end(
        self,
        epoch: int,
        logs: dict | None = None,
    ) -> None:
        predictions = recursive_forecast(
            model=self.model,
            initial_window=self.initial_window,
            horizon=len(self.observed_scaled),
        )

        rmse = float(
            np.sqrt(
                np.mean(
                    np.square(
                        self.observed_scaled - predictions
                    )
                )
            )
        )

        self.rmse_history.append(rmse)

        if logs is not None:
            logs["rmse_validacao_recursiva"] = rmse

        if rmse < self.best_rmse - self.min_delta:
            self.best_rmse = rmse
            self.best_epoch = epoch + 1
            self.best_weights = self.model.get_weights()
            self.wait = 0
        else:
            self.wait += 1

            if self.wait >= self.patience:
                self.model.stop_training = True

    def on_train_end(self, logs: dict | None = None) -> None:
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)


def calculate_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    error = observed - predicted
    absolute_error = np.abs(error)

    mae = float(np.mean(absolute_error))
    rmse = float(np.sqrt(np.mean(np.square(error))))

    nonzero = observed != 0
    mape = float(
        np.mean(absolute_error[nonzero] / np.abs(observed[nonzero])) * 100
    )

    return {
        "mae_mwh": mae,
        "rmse_mwh": rmse,
        "mape_pct": mape,
    }


def validate_monthly_series(
    state: pd.DataFrame,
    uf: str,
) -> None:
    if state["mes"].duplicated().any():
        duplicated = state.loc[state["mes"].duplicated(), "mes"].tolist()
        raise ValueError(f"{uf}: meses duplicados: {duplicated}")

    expected = pd.date_range(
        state["mes"].min(),
        state["mes"].max(),
        freq="MS",
    )

    observed = pd.DatetimeIndex(state["mes"])

    if not observed.equals(expected):
        missing = expected.difference(observed)
        raise ValueError(
            f"{uf}: série mensal incompleta. Meses ausentes: "
            f"{missing.strftime('%Y-%m').tolist()}"
        )

    if state["consumo_mwh"].isna().any():
        raise ValueError(f"{uf}: consumo ausente na série.")


def train_state_model(
    state: pd.DataFrame,
    uf: str,
    window: int,
    max_epochs: int,
    patience: int,
    batch_size: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    train = state.loc[state["ano"].between(2016, 2023)].copy()
    test = state.loc[state["ano"].between(2024, 2025)].copy()

    if len(train) != 96:
        raise ValueError(
            f"{uf}: eram esperados 96 meses de treinamento, "
            f"mas foram encontrados {len(train)}."
        )

    if len(test) != 24:
        raise ValueError(
            f"{uf}: eram esperados 24 meses de teste, "
            f"mas foram encontrados {len(test)}."
        )

    scaler = MinMaxScaler1D().fit(
        train["consumo_mwh"].to_numpy(dtype=np.float64)
    )

    scaled_train = scaler.transform(
        train["consumo_mwh"].to_numpy(dtype=np.float64)
    )

    x_all, y_all, target_dates = create_sequences(
        scaled_train,
        train["mes"],
        window,
    )

    fit_mask = target_dates.year <= 2022

    x_fit = x_all[fit_mask]
    y_fit = y_all[fit_mask]

    fit_values = scaled_train[
        train["ano"].to_numpy() <= 2022
    ]
    validation_observed_scaled = scaled_train[
        train["ano"].to_numpy() == 2023
    ]

    if len(x_fit) == 0:
        raise ValueError(
            f"{uf}: conjunto de treinamento vazio."
        )

    if len(validation_observed_scaled) != 12:
        raise ValueError(
            f"{uf}: eram esperados 12 meses de validação em 2023, "
            f"mas foram encontrados {len(validation_observed_scaled)}."
        )

    initial_validation_window = fit_values[-window:]

    # Primeira etapa: seleção da quantidade de épocas mediante uma
    # previsão recursiva dos 12 meses de 2023.
    keras.backend.clear_session()
    configure_seed(seed)

    selection_model = build_model(window)

    recursive_validation = RecursiveValidationCallback(
        initial_window=initial_validation_window,
        observed_scaled=validation_observed_scaled,
        patience=patience,
    )

    history = selection_model.fit(
        x_fit,
        y_fit,
        epochs=max_epochs,
        batch_size=batch_size,
        shuffle=False,
        callbacks=[recursive_validation],
        verbose=0,
    )

    best_epoch = recursive_validation.best_epoch

    validation_prediction_scaled = recursive_forecast(
        model=selection_model,
        initial_window=initial_validation_window,
        horizon=len(validation_observed_scaled),
    )

    validation_prediction = scaler.inverse_transform(
        validation_prediction_scaled
    )
    validation_observed = scaler.inverse_transform(
        validation_observed_scaled
    )

    validation_rmse = float(
        np.sqrt(
            np.mean(
                np.square(
                    validation_observed - validation_prediction
                )
            )
        )
    )

    # Primeira etapa: seleção da quantidade de épocas com validação em 2023.
    keras.backend.clear_session()
    configure_seed(seed)

    selection_model = build_model(window)

    # Segunda etapa: novo treinamento com todo o período de 2016 a 2023,
    # utilizando a quantidade de épocas selecionada anteriormente.
    keras.backend.clear_session()
    configure_seed(seed)

    final_model = build_model(window)

    final_model.fit(
        x_all,
        y_all,
        epochs=best_epoch,
        batch_size=batch_size,
        shuffle=False,
        verbose=0,
    )

    initial_window = scaled_train[-window:]

    forecast_scaled = recursive_forecast(
        model=final_model,
        initial_window=initial_window,
        horizon=len(test),
    )

    forecast = scaler.inverse_transform(forecast_scaled)
    forecast = np.maximum(forecast, 0.0)

    observed = test["consumo_mwh"].to_numpy(dtype=np.float64)
    error = observed - forecast
    absolute_error = np.abs(error)

    with np.errstate(divide="ignore", invalid="ignore"):
        percentage_error = np.where(
            observed != 0,
            absolute_error / np.abs(observed) * 100,
            np.nan,
        )

    predictions = pd.DataFrame(
        {
            "mes": test["mes"].to_numpy(),
            "uf": uf,
            "observado_mwh": observed,
            "previsto_lstm_mwh": forecast,
            "erro_mwh": error,
            "erro_absoluto_mwh": absolute_error,
            "erro_percentual_absoluto_pct": percentage_error,
            "janela_meses": window,
            "seed": seed,
        }
    )

    parameters = {
        "uf": uf,
        "janela_meses": window,
        "unidades_lstm": 32,
        "unidades_densa": 16,
        "dropout": 0.10,
        "batch_size": batch_size,
        "seed": seed,
        "epocas_executadas_selecao": len(history.history["loss"]),
        "melhor_epoca": best_epoch,
        "rmse_validacao_mwh": validation_rmse,
        "parametros_treinaveis": int(final_model.count_params()),
        "estrategia_teste": "previsao_recursiva_24_meses",
        "criterio_selecao": "rmse_validacao_recursiva_12_meses",
    }

    return predictions, parameters


def main() -> None:
    args = parse_args()

    if args.window <= 0:
        raise ValueError("A janela deve ser maior que zero.")

    if not args.features_file.exists():
        raise FileNotFoundError(
            f"Arquivo de atributos não encontrado: {args.features_file}"
        )

    data = pd.read_csv(args.features_file)
    data["mes"] = pd.to_datetime(data["mes"], errors="raise")
    data["ano"] = data["mes"].dt.year

    required_columns = {"mes", "ano", "uf", "consumo_mwh"}
    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        raise ValueError(
            "Colunas obrigatórias ausentes: "
            f"{sorted(missing_columns)}"
        )

    data = (
        data.loc[
            data["uf"].isin(UF_ORDER)
            & data["ano"].between(2016, 2025)
        ]
        .sort_values(["uf", "mes"])
        .reset_index(drop=True)
    )

    all_predictions: list[pd.DataFrame] = []
    all_parameters: list[dict[str, object]] = []

    for uf in UF_ORDER:
        print(f"{uf}: treinando LSTM univariada...")

        state = (
            data.loc[data["uf"].eq(uf)]
            .sort_values("mes")
            .reset_index(drop=True)
        )

        validate_monthly_series(state, uf)

        predictions, parameters = train_state_model(
            state=state,
            uf=uf,
            window=args.window,
            max_epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            seed=args.seed,
        )

        all_predictions.append(predictions)
        all_parameters.append(parameters)

        state_metrics = calculate_metrics(
            predictions["observado_mwh"].to_numpy(),
            predictions["previsto_lstm_mwh"].to_numpy(),
        )

        print(
            f"{uf}: melhor época={parameters['melhor_epoca']}; "
            f"MAPE teste={state_metrics['mape_pct']:.3f}%"
        )

    predictions_table = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    metric_rows: list[dict[str, object]] = []

    for uf in UF_ORDER:
        state = predictions_table.loc[
            predictions_table["uf"].eq(uf)
        ]

        metrics = calculate_metrics(
            state["observado_mwh"].to_numpy(dtype=np.float64),
            state["previsto_lstm_mwh"].to_numpy(dtype=np.float64),
        )

        metric_rows.append(
            {
                "uf": uf,
                "observacoes": len(state),
                **metrics,
            }
        )

    general_metrics = calculate_metrics(
        predictions_table["observado_mwh"].to_numpy(dtype=np.float64),
        predictions_table["previsto_lstm_mwh"].to_numpy(dtype=np.float64),
    )

    metric_rows.append(
        {
            "uf": "GERAL",
            "observacoes": len(predictions_table),
            **general_metrics,
        }
    )

    metrics_table = pd.DataFrame(metric_rows)
    parameters_table = pd.DataFrame(all_parameters)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = (
        args.output_dir / "lstm_previsoes_2024_2025.csv"
    )
    metrics_path = (
        args.output_dir / "metricas_lstm_2024_2025.csv"
    )
    parameters_path = (
        args.output_dir / "parametros_lstm_por_uf.csv"
    )

    predictions_table.to_csv(
        predictions_path,
        index=False,
        encoding="utf-8",
    )
    metrics_table.to_csv(
        metrics_path,
        index=False,
        encoding="utf-8",
    )
    parameters_table.to_csv(
        parameters_path,
        index=False,
        encoding="utf-8",
    )

    print("\nLSTM univariada concluída.")
    print(
        metrics_table.to_string(
            index=False,
            formatters={
                "mae_mwh": "{:.3f}".format,
                "rmse_mwh": "{:.3f}".format,
                "mape_pct": "{:.3f}".format,
            },
        )
    )
    print(f"\nPrevisões: {predictions_path}")
    print(f"Métricas: {metrics_path}")
    print(f"Parâmetros: {parameters_path}")


if __name__ == "__main__":
    main()