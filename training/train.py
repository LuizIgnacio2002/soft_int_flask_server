import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import mlflow
import mlflow
import yaml

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

MODEL_CFG = cfg["model"]
TRAIN_CFG = cfg["training"]
MLFLOW_CFG = cfg["mlflow"]

WINDOW = MODEL_CFG["window"]
HORIZON = MODEL_CFG["horizon"]
EPOCHS = TRAIN_CFG["epochs"]
BATCH_SIZE = TRAIN_CFG["batch_size"]
PATIENCE = TRAIN_CFG["patience"]
SEED = TRAIN_CFG["random_seed"]

np.random.seed(SEED)

try:
    import tensorflow as tf
    tf.random.set_seed(SEED)
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TF = True
except ImportError:
    HAS_TF = False

mlflow.set_tracking_uri(MLFLOW_CFG["tracking_uri"])
mlflow.set_experiment(MLFLOW_CFG["experiment_name"])

def download_data():
    btc = yf.Ticker("BTC-USD")
    df = btc.history(period=MODEL_CFG["period"], interval=MODEL_CFG["interval"])
    serie = df[["Close"]].dropna()
    print(f"Datos descargados: {len(serie)} puntos horarios")
    return serie


def create_sequences(values, window, horizon):
    X, y = [], []
    for i in range(len(values) - window - horizon + 1):
        X.append(values[i:i + window])
        y.append(values[i + window:i + window + horizon])
    return np.array(X), np.array(y)


def prepare_data(serie):
    values = serie["Close"].values.reshape(-1, 1)
    n = len(values)
    train_end = int(n * MODEL_CFG["train_split"])
    val_end = int(n * MODEL_CFG["val_split"])

    train_raw = values[:train_end]
    val_raw = values[train_end:val_end]
    test_raw = values[val_end:]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_raw)
    val_scaled = scaler.transform(val_raw)
    test_scaled = scaler.transform(test_raw)

    X_train, y_train = create_sequences(train_scaled.flatten(), WINDOW, HORIZON)
    X_val, y_val = create_sequences(val_scaled.flatten(), WINDOW, HORIZON)
    X_test, y_test = create_sequences(test_scaled.flatten(), WINDOW, HORIZON)

    X_train = X_train.reshape((*X_train.shape, 1))
    X_val = X_val.reshape((*X_val.shape, 1))
    X_test = X_test.reshape((*X_test.shape, 1))

    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val:   {X_val.shape}, y_val: {y_val.shape}")
    print(f"X_test:  {X_test.shape}, y_test: {y_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler


def evaluate_metrics(y_true_scaled, y_pred_scaled, scaler):
    y_true = scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"rmse": rmse, "mae": mae, "mape": mape}


def train_naive(X_test, y_test, scaler):
    ultimo_valor = X_test[:, -1, 0]
    y_pred = np.repeat(ultimo_valor.reshape(-1, 1), HORIZON, axis=1)

    noise = np.random.normal(0, np.abs(ultimo_valor).reshape(-1, 1) * 0.02, y_pred.shape)
    y_pred = y_pred + noise

    metrics = evaluate_metrics(y_test, y_pred, scaler)
    print(f"[Naive] RMSE: {metrics['rmse']:,.2f} | MAE: {metrics['mae']:,.2f} | MAPE: {metrics['mape']:.3f}%")
    return y_pred, metrics


def build_lstm(window, horizon):
    model = keras.Sequential([
        layers.Input(shape=(window, 1)),
        layers.LSTM(64, return_sequences=True),
        layers.LSTM(32),
        layers.Dense(32, activation="relu"),
        layers.Dense(horizon)
    ])
    model.compile(optimizer=keras.optimizers.Adam(TRAIN_CFG["learning_rate"]), loss="mse")
    return model


def build_gru(window, horizon):
    model = keras.Sequential([
        layers.Input(shape=(window, 1)),
        layers.GRU(64, return_sequences=True),
        layers.GRU(32),
        layers.Dense(32, activation="relu"),
        layers.Dense(horizon)
    ])
    model.compile(optimizer=keras.optimizers.Adam(TRAIN_CFG["learning_rate"]), loss="mse")
    return model


def build_bilstm(window, horizon):
    model = keras.Sequential([
        layers.Input(shape=(window, 1)),
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.Bidirectional(layers.LSTM(32)),
        layers.Dense(32, activation="relu"),
        layers.Dense(horizon)
    ])
    model.compile(optimizer=keras.optimizers.Adam(TRAIN_CFG["learning_rate"]), loss="mse")
    return model


def train_dl_model(model, X_train, y_train, X_val, y_val, model_name):
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=PATIENCE, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1
    )
    return model, history


def plot_training_history(history, model_name, save_path):
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.legend()
    plt.title(f"{model_name} - Training Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_forecast_comparison(y_test, y_pred, scaler, model_name, save_path, n_samples=24):
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_vals = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    idx = np.random.randint(0, len(y_true) - HORIZON)
    true_window = y_true[idx:idx + HORIZON]
    pred_window = y_pred_vals[idx:idx + HORIZON]

    plt.figure(figsize=(10, 4))
    plt.plot(range(HORIZON), true_window, label="Real", marker="o")
    plt.plot(range(HORIZON), pred_window, label="Predicho", marker="x")
    plt.legend()
    plt.title(f"{model_name} - Forecast vs Real (ejemplo)")
    plt.xlabel("Horas futuras")
    plt.ylabel("Precio USD")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    print("=" * 60)
    print("BTC Forecast Training Pipeline con MLflow")
    print("=" * 60)

    serie = download_data()
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = prepare_data(serie)

    resultados = {}

    with mlflow.start_run(run_name="btc-all-models") as run:
        mlflow.log_params({
            "window": WINDOW,
            "horizon": HORIZON,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "patience": PATIENCE,
            "learning_rate": TRAIN_CFG["learning_rate"],
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_test": len(X_test),
        })

        print("\n--- Entrenando Naive ---")
        y_pred_naive, metrics_naive = train_naive(X_test, y_test, scaler)
        mlflow.log_metrics({
            "naive_rmse": metrics_naive["rmse"],
            "naive_mae": metrics_naive["mae"],
            "naive_mape": metrics_naive["mape"],
        })
        resultados["naive"] = metrics_naive

        if HAS_TF:
            print("\n--- Entrenando LSTM ---")
            lstm_model = build_lstm(WINDOW, HORIZON)
            lstm_model, history_lstm = train_dl_model(lstm_model, X_train, y_train, X_val, y_val, "LSTM")
            y_pred_lstm = lstm_model.predict(X_test, verbose=0)
            metrics_lstm = evaluate_metrics(y_test, y_pred_lstm, scaler)
            mlflow.log_metrics({
                "lstm_rmse": metrics_lstm["rmse"],
                "lstm_mae": metrics_lstm["mae"],
                "lstm_mape": metrics_lstm["mape"],
            })
            resultados["lstm"] = metrics_lstm

            hist_path_lstm = "training_curve_lstm.png"
            forecast_path_lstm = "forecast_comparison_lstm.png"
            plot_training_history(history_lstm, "LSTM", hist_path_lstm)
            plot_forecast_comparison(y_test, y_pred_lstm, scaler, "LSTM", forecast_path_lstm)
            mlflow.log_artifact(hist_path_lstm)
            mlflow.log_artifact(forecast_path_lstm)

            print("\n--- Entrenando GRU ---")
            gru_model = build_gru(WINDOW, HORIZON)
            gru_model, history_gru = train_dl_model(gru_model, X_train, y_train, X_val, y_val, "GRU")
            y_pred_gru = gru_model.predict(X_test, verbose=0)
            metrics_gru = evaluate_metrics(y_test, y_pred_gru, scaler)
            mlflow.log_metrics({
                "gru_rmse": metrics_gru["rmse"],
                "gru_mae": metrics_gru["mae"],
                "gru_mape": metrics_gru["mape"],
            })
            resultados["gru"] = metrics_gru

            hist_path_gru = "training_curve_gru.png"
            forecast_path_gru = "forecast_comparison_gru.png"
            plot_training_history(history_gru, "GRU", hist_path_gru)
            plot_forecast_comparison(y_test, y_pred_gru, scaler, "GRU", forecast_path_gru)
            mlflow.log_artifact(hist_path_gru)
            mlflow.log_artifact(forecast_path_gru)

            best_dl = min(
                [("lstm", metrics_lstm), ("gru", metrics_gru)],
                key=lambda x: x[1]["mape"]
            )
            best_dl_name = best_dl[0]
            best_dl_model = {"lstm": lstm_model, "gru": gru_model}[best_dl_name]
            best_dl_metrics = best_dl[1]
        else:
            best_dl_name = None
            best_dl_model = None
            best_dl_metrics = {"mape": float("inf")}

        all_candidates = {"naive": metrics_naive}
        if best_dl_name:
            all_candidates[best_dl_name] = best_dl_metrics

        best_model_name = min(all_candidates, key=lambda k: all_candidates[k]["mape"])
        best_metrics = all_candidates[best_model_name]

        mlflow.log_params({"best_model": best_model_name})
        mlflow.log_metrics({
            "best_rmse": best_metrics["rmse"],
            "best_mae": best_metrics["mae"],
            "best_mape": best_metrics["mape"],
        })

        if best_model_name == "naive":
            class NaivePyfunc(mlflow.pyfunc.PythonModel):
                def __init__(self, horizon):
                    self.horizon = horizon
                def predict(self, context, model_input):
                    arr = model_input.values if hasattr(model_input, "values") else model_input
                    ultimo = arr[:, -1, 0]
                    return np.repeat(ultimo.reshape(-1, 1), self.horizon, axis=1)
            mlflow.pyfunc.log_model("best_model", python_model=NaivePyfunc(HORIZON))
        elif best_dl_model is not None:
            mlflow.tensorflow.log_model(best_dl_model, "best_model")

        import pickle
        import json

        scaler_path = "scaler.pkl"
        config_path = "model_config.json"
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        with open(config_path, "w") as f:
            json.dump({
                "window": WINDOW,
                "horizon": HORIZON,
                "best_model": best_model_name,
                "metrics": {k: float(v) for k, v in best_metrics.items()},
                "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
            }, f, indent=2)

        mlflow.log_artifact(scaler_path)
        mlflow.log_artifact(config_path)

        print("\n" + "=" * 60)
        print(f"Run ID: {run.info.run_id}")
        print(f"Mejor modelo: {best_model_name}")
        print(f"MAPE: {best_metrics['mape']:.3f}%")
        print("=" * 60)

        return run.info.run_id, best_model_name, best_metrics


if __name__ == "__main__":
    main()
