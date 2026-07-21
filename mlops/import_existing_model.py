"""
Importa los modelos ya entrenados (.pkl) en MLflow sin reentrenar.
Lee btc_modelo_final.pkl, btc_scaler.pkl y los registra en MLflow.
"""

import os
import sys
import json
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml
import mlflow

sys.stdout.reconfigure(encoding="utf-8")

# ---------- Config ----------
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

MLFLOW_CFG = cfg["mlflow"]
MODEL_CFG = cfg["model"]

WINDOW = MODEL_CFG["window"]
HORIZON = MODEL_CFG["horizon"]
REGISTRY_NAME = MLFLOW_CFG["registry_model_name"]

# ---------- Paths a modelos existentes ----------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..")
MODEL_PKL = os.path.join(MODEL_DIR, "btc_modelo_final.pkl")
SCALER_PKL = os.path.join(MODEL_DIR, "btc_scaler.pkl")
CONFIG_PKL = os.path.join(MODEL_DIR, "btc_config.pkl")


class NaiveForecastModel:
    def __init__(self, horizon):
        self.horizon = horizon

    def predict(self, X):
        ultimo_valor = X[:, -1, 0]
        return np.repeat(ultimo_valor.reshape(-1, 1), self.horizon, axis=1)


class BtcForecastWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, bundle, scaler, model_type):
        self.bundle = bundle
        self.scaler = scaler
        self.model_type = model_type
        self.window = bundle["window"]
        self.horizon = bundle["horizon"]

    def predict(self, context, model_input):
        if self.model_type == "Naive":
            return self.bundle["modelo"].predict(model_input.values if hasattr(model_input, "values") else model_input)
        else:
            from tensorflow import keras
            keras_model = keras.models.model_from_json(self.bundle["arquitectura_json"])
            keras_model.set_weights(self.bundle["pesos"])
            return keras_model.predict(model_input, verbose=0)

    def forecast_from_prices(self, ultimos_valores):
        scaled = self.scaler.transform(ultimos_valores.reshape(-1, 1)).flatten()
        X_input = scaled.reshape(1, self.window, 1)

        if self.model_type == "Naive":
            pred_scaled = self.bundle["modelo"].predict(X_input)[0]
        else:
            from tensorflow import keras
            keras_model = keras.models.model_from_json(self.bundle["arquitectura_json"])
            keras_model.set_weights(self.bundle["pesos"])
            pred_scaled = keras_model.predict(X_input, verbose=0)[0]

        return self.scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()


def cargar_bundle():
    with open(MODEL_PKL, "rb") as f:
        bundle = pickle.load(f)
    return bundle


def calcular_mape_real(bundle, scaler):
    import yfinance as yf
    model_type = bundle["tipo"]
    window = bundle["window"]
    horizon = bundle["horizon"]
    modelo = bundle["modelo"]

    btc = yf.Ticker("BTC-USD")
    df = btc.history(period="730d", interval="1h")
    serie = df[["Close"]].dropna()
    values = serie["Close"].values.reshape(-1, 1)

    n = len(values)
    test_start = int(n * 0.9)
    test_raw = values[test_start:]
    test_scaled = scaler.transform(test_raw).flatten()

    errors = []
    for i in range(len(test_scaled) - window - horizon + 1):
        X = test_scaled[i:i+window].reshape(1, window, 1)
        if model_type == "Naive":
            pred_scaled = modelo.predict(X)[0]
        else:
            pred_scaled = modelo.predict(X, verbose=0)[0]
        pred_real = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        true_real = scaler.inverse_transform(test_scaled[i+window:i+window+horizon].reshape(-1, 1)).flatten()
        mape = np.mean(np.abs((true_real - pred_real) / true_real)) * 100
        errors.append(mape)

    return np.mean(errors) if errors else float("inf")


def importar_a_mlflow():
    mlflow.set_tracking_uri(MLFLOW_CFG["tracking_uri"])
    mlflow.set_experiment(MLFLOW_CFG["experiment_name"])

    bundle = cargar_bundle()
    model_type = bundle["tipo"]
    window = bundle["window"]
    horizon = bundle["horizon"]

    with open(SCALER_PKL, "rb") as f:
        scaler = pickle.load(f)

    print(f"Modelo existente: tipo={model_type}, window={window}, horizon={horizon}")
    print("Calculando MAPE real con datos actuales...")
    real_mape = calcular_mape_real(bundle, scaler)
    print(f"MAPE real calculado: {real_mape:.3f}%")

    with mlflow.start_run(run_name=f"import-existing-{model_type.lower()}") as run:
        mlflow.log_params({
            "window": window,
            "horizon": horizon,
            "modelo_tipo": model_type,
            "origen": "existing_pkl_import",
        })
        mlflow.log_metrics({
            "naive_mape": real_mape,
            "best_mape": real_mape,
        })

        wrapper = BtcForecastWrapper(bundle, scaler, model_type)
        mlflow.pyfunc.log_model("best_model", python_model=wrapper)

        scaler_path = "scaler.pkl"
        bundle_path = "model_bundle.pkl"
        config_path = "model_config.json"

        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        with open(bundle_path, "wb") as f:
            pickle.dump(bundle, f)
        with open(config_path, "w") as f:
            json.dump({
                "window": window,
                "horizon": horizon,
                "best_model": model_type.lower(),
                "modelo_tipo": model_type,
                "metrics": {"mape": real_mape},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }, f, indent=2)

        mlflow.log_artifact(scaler_path)
        mlflow.log_artifact(bundle_path)
        mlflow.log_artifact(config_path)

        print(f"Run ID: {run.info.run_id}")
        print(f"Modelo importado exitosamente a MLflow como pyfunc")

        return run.info.run_id


if __name__ == "__main__":
    run_id = importar_a_mlflow()
    print(f"\nSiguiente paso: python mlops/register_model.py")
    print(f"O usa este run directamente: runs:/{run_id}/best_model")
