"""
Flask serving server for BTC forecast.
Carga modelo desde MLflow artifacts y expone endpoints:
  GET /              - Status
  GET /consultar     - Forecast completo
  GET /signal        - Senal de inversion (para N8N)
"""

import os
import sys
import json
import pickle
import threading
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from flask import Flask, jsonify
import mlflow
import logging
import subprocess

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "serving.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger("serving")

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

# ---------- Config ----------
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

SERVING_CFG = cfg["serving"]
MODEL_CFG = cfg["model"]
MLFLOW_CFG = cfg["mlflow"]
ALERT_CFG = cfg["alert"]

WINDOW = MODEL_CFG["window"]
HORIZON = MODEL_CFG["horizon"]
MODEL_STAGE = SERVING_CFG["model_stage"]
REGISTRY_NAME = MLFLOW_CFG["registry_model_name"]
UPDATE_INTERVAL = SERVING_CFG["update_interval_seconds"]
MIN_GAIN_PCT = ALERT_CFG["min_gain_pct"]

# ---------- MLflow ----------
mlflow.set_tracking_uri(MLFLOW_CFG["tracking_uri"])

# ---------- Global state ----------
modelo_bundle = None
modelo_lock = threading.Lock()
ultimo_forecast = None
ultimo_forecast_lock = threading.Lock()


class NaiveForecastModel:
    def __init__(self, horizon):
        self.horizon = horizon

    def predict(self, X):
        ultimo_valor = X[:, -1, 0]
        return np.repeat(ultimo_valor.reshape(-1, 1), self.horizon, axis=1)


def cargar_modelo_desde_artifacts():
    print("[MLflow] Buscando modelo en el registry...")
    client = mlflow.tracking.MlflowClient()

    versions = client.search_model_versions(f"name='{REGISTRY_NAME}'")
    prod_versions = [v for v in versions if v.current_stage == MODEL_STAGE]

    if prod_versions:
        version = max(prod_versions, key=lambda v: int(v.version))
    else:
        all_versions = sorted(versions, key=lambda v: int(v.version), reverse=True)
        if not all_versions:
            raise ValueError("No se encontraron modelos en el registry. Ejecuta mlops/register_model.py primero.")
        version = all_versions[0]

    run_id = version.run_id
    print(f"[MLflow] Usando {REGISTRY_NAME} v{version.version} (run: {run_id})")

    scaler_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="scaler.pkl")
    config_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model_config.json")
    bundle_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model_bundle.pkl")

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    with open(config_path) as f:
        config_data = json.load(f)

    with open(bundle_path, "rb") as f:
        bundle_data = pickle.load(f)

    model_type = bundle_data["tipo"]
    if model_type == "Naive":
        modelo = bundle_data["modelo"]
    else:
        from tensorflow import keras
        modelo = keras.models.model_from_json(bundle_data["arquitectura_json"])
        modelo.set_weights(bundle_data["pesos"])

    bundle = {
        "modelo": modelo,
        "model_type": model_type,
        "scaler": scaler,
        "config": config_data,
        "bundle_data": bundle_data,
        "window": bundle_data["window"],
        "horizon": bundle_data["horizon"],
        "loaded_version": int(version.version),
    }
    print(f"[MLflow] Modelo cargado: tipo={model_type}")
    return bundle


def obtener_version_production():
    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name='{REGISTRY_NAME}'")
    prod_versions = [v for v in versions if v.current_stage == MODEL_STAGE]
    if prod_versions:
        latest = max(prod_versions, key=lambda v: int(v.version))
        return int(latest.version)
    return 0


def obtener_ultimos_precios(window):
    btc = yf.Ticker("BTC-USD")
    df = btc.history(period="7d", interval="1h")
    serie = df[["Close"]].dropna()

    if len(serie) < window:
        raise ValueError(f"Se necesitan {window} horas, se obtuvieron {len(serie)}.")

    return serie.index[-1], serie["Close"].values[-window:]


def predecir(bundle, ultimos_valores):
    scaler = bundle["scaler"]
    window = bundle["window"]
    modelo = bundle["modelo"]

    scaled = scaler.transform(ultimos_valores.reshape(-1, 1)).flatten()
    X_input = scaled.reshape(1, window, 1)

    if bundle["model_type"] == "Naive":
        pred_scaled = modelo.predict(X_input)[0]
    else:
        pred_scaled = modelo.predict(X_input, verbose=0)[0]

    return scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()


def construir_forecast_df(ultima_fecha, precios_predichos, horizon):
    fechas_futuras = pd.date_range(start=ultima_fecha, periods=horizon + 1, freq="h")[1:]
    return pd.DataFrame({
        "Datetime": fechas_futuras,
        "Precio_predicho": precios_predichos,
    })


def loop_prediccion():
    global modelo_bundle, ultimo_forecast
    logger.info("Loop thread started")
    while True:
        try:
            # 1. Verificar si hay nueva version en Production
            with modelo_lock:
                current_version = modelo_bundle.get("loaded_version", 0) if modelo_bundle else 0
            latest_version = obtener_version_production()
            logger.info(f"Version check: loaded=v{current_version}, registry=v{latest_version}")

            if latest_version > current_version:
                with modelo_lock:
                    msg = f"Nueva version v{latest_version} detectada (actual: v{current_version}). Recargando..."
                    print(f"[MLflow] {msg}")
                    logger.info(f"[MLflow] {msg}")
                    try:
                        modelo_bundle = cargar_modelo_desde_artifacts()
                        print(f"[MLflow] Modelo actualizado a v{latest_version}")
                        logger.info(f"[MLflow] Modelo actualizado a v{latest_version}")
                    except Exception as e:
                        print(f"[ERROR] Fallo al recargar modelo: {e}")
                        logger.error(f"Fallo al recargar modelo: {e}")

            # 2. Obtener referencia segura al modelo actual
            with modelo_lock:
                bundle = modelo_bundle

            logger.info("Downloading prices...")
            ultima_fecha, ultimos_valores = obtener_ultimos_precios(bundle["window"])
            logger.info(f"Got {len(ultimos_valores)} price points")

            logger.info("Making prediction...")
            precios_predichos = predecir(bundle, ultimos_valores)
            forecast_df = construir_forecast_df(ultima_fecha, precios_predichos, bundle["horizon"])

            mejor_hora = forecast_df.loc[forecast_df["Precio_predicho"].idxmax()]
            precio_actual = precios_predichos[0]
            precio_mejor = mejor_hora["Precio_predicho"]
            gain_pct = ((precio_mejor - precio_actual) / precio_actual) * 100

            signal = "BUY" if gain_pct >= MIN_GAIN_PCT else "HOLD"

            payload = {
                "modelo": bundle["config"].get("best_model", bundle["model_type"].lower()) if bundle["config"] else bundle["model_type"].lower(),
                "modelo_version": bundle.get("loaded_version", 0),
                "generado_en": datetime.now(timezone.utc).isoformat(),
                "precio_actual": round(float(precio_actual), 2),
                "forecast": [
                    {
                        "datetime": fila["Datetime"].isoformat(),
                        "precio_predicho": round(float(fila["Precio_predicho"]), 2),
                    }
                    for _, fila in forecast_df.iterrows()
                ],
                "mejor_momento": {
                    "datetime": mejor_hora["Datetime"].isoformat(),
                    "precio_predicho": round(float(precio_mejor), 2),
                },
                "signal": signal,
                "expected_gain_pct": round(float(gain_pct), 3),
                "min_gain_threshold_pct": MIN_GAIN_PCT,
            }

            with ultimo_forecast_lock:
                ultimo_forecast = payload

            msg = f"Signal: {signal} | Gain: {gain_pct:.3f}% | Mejor: ${precio_mejor:,.2f}"
            print(f"[{datetime.now(timezone.utc).isoformat()}] {msg}")
            logger.info(msg)

        except Exception as exc:
            print(f"[ERROR] No se pudo generar forecast: {exc}")
            logger.error(f"No se pudo generar forecast: {exc}", exc_info=True)

        logger.info(f"Sleeping {UPDATE_INTERVAL}s...")
        time.sleep(UPDATE_INTERVAL)


# ---------- Flask App ----------
app = Flask(__name__)


@app.route("/")
def index():
    with modelo_lock:
        bundle = modelo_bundle
    if bundle is None:
        return jsonify({"status": "loading"})
    return jsonify({
        "status": "ok",
        "modelo": bundle["config"].get("best_model", "unknown") if bundle.get("config") else bundle.get("model_type", "unknown"),
        "version": bundle.get("loaded_version", 0),
    })


@app.route("/consultar", methods=["GET"])
def consultar():
    with ultimo_forecast_lock:
        if ultimo_forecast is None:
            return jsonify({"status": "pendiente", "mensaje": "Aun no hay forecast calculado."}), 503
        return jsonify(ultimo_forecast)


@app.route("/signal", methods=["GET"])
def signal():
    with ultimo_forecast_lock:
        if ultimo_forecast is None:
            return jsonify({"status": "pendiente", "mensaje": "Aun no hay senial calculada."}), 503

        return jsonify({
            "signal": ultimo_forecast["signal"],
            "confidence": round(max(0, min(100, ultimo_forecast["expected_gain_pct"])), 2),
            "current_price": ultimo_forecast["precio_actual"],
            "best_time": ultimo_forecast["mejor_momento"]["datetime"],
            "predicted_price": ultimo_forecast["mejor_momento"]["precio_predicho"],
            "expected_gain_pct": ultimo_forecast["expected_gain_pct"],
            "min_gain_threshold_pct": ultimo_forecast["min_gain_threshold_pct"],
            "modelo": ultimo_forecast["modelo"],
            "generado_en": ultimo_forecast["generado_en"],
        })


@app.route("/retrain", methods=["POST"])
def retrain():
    project_root = os.path.join(os.path.dirname(__file__), "..")
    try:
        result_train = subprocess.run(
            [sys.executable, "training/train.py"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if result_train.returncode != 0:
            return jsonify({
                "status": "error",
                "message": "Entrenamiento fallo",
                "stderr": result_train.stderr[-500:],
            }), 500

        result_reg = subprocess.run(
            [sys.executable, "mlops/register_model.py"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result_reg.returncode != 0:
            return jsonify({
                "status": "error",
                "message": "Registro fallo",
                "stderr": result_reg.stderr[-500:],
            }), 500

        return jsonify({
            "status": "ok",
            "message": "Modelo reentrenado y registrado. El servidor lo detectara automaticamente en el proximo ciclo.",
        })
    except subprocess.TimeoutExpired:
        return jsonify({"status": "error", "message": "Timeout durante reentrenamiento"}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    print("=" * 60)
    print("BTC Forecast Server - MLflow Registry")
    print("=" * 60)

    with modelo_lock:
        modelo_bundle = cargar_modelo_desde_artifacts()

    hilo = threading.Thread(target=loop_prediccion, daemon=True)
    hilo.start()

    app.run(
        host=SERVING_CFG["host"],
        port=SERVING_CFG["port"],
        debug=False,
    )
