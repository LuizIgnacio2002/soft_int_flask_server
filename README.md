# BTC Forecast - MLOps Pipeline

Sistema de prediccion de precios de Bitcoin con Deep Learning, MLOps con MLflow, serving con Flask, y alertas via N8N + WhatsApp.

## Arquitectura

```
yfinance → MLflow Training → Model Registry → Flask API → N8N Workflow → WhatsApp Alert
                              ↑
                          CI/CD (GitHub Actions)
```

## Estructura del Proyecto

```
├── training/                   # Pipeline de entrenamiento con MLflow
│   ├── train.py                # Entrena Naive, LSTM, GRU
│   └── requirements.txt
├── serving/                    # Flask server para inference
│   ├── main.py                 # API con endpoints /consultar y /signal
│   └── requirements.txt
├── mlops/                      # Registro de modelos
│   ├── register_model.py       # Registra mejor modelo en MLflow Registry
│   ├── import_existing_model.py
│   └── requirements.txt
├── tests/
│   └── test_config.py
├── scripts/                    # Scripts de utilidad
│   ├── retrain.bat
│   ├── retrain_scheduled.ps1
│   └── logs/
├── n8n/workflows/
│   └── btc_alert.json          # Workflow de alertas a WhatsApp (CallMeBot)
├── mlartifacts/                # Artefactos de MLflow (modelos, scalers, plots)
├── mlruns/                     # Datos locales de MLflow
├── mlflow.db                   # Base de datos SQLite de MLflow
├── logs/                       # Logs del servidor
├── config.yaml                 # Configuracion centralizada
├── model_config.json
├── scaler.pkl                  # Scaler usado en entrenamiento
├── forecast_comparison_*.png   # Graficos de comparacion de forecasts
├── training_curve_bilstm.png
├── Dockerfile
├── start_servers.bat
└── .github/workflows/
    └── ci_cd.yml               # Pipeline CI/CD
```

## Setup Local

### 1. Instalar dependencias

```bash
# Training
pip install -r training/requirements.txt

# Serving
pip install -r serving/requirements.txt

# MLOps
pip install -r mlops/requirements.txt
```

### 2. Iniciar MLflow Tracking Server

```bash
# Windows
start /b python -m mlflow server --host 0.0.0.0 --port 5001 --backend-store-uri sqlite:///./mlflow.db --default-artifact-root ./mlruns

# O usa el script incluido
start_servers.bat
```

Se accede en: `http://localhost:5001`

**Nota:** MLflow 3.x usa SQLite como backend por defecto. Los datos se guardan en `mlflow.db` y los artefactos en `mlartifacts/`.

### 3. Entrenar Modelos

```bash
python training/train.py
```

Entrena Naive, LSTM y GRU. Loguea metrics, artifacts y el mejor modelo en MLflow.

### 4. Registrar Mejor Modelo

```bash
python mlops/register_model.py
```

Busca el run con menor MAPE y lo registra en MLflow Model Registry como `Production`.

### 5. Iniciar Flask Server

```bash
python serving/main.py
```

Carga el modelo desde MLflow Registry y empieza a generar forecasts cada 3 minutos.

Endpoints:
- `GET /` → Status del servidor
- `GET /consultar` → Forecast completo (24 horas)
- `GET /signal` → Senal de compra/venta (para N8N)
- `POST /retrain` → Triggers reentrenamiento

### 6. Configurar N8N + WhatsApp

#### Instalar N8N (Docker)

```bash
docker run -it --rm --name n8n -p 5678:5678 -v ~/.n8n:/home/node/.n8n n8nio/n8n
```

Se accede en: `http://localhost:5678`

#### Configurar CallMeBot (WhatsApp)

1. Envia `"I allow callmebot to send me messages"` al numero **+34 644 71 64 33** via WhatsApp
2. Recibiras un **API Key** de CallMeBot
3. En N8N, ve a **Credentials** → **Create** → **HTTP Request**
4. Crea credenciales con tu **phone** y **api_key** (formato: `apikey=TU_API_KEY`)

#### Importar Workflow

1. En N8N, ve a **Workflows** → **Import from File**
2. Selecciona `n8n/workflows/btc_alert.json`
3. Abre el workflow importado
4. Configura los parametros `phone` y `apikey` en los nodos HTTP de CallMeBot
5. Activa el workflow (toggle **Active**)

### 7. Configurar Umbral de Alerta

Edita `config.yaml`:

```yaml
alert:
  min_gain_pct: 1.0  # Solo alerta si predice >= 1% de ganancia
```

## CI/CD

El pipeline `.github/workflows/ci_cd.yml` se ejecuta:
- En cada push a `main`
- Cada lunes a las 6:00 UTC (reentrenamiento semanal)

Jobs:
1. **test** → Valida configuracion y dependencias
2. **retrain** → Entrena modelos, loguea en MLflow, registra el mejor

## Configuracion

Todos los parametros estan en `config.yaml`:

| Parametro | Default | Descripcion |
|---|---|---|
| `model.window` | 72 | Horas de historia que ve el modelo |
| `model.horizon` | 24 | Horas futuras a predecir |
| `training.epochs` | 10 | Epochs maximos de entrenamiento |
| `training.batch_size` | 32 | Batch size |
| `mlflow.tracking_uri` | http://localhost:5001 | URL del MLflow server |
| `serving.port` | 5000 | Puerto del Flask server |
| `alert.min_gain_pct` | 1.0 | Umbral minimo de ganancia para alertar |

## Endpoints de la API

### GET /signal

```json
{
  "signal": "BUY",
  "confidence": 1.08,
  "current_price": 64716.48,
  "predicted_price": 65420.50,
  "best_time": "2026-07-20T14:00:00+00:00",
  "expected_gain_pct": 1.08,
  "modelo": "lstm",
  "generado_en": "2026-07-20T12:00:00+00:00"
}
```

### GET /consultar

```json
{
  "modelo": "lstm",
  "precio_actual": 64716.48,
  "forecast": [
    {"datetime": "...", "precio_predicho": 64716.48}
  ],
  "mejor_momento": {
    "datetime": "2026-07-20T14:00:00+00:00",
    "precio_predicho": 65420.50
  },
  "signal": "BUY",
  "expected_gain_pct": 1.08
}
```

## Notas

- El modelo se entrena con datos de los ultimos 730 dias (intervalo 1h) via `yfinance`
- Window: 72 horas pasadas → Horizon: 24 horas futuras
- El Flask server recalcula el forecast cada 3 minutos
- N8N consulta `/signal` cada 3 minutos y alerta a WhatsApp (via CallMeBot) si `expected_gain_pct >= min_gain_pct`
- Los artefactos de MLflow (modelos, scalers, graficos) se almacenan en `mlartifacts/` y `mlflow.db`
