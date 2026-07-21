import os
import yaml
import pytest

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")


def test_config_exists():
    assert os.path.exists(CONFIG_PATH)


def test_config_valid():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    assert "model" in cfg
    assert "training" in cfg
    assert "mlflow" in cfg
    assert "serving" in cfg
    assert "alert" in cfg


def test_model_config():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    assert cfg["model"]["window"] == 72
    assert cfg["model"]["horizon"] == 24
    assert cfg["model"]["interval"] == "1h"
    assert cfg["model"]["period"] == "730d"


def test_training_config():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    assert cfg["training"]["epochs"] == 60
    assert cfg["training"]["batch_size"] == 32
    assert cfg["training"]["patience"] == 8
    assert "naive" in cfg["training"]["models"]
    assert "lstm" in cfg["training"]["models"]


def test_mlflow_config():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    assert "tracking_uri" in cfg["mlflow"]
    assert "experiment_name" in cfg["mlflow"]
    assert cfg["mlflow"]["experiment_name"] == "btc-forecast"


def test_serving_config():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    assert cfg["serving"]["port"] == 5000
    assert cfg["serving"]["update_interval_seconds"] == 180


def test_alert_config():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    assert cfg["alert"]["min_gain_pct"] == 1.0
    assert cfg["alert"]["check_interval_seconds"] == 180


def test_n8n_workflow_exists():
    workflow_path = os.path.join(os.path.dirname(__file__), "..", "n8n", "workflows", "btc_alert.json")
    assert os.path.exists(workflow_path)
