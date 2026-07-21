"""
Registro del mejor modelo de MLflow en el Model Registry.
Busca el run FINISHED con menor MAPE y lo registra como nueva version.
"""

import os
import yaml
import mlflow

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

MLFLOW_CFG = cfg["mlflow"]
REGISTRY_NAME = MLFLOW_CFG["registry_model_name"]


def register_best_model():
    mlflow.set_tracking_uri(MLFLOW_CFG["tracking_uri"])
    client = mlflow.tracking.MlflowClient()

    experiment = mlflow.get_experiment_by_name(MLFLOW_CFG["experiment_name"])
    if experiment is None:
        raise ValueError(f"Experimento '{MLFLOW_CFG['experiment_name']}' no encontrado.")

    runs_all = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="status = 'FINISHED'",
        order_by=["start_time DESC"],
        max_results=50,
    )

    if not runs_all:
        raise ValueError("No se encontraron runs FINISHED.")

    # Tomar solo los runs con best_mape y ordenar por tiempo DESC
    runs = [r for r in runs_all if r.data.metrics.get("best_mape") is not None]

    if not runs:
        raise ValueError("No se encontraron runs con best_mape.")

    # El run mas reciente es el de la ultima sesion de entrenamiento
    latest_run = runs[0]
    runs = [latest_run]

    if not runs:
        raise ValueError("No se encontraron runs FINISHED con best_mape.")

    best_run = runs[0]
    run_id = best_run.info.run_id
    best_mape = best_run.data.metrics.get("best_mape", float("inf"))
    best_model_name = best_run.data.params.get("best_model", "unknown")

    print(f"Mejor run: {run_id}")
    print(f"Modelo: {best_model_name} | MAPE: {best_mape:.3f}%")

    model_uri = f"runs:/{run_id}/best_model"

    try:
        result = mlflow.register_model(model_uri=model_uri, name=REGISTRY_NAME)
        print(f"Modelo registrado: {result.name} v{result.version}")
        version = result.version
    except mlflow.exceptions.MlflowException as e:
        if "RESOURCE_ALREADY_EXISTS" in str(e):
            new_version = client.create_model_version(
                name=REGISTRY_NAME,
                source=model_uri,
                run_id=run_id,
            )
            print(f"Nueva version creada: v{new_version.version}")
            version = new_version.version
        else:
            raise

    # Archivar versiones anteriores en Production
    all_versions = client.search_model_versions(f"name='{REGISTRY_NAME}'")
    for v in all_versions:
        if str(v.version) != str(version) and v.current_stage == "Production":
            try:
                client.transition_model_version_stage(
                    name=REGISTRY_NAME,
                    version=v.version,
                    stage="Archived",
                )
                print(f"Version anterior v{v.version} archivada")
            except Exception:
                pass

    return REGISTRY_NAME, version


if __name__ == "__main__":
    name, version = register_best_model()
    print(f"\nListo para servir: models:/{name}/{version}")
