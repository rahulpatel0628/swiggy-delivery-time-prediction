import json
import logging
from pathlib import Path
import mlflow
from mlflow import MlflowClient
import dagshub

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def setup_mlflow():
    try:
        dagshub.init(
            repo_owner="rahulpatel16092005",
            repo_name="swiggy-delivery-time-prediction",
            mlflow=True
        )

        tracking_uri = "https://dagshub.com/rahulpatel16092005/swiggy-delivery-time-prediction.mlflow"
        mlflow.set_tracking_uri(tracking_uri)

        logger.info("MLflow setup completed")
        return tracking_uri

    except Exception as e:
        logger.error(f"MLflow setup failed: {e}")
        raise


def load_run_info(path: Path) -> dict:
    try:
        if not path.exists():
            raise FileNotFoundError(f"Run info not found: {path}")

        with open(path, "r") as f:
            data = json.load(f)

        logger.info("Run information loaded")
        return data

    except Exception as e:
        logger.error(f"Error loading run info: {e}")
        raise


def get_artifacts(client: MlflowClient, run_id: str):
    try:
        artifacts = client.list_artifacts(run_id)
        paths = [a.path for a in artifacts]

        logger.info(f"Artifacts found: {paths}")
        return paths

    except Exception as e:
        logger.error(f"Error fetching artifacts: {e}")
        raise


def resolve_model_name(model_name: str, artifact_paths: list):
    try:
        if model_name in artifact_paths:
            return model_name

        if "model" in artifact_paths:
            logger.warning("Fallback to 'model' artifact")
            return "model"

        raise RuntimeError("Model artifact not found")

    except Exception as e:
        logger.error(f"Error resolving model name: {e}")
        raise


def register_model(run_id: str, model_name: str):
    try:
        model_uri = f"runs:/{run_id}/{model_name}"

        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )

        logger.info(f"Model registered: {model_version.name} v{model_version.version}")
        return model_version

    except Exception as e:
        logger.error(f"Error registering model: {e}")
        raise


def promote_to_staging(client: MlflowClient, model_version):
    try:
        client.transition_model_version_stage(
            name=model_version.name,
            version=model_version.version,
            stage="Staging",
            archive_existing_versions=False
        )

        logger.info("Model promoted to STAGING")

    except Exception as e:
        logger.error(f"Error promoting model: {e}")
        raise


def main():
    try:
        root = Path(__file__).parent.parent.parent
        run_info_path = root / "run_information.json"

        tracking_uri = setup_mlflow()
        client = MlflowClient(tracking_uri=tracking_uri)

        run_info = load_run_info(run_info_path)

        run_id = run_info["run_id"]
        model_name = run_info["model_name"]

        logger.info(f"Run ID: {run_id}")
        logger.info(f"Model Name: {model_name}")

        artifact_paths = get_artifacts(client, run_id)

        model_name = resolve_model_name(model_name, artifact_paths)

        model_version = register_model(run_id, model_name)

        promote_to_staging(client, model_version)

    except Exception as e:
        logger.critical(f"Model registration pipeline failed: {e}")


if __name__ == "__main__":
    main()