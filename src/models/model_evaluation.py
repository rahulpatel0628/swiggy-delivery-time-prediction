import pandas as pd
import joblib
import logging
import mlflow
import mlflow.sklearn
import dagshub
from pathlib import Path
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import json
import shutil
import os

TARGET = "time_taken"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def setup_mlflow():
    try:
        dagshub.init(
            repo_owner='rahulpatel16092005',
            repo_name='swiggy-delivery-time-prediction',
            mlflow=True
        )

        mlflow.set_tracking_uri(
            "https://dagshub.com/rahulpatel16092005/swiggy-delivery-time-prediction.mlflow"
        )

        mlflow.set_experiment("DVC Pipeline")
        logger.info("MLflow setup completed")

    except Exception as e:
        logger.error(f"MLflow setup failed: {e}")
        raise


def load_data(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        logger.info(f"Loaded data from {path}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def split_X_y(df: pd.DataFrame, target: str):
    try:
        X = df.drop(columns=[target])
        y = df[target]
        return X, y
    except Exception as e:
        logger.error(f"Error splitting data: {e}")
        raise


def load_model(path: Path):
    try:
        model = joblib.load(path)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def evaluate_model(model, X_train, y_train, X_test, y_test):
    try:
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        metrics = {
            "train_mae": mean_absolute_error(y_train, y_train_pred),
            "test_mae": mean_absolute_error(y_test, y_test_pred),
            "train_r2": r2_score(y_train, y_train_pred),
            "test_r2": r2_score(y_test, y_test_pred)
        }

        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=5,
            scoring="neg_mean_absolute_error",
            n_jobs=-1
        )

        metrics["mean_cv_score"] = -cv_scores.mean()
        metrics["cv_scores"] = -cv_scores

        logger.info("Model evaluation completed")
        return metrics

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


def log_to_mlflow(model, metrics, train_df, test_df, root_path: Path):
    try:
        with mlflow.start_run() as run:

            mlflow.set_tag("model", "Food Delivery Time Regressor")
            mlflow.log_params(model.get_params())

            for k, v in metrics.items():
                if k != "cv_scores":
                    mlflow.log_metric(k, v)

            mlflow.log_metrics({
                f"cv_{i}": score for i, score in enumerate(metrics["cv_scores"])
            })

            train_input = mlflow.data.from_pandas(train_df, targets=TARGET)
            test_input = mlflow.data.from_pandas(test_df, targets=TARGET)

            mlflow.log_input(train_input, context="training")
            mlflow.log_input(test_input, context="validation")

            signature = mlflow.models.infer_signature(
                model_input=train_df.drop(columns=[TARGET]).sample(20, random_state=42),
                model_output=model.predict(train_df.drop(columns=[TARGET]).sample(20))
            )

            temp_path = root_path / "models" / "mlflow_model"

            if os.path.exists(temp_path):
                shutil.rmtree(temp_path)

            mlflow.sklearn.save_model(
                sk_model=model,
                path=temp_path,
                signature=signature
            )

            mlflow.log_artifacts(temp_path, artifact_path="model")
            shutil.rmtree(temp_path)

            mlflow.log_artifact(root_path / "models" / "stacking_regressor.joblib")
            mlflow.log_artifact(root_path / "models" / "power_transformer.joblib")
            mlflow.log_artifact(root_path / "models" / "preprocessor.joblib")

            artifact_uri = mlflow.get_artifact_uri()

            logger.info("MLflow logging completed")
            return run.info.run_id, artifact_uri

    except Exception as e:
        logger.error(f"MLflow logging failed: {e}")
        raise


def save_run_info(path: Path, run_id: str, artifact_uri: str):
    try:
        info = {
            "run_id": run_id,
            "artifact_path": artifact_uri,
            "model_name": "model"
        }

        with open(path, "w") as f:
            json.dump(info, f, indent=4)

        logger.info("Run information saved")

    except Exception as e:
        logger.error(f"Error saving run info: {e}")
        raise


def main():
    try:
        setup_mlflow()

        root = Path(__file__).parent.parent.parent

        train_path = root / "data" / "processed" / "train_trans.csv"
        test_path = root / "data" / "processed" / "test_trans.csv"
        model_path = root / "models" / "model.joblib"

        train_df = load_data(train_path)
        test_df = load_data(test_path)

        X_train, y_train = split_X_y(train_df, TARGET)
        X_test, y_test = split_X_y(test_df, TARGET)

        model = load_model(model_path)

        metrics = evaluate_model(model, X_train, y_train, X_test, y_test)

        run_id, artifact_uri = log_to_mlflow(
            model,
            metrics,
            train_df,
            test_df,
            root
        )

        save_run_info(
            root / "run_information.json",
            run_id,
            artifact_uri
        )

    except Exception as e:
        logger.critical(f"Evaluation pipeline failed: {e}")


if __name__ == "__main__":
    main()