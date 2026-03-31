import pandas as pd
import yaml
import joblib
import logging
from pathlib import Path
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import PowerTransformer
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor

TARGET = "time_taken"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def load_data(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        logger.info(f"Data loaded from {path}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def read_params(path: Path) -> dict:
    try:
        with open(path, "r") as f:
            params = yaml.safe_load(f)
        logger.info("Parameters loaded successfully")
        return params
    except Exception as e:
        logger.error(f"Error reading params: {e}")
        raise


def split_X_y(df: pd.DataFrame, target: str):
    try:
        X = df.drop(columns=[target])
        y = df[target]
        return X, y
    except Exception as e:
        logger.error(f"Error splitting X and y: {e}")
        raise


def build_models(params: dict):
    try:
        rf = RandomForestRegressor(**params["Random_Forest"])
        lgbm = LGBMRegressor(**params["LightGBM"])
        meta = LinearRegression()

        stacking = StackingRegressor(
            estimators=[
                ("rf", rf),
                ("lgbm", lgbm)
            ],
            final_estimator=meta,
            cv=5,
            n_jobs=-1
        )

        transformer = PowerTransformer()

        model = TransformedTargetRegressor(
            regressor=stacking,
            transformer=transformer
        )

        logger.info("Models built successfully")
        return model

    except Exception as e:
        logger.error(f"Error building models: {e}")
        raise


def train_model(model, X: pd.DataFrame, y: pd.Series):
    try:
        model.fit(X, y)
        logger.info("Model training completed")
        return model
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise


def save_artifact(obj, path: Path):
    try:
        joblib.dump(obj, path)
        logger.info(f"Saved artifact at {path}")
    except Exception as e:
        logger.error(f"Error saving artifact: {e}")
        raise


def main():
    try:
        root = Path(__file__).parent.parent.parent

        data_path = root / "data" / "processed" / "train_trans.csv"
        params_path = root / "params.yaml"
        model_dir = root / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / "model.joblib"
        stacking_path = model_dir / "stacking_regressor.joblib"
        transformer_path = model_dir / "power_transformer.joblib"

        df = load_data(data_path)

        X_train, y_train = split_X_y(df, TARGET)

        params = read_params(params_path)["Train"]

        model = build_models(params)
        model = train_model(model, X_train, y_train)

        save_artifact(model, model_path)
        save_artifact(model.regressor_, stacking_path)
        save_artifact(model.transformer_, transformer_path)

    except Exception as e:
        logger.critical(f"Training pipeline failed: {e}")


if __name__ == "__main__":
    main()