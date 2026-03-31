import pandas as pd
import logging
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder
import joblib
from sklearn import set_config

set_config(transform_output="pandas")

NUM_COLS = ["age", "ratings", "pickup_time_minutes", "distance"]

NOMINAL_COLS = [
    "weather", "type_of_order", "type_of_vehicle",
    "festival", "city_type", "is_weekend", "order_time_of_day"
]

ORDINAL_COLS = ["traffic", "distance_type"]

TARGET = "time_taken"

TRAFFIC_ORDER = ["low", "medium", "high", "jam"]
DISTANCE_ORDER = ["short", "medium", "long", "very_long"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def load_data(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        logger.info(f"Loaded data from {path}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def drop_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.info(f"Original shape: {df.shape}")
        df_clean = df.dropna()

        if df_clean.isna().sum().sum() > 0:
            raise ValueError("Missing values still present after dropna")

        logger.info(f"After dropping NA: {df_clean.shape}")
        return df_clean

    except Exception as e:
        logger.error(f"Error dropping missing values: {e}")
        raise


def make_X_y(df: pd.DataFrame, target: str):
    try:
        X = df.drop(columns=[target])
        y = df[target]
        return X, y
    except Exception as e:
        logger.error(f"Error splitting X and y: {e}")
        raise


def build_preprocessor() -> ColumnTransformer:
    try:
        preprocessor = ColumnTransformer(
            transformers=[
                ("scale", MinMaxScaler(), NUM_COLS),
                ("nominal", OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False
                ), NOMINAL_COLS),
                ("ordinal", OrdinalEncoder(
                    categories=[TRAFFIC_ORDER, DISTANCE_ORDER],
                    encoded_missing_value=-999,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1
                ), ORDINAL_COLS)
            ],
            remainder="passthrough",
            n_jobs=-1,
            verbose_feature_names_out=False
        )
        return preprocessor

    except Exception as e:
        logger.error(f"Error building preprocessor: {e}")
        raise


def fit_preprocessor(preprocessor, X: pd.DataFrame):
    try:
        preprocessor.fit(X)
        logger.info("Preprocessor trained")
        return preprocessor
    except Exception as e:
        logger.error(f"Error fitting preprocessor: {e}")
        raise


def transform_data(preprocessor, X: pd.DataFrame):
    try:
        return preprocessor.transform(X)
    except Exception as e:
        logger.error(f"Error transforming data: {e}")
        raise


def join_X_y(X: pd.DataFrame, y: pd.Series):
    try:
        return X.join(y, how="inner")
    except Exception as e:
        logger.error(f"Error joining X and y: {e}")
        raise


def save_data(df: pd.DataFrame, path: Path):
    try:
        df.to_csv(path, index=False)
        logger.info(f"Saved data to {path}")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise


def save_preprocessor(preprocessor, path: Path):
    try:
        joblib.dump(preprocessor, path)
        logger.info(f"Preprocessor saved at {path}")
    except Exception as e:
        logger.error(f"Error saving preprocessor: {e}")
        raise


def main():
    try:
        root = Path(__file__).parent.parent.parent

        train_path = root / "data" / "interim" / "train.csv"
        test_path = root / "data" / "interim" / "test.csv"

        processed_dir = root / "data" / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)

        train_out = processed_dir / "train_trans.csv"
        test_out = processed_dir / "test_trans.csv"

        model_dir = root / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        preprocessor_path = model_dir / "preprocessor.joblib"

        train_df = drop_missing_values(load_data(train_path))
        test_df = drop_missing_values(load_data(test_path))

        X_train, y_train = make_X_y(train_df, TARGET)
        X_test, y_test = make_X_y(test_df, TARGET)

        preprocessor = build_preprocessor()
        preprocessor = fit_preprocessor(preprocessor, X_train)

        X_train_trans = transform_data(preprocessor, X_train)
        X_test_trans = transform_data(preprocessor, X_test)

        train_final = join_X_y(X_train_trans, y_train)
        test_final = join_X_y(X_test_trans, y_test)

        save_data(train_final, train_out)
        save_data(test_final, test_out)

        save_preprocessor(preprocessor, preprocessor_path)

    except Exception as e:
        logger.critical(f"Pipeline failed: {e}")


if __name__ == "__main__":
    main()