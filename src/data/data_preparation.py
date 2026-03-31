import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import logging
from pathlib import Path

TARGET = "time_taken"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def load_data(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        logger.info("Data loaded successfully")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def split_data(df: pd.DataFrame, test_size: float, random_state: int):
    try:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state
        )
        logger.info("Data split completed")
        return train_df, test_df
    except Exception as e:
        logger.error(f"Error splitting data: {e}")
        raise


def read_params(path: Path) -> dict:
    try:
        with open(path, "r") as f:
            params = yaml.safe_load(f)
        logger.info("Parameters loaded successfully")
        return params
    except Exception as e:
        logger.error(f"Error reading parameters: {e}")
        raise


def save_data(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_csv(path, index=False)
        logger.info(f"Data saved at {path}")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise


def main():
    try:
        root = Path(__file__).parent.parent.parent

        data_path = root / "data" / "cleaned" / "swiggy_cleaned.csv"
        save_dir = root / "data" / "interim"
        params_path = root / "params.yaml"

        save_dir.mkdir(parents=True, exist_ok=True)

        train_path = save_dir / "train.csv"
        test_path = save_dir / "test.csv"

        df = load_data(data_path)

        params = read_params(params_path)["Data_Preparation"]
        test_size = params["test_size"]
        random_state = params["random_state"]

        train_df, test_df = split_data(
            df,
            test_size=test_size,
            random_state=random_state
        )

        save_data(train_df, train_path)
        save_data(test_df, test_path)

    except Exception as e:
        logger.critical(f"Pipeline failed: {e}")


if __name__ == "__main__":
    main()