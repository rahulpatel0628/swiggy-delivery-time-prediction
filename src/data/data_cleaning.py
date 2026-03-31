import numpy as np
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

COLUMNS_TO_DROP = [
    'rider_id', 'restaurant_latitude', 'restaurant_longitude',
    'delivery_latitude', 'delivery_longitude', 'order_date',
    'order_time_hour', 'order_day', 'city_name',
    'order_day_of_week', 'order_month'
]


def load_data(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        logger.info("Data loaded successfully")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def change_column_names(df: pd.DataFrame) -> pd.DataFrame:
    try:
        return (
            df.rename(str.lower, axis=1)
            .rename({
                "delivery_person_id": "rider_id",
                "delivery_person_age": "age",
                "delivery_person_ratings": "ratings",
                "delivery_location_latitude": "delivery_latitude",
                "delivery_location_longitude": "delivery_longitude",
                "time_orderd": "order_time",
                "time_order_picked": "order_picked_time",
                "weatherconditions": "weather",
                "road_traffic_density": "traffic",
                "city": "city_type",
                "time_taken(min)": "time_taken"
            }, axis=1)
        )
    except Exception as e:
        logger.error(f"Error renaming columns: {e}")
        raise


def data_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.replace("NaN ", np.nan)

        df = df[df['age'].astype(float) >= 18]
        df = df[df['ratings'] != "6"]

        df = (
            df.drop(columns="id")
            .assign(
                city_name=lambda x: x['rider_id'].str.split("RES").str.get(0),
                age=lambda x: x['age'].astype(float),
                ratings=lambda x: x['ratings'].astype(float),
                restaurant_latitude=lambda x: x['restaurant_latitude'].abs(),
                restaurant_longitude=lambda x: x['restaurant_longitude'].abs(),
                delivery_latitude=lambda x: x['delivery_latitude'].abs(),
                delivery_longitude=lambda x: x['delivery_longitude'].abs(),
                order_date=lambda x: pd.to_datetime(x['order_date'], dayfirst=True),
                order_day=lambda x: x['order_date'].dt.day,
                order_month=lambda x: x['order_date'].dt.month,
                order_day_of_week=lambda x: x['order_date'].dt.day_name().str.lower(),
                is_weekend=lambda x: x['order_date'].dt.day_name().isin(["Saturday", "Sunday"]).astype(int),
                order_time=lambda x: pd.to_datetime(x['order_time'], format='mixed'),
                order_picked_time=lambda x: pd.to_datetime(x['order_picked_time'], format='mixed'),
                pickup_time_minutes=lambda x: (x['order_picked_time'] - x['order_time']).dt.seconds / 60,
                order_time_hour=lambda x: x['order_time'].dt.hour,
                order_time_of_day=lambda x: time_of_day(x['order_time_hour']),
                weather=lambda x: x['weather'].str.replace("conditions ", "").str.lower().replace("nan", np.nan),
                traffic=lambda x: x['traffic'].str.strip().str.lower(),
                type_of_order=lambda x: x['type_of_order'].str.strip().str.lower(),
                type_of_vehicle=lambda x: x['type_of_vehicle'].str.strip().str.lower(),
                festival=lambda x: x['festival'].str.strip().str.lower(),
                city_type=lambda x: x['city_type'].str.strip().str.lower(),
                multiple_deliveries=lambda x: x['multiple_deliveries'].astype(float),
                time_taken=lambda x: x['time_taken'].str.replace("(min) ", "").astype(int)
            )
            .drop(columns=["order_time", "order_picked_time"])
        )

        logger.info("Data cleaning completed")
        return df

    except Exception as e:
        logger.error(f"Error in data cleaning: {e}")
        raise


def clean_lat_long(df: pd.DataFrame, threshold: float = 1.0) -> pd.DataFrame:
    try:
        cols = ['restaurant_latitude', 'restaurant_longitude',
                'delivery_latitude', 'delivery_longitude']

        for col in cols:
            df[col] = np.where(df[col] < threshold, np.nan, df[col])

        return df

    except Exception as e:
        logger.error(f"Error cleaning lat/long: {e}")
        raise


def time_of_day(series: pd.Series) -> pd.Series:
    return pd.cut(
        series,
        bins=[0, 6, 12, 17, 20, 24],
        labels=["after_midnight", "morning", "afternoon", "evening", "night"]
    )


def calculate_haversine_distance(df: pd.DataFrame) -> pd.DataFrame:
    try:
        lat1, lon1, lat2, lon2 = map(
            np.radians,
            [
                df['restaurant_latitude'],
                df['restaurant_longitude'],
                df['delivery_latitude'],
                df['delivery_longitude']
            ]
        )

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))

        df['distance'] = 6371 * c
        return df

    except Exception as e:
        logger.error(f"Error calculating distance: {e}")
        raise


def create_distance_type(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df['distance_type'] = pd.cut(
            df['distance'],
            bins=[0, 5, 10, 15, 25],
            labels=["short", "medium", "long", "very_long"]
        )
        return df

    except Exception as e:
        logger.error(f"Error creating distance type: {e}")
        raise


def drop_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    try:
        return df.drop(columns=columns)
    except Exception as e:
        logger.error(f"Error dropping columns: {e}")
        raise


def perform_data_cleaning(df: pd.DataFrame, save_path: Path) -> None:
    try:
        cleaned_df = (
            df.pipe(change_column_names)
              .pipe(data_cleaning)
              .pipe(clean_lat_long)
              .pipe(calculate_haversine_distance)
              .pipe(create_distance_type)
              .pipe(drop_columns, columns=COLUMNS_TO_DROP)
        )

        cleaned_df.to_csv(save_path, index=False)
        logger.info("Cleaned data saved successfully")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


def main():
    try:
        root = Path(__file__).parent.parent.parent
        raw_path = root / "data" / "raw" / "swiggy.csv"
        save_dir = root / "data" / "cleaned"
        save_dir.mkdir(parents=True, exist_ok=True)

        save_path = save_dir / "swiggy_cleaned.csv"

        df = load_data(raw_path)
        perform_data_cleaning(df, save_path)

    except Exception as e:
        logger.critical(f"Application failed: {e}")


if __name__ == "__main__":
    main()