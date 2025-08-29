from eda_listings import eda_listings
from eda_calendar import eda_calendar
from eda_reviews import eda_reviews
import os

DATASETS = {
    "paris": {
        "listings": "data/processed/paris/listings_clean.csv",
        "calendar": "data/processed/paris/calendar_clean.csv",
        "reviews": "data/processed/paris/reviews_clean8000.csv"
    },
    "seattle": {
        "listings": "data/processed/seattle/listings_clean.csv",
        "calendar": "data/processed/seattle/calendar_clean.csv",
        "reviews": "data/processed/seattle/reviews_clean20000.csv"
    }
}

if __name__ == "__main__":
    for city, files in DATASETS.items():
        print(f"\n==== {city.upper()} ====")
        if os.path.exists(files["listings"]):
            eda_listings(city, files["listings"])
        if os.path.exists(files["calendar"]):
            eda_calendar(city, files["calendar"])
        if os.path.exists(files["reviews"]):
            eda_reviews(city, files["reviews"])
