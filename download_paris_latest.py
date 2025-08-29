import os
import re
import requests
from bs4 import BeautifulSoup
import gzip
import shutil

BASE_URL = "http://data.insideairbnb.com/france/ile-de-france/paris/"
DATA_DIR = "data/raw/paris"

def fetch_latest_date():
    # Scraper la page d'exploration des données pour Paris
    explore_url = "https://insideairbnb.com/explore/?city=Paris"
    resp = requests.get(explore_url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    # Extraire les liens contenant la structure /paris/YYYY-MM-DD/
    links = soup.find_all("a", href=True)
    dates = []
    pattern = re.compile(r"/france/ile-de-france/paris/(\d{4}-\d{2}-\d{2})/")
    for a in links:
        m = pattern.search(a['href'])
        if m:
            dates.append(m.group(1))
    # Retourner la date la plus récente
    return max(dates) if dates else None

def download_and_extract(date_str):
    urls = {
        "listings.csv.gz": f"{BASE_URL}{date_str}/data/listings.csv.gz",
        "calendar.csv.gz": f"{BASE_URL}{date_str}/data/calendar.csv.gz",
        "reviews.csv.gz": f"{BASE_URL}{date_str}/data/reviews.csv.gz",
    }
    os.makedirs(DATA_DIR, exist_ok=True)
    for fname, url in urls.items():
        out_path = os.path.join(DATA_DIR, fname)
        print(f"Downloading {url} -> {out_path} …")
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            shutil.copyfileobj(r.raw, f)
        # Décompression
        with gzip.open(out_path, "rb") as f_in:
            with open(out_path[:-3], "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(out_path)
        print(f"Extracted to {out_path[:-3]}")

def main():
    print("Fetching latest available date for Paris …")
    date_str = fetch_latest_date()
    if not date_str:
        print("  ERREUR : impossible de trouver une date disponible.")
        return
    print(f"Latest data date found: {date_str}")
    download_and_extract(date_str)
    print("  ✅ Download and extraction complete.")

if __name__ == "__main__":
    main()
