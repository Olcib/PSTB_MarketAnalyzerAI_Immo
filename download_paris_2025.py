import os
import requests
import gzip
import shutil

# URLs officielles (InsideAirbnb - Paris, 06 Juin 2025)
URLS = {
    "listings.csv.gz": "http://data.insideairbnb.com/france/ile-de-france/paris/2025-06-06/data/listings.csv.gz",
    "calendar.csv.gz": "http://data.insideairbnb.com/france/ile-de-france/paris/2025-06-06/data/calendar.csv.gz",
    "reviews.csv.gz": "http://data.insideairbnb.com/france/ile-de-france/paris/2025-06-06/data/reviews.csv.gz",
    "neighbourhoods.csv": "http://data.insideairbnb.com/france/ile-de-france/paris/2025-06-06/visualisations/neighbourhoods.csv",
    "neighbourhoods.geojson": "http://data.insideairbnb.com/france/ile-de-france/paris/2025-06-06/visualisations/neighbourhoods.geojson",
}

OUT_DIR = "data/raw/paris"
os.makedirs(OUT_DIR, exist_ok=True)

def download_file(url, out_path):
    print(f"⬇️  Downloading {url} ...")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        shutil.copyfileobj(r.raw, f)

    # Décompression si c'est un .gz
    if out_path.endswith(".gz"):
        decompressed_path = out_path[:-3]
        with gzip.open(out_path, "rb") as f_in:
            with open(decompressed_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(out_path)  # supprimer l’archive .gz
        print(f"✅ Extracted {decompressed_path}")
    else:
        print(f"✅ Saved {out_path}")


def main():
    for fname, url in URLS.items():
        out_path = os.path.join(OUT_DIR, fname)
        download_file(url, out_path)
    print("\n🎉 Tous les fichiers Paris (06 Juin 2025) ont été téléchargés dans:", OUT_DIR)


if __name__ == "__main__":
    main()
