import pandas as pd

def eda_calendar(city, fp):
    df = pd.read_csv(fp, parse_dates=["date"])
    print(df.shape)
    print(df["date"].min(), df["date"].max())
    df["available_flag"] = (df["available"] == "f").astype(int)
    occ = df.groupby(df["date"].dt.to_period("M"))["available_flag"].mean()
    print("Taux occupation mensuel:", occ.head())
