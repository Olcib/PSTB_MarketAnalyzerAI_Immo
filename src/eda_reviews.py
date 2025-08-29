import pandas as pd

def eda_reviews(city, fp):
    df = pd.read_csv(fp, parse_dates=["date"])
    print(df.shape)
    print(df["date"].min(), df["date"].max())
    print("Nb de reviews par an:")
    print(df.groupby(df["date"].dt.year).size())
    df["comment_len"] = df["comments"].astype(str).str.len()
    print("Longueur moyenne des commentaires:", df["comment_len"].mean())
    print("Reviewers actifs:", df["reviewer_id"].value_counts().head(10))
