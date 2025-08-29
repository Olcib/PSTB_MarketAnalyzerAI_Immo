import pandas as pd
import argparse
from pathlib import Path

def split_column(input_file, column_name, sep=",", output_file=None):
    """
    Sépare une colonne d'un CSV en plusieurs colonnes.

    Args:
        input_file (str): chemin du CSV d'entrée
        column_name (str): nom de la colonne à éclater
        sep (str): séparateur utilisé dans cette colonne (ex: ',' ou '|')
        output_file (str): chemin du CSV de sortie (si None -> ajoute '_split')
    """
    # Charger le fichier CSV
    df = pd.read_csv(input_file)

    if column_name not in df.columns:
        raise ValueError(f"Colonne {column_name} non trouvée dans {input_file}")

    # Séparer la colonne
    split_cols = df[column_name].astype(str).str.split(sep, expand=True)

    # Renommer les nouvelles colonnes
    split_cols.columns = [f"{column_name}_{i+1}" for i in range(split_cols.shape[1])]

    # Fusionner avec le DataFrame original
    df_out = pd.concat([df.drop(columns=[column_name]), split_cols], axis=1)

    # Déterminer le fichier de sortie
    if output_file is None:
        p = Path(input_file)
        output_file = str(p.with_name(p.stem + "_split.csv"))

    # Sauvegarder
    df_out.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"✅ Fichier généré: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Séparer une colonne en plusieurs colonnes dans un CSV")
    parser.add_argument("input_file", help="Chemin du CSV d'entrée")
    parser.add_argument("column_name", help="Nom de la colonne à séparer")
    parser.add_argument("--sep", default=",", help="Séparateur utilisé dans la colonne (par défaut ',')")
    parser.add_argument("--output", default=None, help="Chemin du fichier CSV de sortie")

    args = parser.parse_args()
    split_column(args.input_file, args.column_name, args.sep, args.output)
