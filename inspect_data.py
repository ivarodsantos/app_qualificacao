
import pandas as pd
import glob
import os

def inspect_file(filepath):
    print(f"\n--- Analyzing: {os.path.basename(filepath)} ---")
    try:
        df = pd.read_csv(filepath, sep=None, engine='python', on_bad_lines='skip')
        print(f"Shape: {df.shape}")
        print("Missing Values per Column (%):")
        missing = df.isnull().mean() * 100
        print(missing[missing > 0].sort_values(ascending=False).to_string())
        print("\nData Types:")
        print(df.dtypes.to_string())
        print("\nSample Data:")
        print(df.head(3).to_string())
    except Exception as e:
        print(f"Error reading {filepath}: {e}")

files = [
    r"c:\Users\note-ceart02\Documents\GitHub\app_qualificacao\data\agrupado_cursos_concluidos_em_execucao_sem_sebrae.csv",
    r"c:\Users\note-ceart02\Documents\GitHub\app_qualificacao\data2\compilado_mentoria_sebrae.csv",
    r"c:\Users\note-ceart02\Documents\GitHub\app_qualificacao\data2\compilado_trilha_sebrae.csv",
    r"c:\Users\note-ceart02\Documents\GitHub\app_qualificacao\data2\data-1762178638816_kitchen.csv"
]

for f in files:
    inspect_file(f)
