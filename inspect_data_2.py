
import pandas as pd
import os

files = [
    r"c:\Users\note-ceart02\Documents\GitHub\app_qualificacao\data\agrupado_cursos_concluidos_em_execucao_sem_sebrae.csv",
    r"c:\Users\note-ceart02\Documents\GitHub\app_qualificacao\data2\compilado_trilha_sebrae.csv"
]

for filepath in files:
    print(f"\n--- Analyzing: {os.path.basename(filepath)} ---")
    try:
        # Try finding a working separator
        try:
             df = pd.read_csv(filepath, sep=None, engine='python', nrows=1000)
        except:
             df = pd.read_csv(filepath, sep=';', encoding='latin1', nrows=1000)
             
        print(f"Columns: {list(df.columns)}")
        print(f"Shape (sample): {df.shape}")
        
        # Check specific columns for usefulness
        print("Missing Values (%):")
        print(df.isnull().mean() * 100)
        
        # Check for text columns for NLP
        object_cols = df.select_dtypes(include=['object']).columns
        print(f"\nText Columns: {list(object_cols)}")
        if len(object_cols) > 0:
            print("Sample text:")
            print(df[object_cols[0]].head(3).values)

    except Exception as e:
        print(f"Error: {e}")
