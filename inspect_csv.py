import pandas as pd
try:
    df = pd.read_csv('data2/compilado_novos_lotes_merge_nomes_cozinhas_05122025.csv', sep=';', encoding='utf-8', nrows=5)
    print("Columns found:")
    for col in df.columns:
        print(f"- {col}")
except Exception as e:
    print(f"Error: {e}")
