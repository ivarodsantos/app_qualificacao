import pandas as pd

file_path = 'data2/compilado_novos_lotes_merge_nomes_cozinhas_05122025.csv'

try:
    df = pd.read_csv(file_path, sep=';', encoding='utf-8', nrows=50) # Read a bit more to get meaningful samples
    
    print("ALL COLUMNS:")
    for col in df.columns:
        print(f" - {col}")
        
    cols_of_interest = ["ENDEREÇO DO LOCAL DO CURSO", "LOCAL DO CURSO", "CIDADE", "LATITUDE", "LONGITUDE", "lat", "lon"]
    
    print("\nCHECKING SPECIFIC COLUMNS:")
    for col in cols_of_interest:
        if col in df.columns:
            print(f"\nValues for '{col}':")
            print(df[col].dropna().unique()[:5])
        else:
            # Check for partial matches
            matches = [c for c in df.columns if col.lower() in c.lower()]
            if matches:
                print(f"\n'{col}' not found, but found similar: {matches}")
            else:
                print(f"\n'{col}' NOT FOUND.")

except Exception as e:
    print(f"Error: {e}")
