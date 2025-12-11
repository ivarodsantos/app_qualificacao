
import pandas as pd
from ml_utils import gerar_clusters_municipios

# Mock data creation similar to app structure
data = {
    "Código Município Completo": [2300101, 2300200, 2300309, 2300408, 2300507, 2300606, 2300705, 2300804],
    "Nome_Município": ["Abaiara", "Acarape", "Acaraú", "Acopiara", "Aiuaba", "Alcântaras", "Altaneira", "Alto Santo"],
    "CURSO": ["Culinária"] * 8,
    "VAGAS OFERTADAS": [20, 40, 60, 20, 100, 20, 20, 20],
    "CONCLUDENTES": [18, 10, 55, 5, 20, 19, 15, 12]
}
df = pd.DataFrame(data)

print("--- Entrada ---")
print(df)

try:
    df_clusters = gerar_clusters_municipios(df)
    print("\n--- Saída Clusters ---")
    print(df_clusters)
    print("\nSucesso!")
except Exception as e:
    print(f"\nErro: {e}")
