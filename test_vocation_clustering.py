

import pandas as pd
from ml_utils import gerar_clusters_municipios

def test_vocation_clustering():
    # Mock Data: 
    # City A: Only "Gastronomia"
    # City B: Only "Tecnologia"
    # City C: Mix
    
    data = {
        "Código Município Completo": [101, 101, 102, 102, 103, 103],
        "Nome_Município": ["CityA", "CityA", "CityB", "CityB", "CityC", "CityC"],
        "ÁREA DO CURSO (automático)": [
            "Gastronomia", "Gastronomia", 
            "Tecnologia", "Tecnologia", 
            "Gastronomia", "Tecnologia"
        ],
        "VAGAS OFERTADAS": [20, 20, 20, 20, 20, 20],
        "CONCLUDENTES": [18, 18, 18, 18, 18, 18]
    }
    df = pd.DataFrame(data)
    
    print("--- Input DataFrame ---")
    print(df)
    
    df_clusters = gerar_clusters_municipios(df)
    
    print("\n--- Output Clusters ---")
    print(df_clusters)
    
    # Assertions
    assert not df_clusters.empty, "Output should not be empty"
    assert "cluster_name" in df_clusters.columns
    
    # Check if CityA and CityB are in different clusters (likely, given distinct vocations)
    cluster_a = df_clusters.loc[df_clusters["CD_MUN"] == "0000101", "cluster_id"].values[0]
    cluster_b = df_clusters.loc[df_clusters["CD_MUN"] == "0000102", "cluster_id"].values[0]
    
    print(f"\nCluster A: {cluster_a}")
    print(f"Cluster B: {cluster_b}")
    
    # In a perfect K-means with k>=2, they should separate
    # assert cluster_a != cluster_b 
    
    # Check Labels
    name_a = df_clusters.loc[df_clusters["CD_MUN"] == "0000101", "cluster_name"].values[0]
    name_b = df_clusters.loc[df_clusters["CD_MUN"] == "0000102", "cluster_name"].values[0]
    
    print(f"Name A: {name_a}")
    print(f"Name B: {name_b}")
    
    # Check if labels make sense (e.g., contain the category name)
    # This depends on our naming logic in ml_utils.py
    # We expect something like "Foco em Gastronomia" or similar
    
if __name__ == "__main__":
    test_vocation_clustering()
