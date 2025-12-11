import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def gerar_clusters_municipios(df_input):
    """
    Gera clusters de municípios baseados na VOCAÇÃO (Áreas de Cursos ofertados).
    Retorna um DataFrame com as colunas: 'CD_MUN', 'cluster_id', 'cluster_name'.
    """
    # 1. Identificação da Coluna de Área
    # Tenta encontrar a coluna de Área do Curso (nomes variam)
    col_area = None
    possible_cols = [
        "ÁREA DO CURSO (automático)", 
        "ÁREA DO CURSO\n(automático)", 
        "ÁREA DO CURSO"
    ]
    
    for c in possible_cols:
        if c in df_input.columns:
            col_area = c
            break
            
    if col_area is None:
        # Fallback se não achar área: Cluster único ou vazio
        print("Aviso: Coluna de Área do Curso não encontrada para clusterização.")
        return pd.DataFrame(columns=["CD_MUN", "cluster_id", "cluster_name"])

    # 2. Pivot Table: Linhas=Municípios, Colunas=Áreas, Valores=Contagem
    # Preenchimento com 0 para áreas não ofertadas no município
    df_pivot = pd.crosstab(
        index=df_input["Código Município Completo"], 
        columns=df_input[col_area]
    )
    
    # Filtra colunas com muito pouca frequência global se necessário (opcional)
    # df_pivot = df_pivot.loc[:, df_pivot.sum() > 5] 

    if df_pivot.empty:
        return pd.DataFrame()

    # 3. Normalização (Percentual de cursos por área no município)
    # Para comparar "perfil" e não "volume"
    # div(axis=0) divide cada valor da linha pela soma da linha
    X_pct = df_pivot.div(df_pivot.sum(axis=1), axis=0).fillna(0)
    
    # 4. K-Means
    # Definir k. Se tivermos poucas áreas, k deve ser pequeno. 
    # Vamos tentar k = min(5, num_areas) para capturar perfis principais
    n_clusters = min(5, len(X_pct.columns))
    if len(X_pct) < n_clusters:
        n_clusters = len(X_pct) # Fallback para pouquíssimos dados
        
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(X_pct)
    
    # Adiciona cluster ao pivot
    X_pct["cluster_id"] = cluster_ids
    
    # 5. Naming (Nomear Clusters pelo Tópico Dominante)
    # Calculamos a média (centróide) de cada área dentro do cluster
    cluster_centers = X_pct.groupby("cluster_id").mean()
    
    cluster_names = {}
    used_names = set()  # Rastreia nomes já usados
    
    for cid in cluster_centers.index:
        # Pega as 2 áreas com maior média neste cluster
        top_areas = cluster_centers.loc[cid].nlargest(2)
        top_area = top_areas.index[0]
        score_1 = top_areas.iloc[0]
        
        # Se o score for baixo (ex: < 0.35), significa que é um mix (Generalista)
        if score_1 < 0.35:
            name = "Generalista / Misto"
        else:
            # Nome base
            base_name = f"Vocação: {str(top_area).title()}"
            
            # Se já existe esse nome, adiciona a segunda área
            if base_name in used_names and len(top_areas) > 1:
                second_area = top_areas.index[1]
                score_2 = top_areas.iloc[1]
                
                # Se a segunda área também é relevante (>20%), inclui no nome
                if score_2 > 0.20:
                    name = f"{str(top_area).title()} + {str(second_area).title()}"
                else:
                    # Senão, adiciona "Foco" para diferenciar
                    name = f"Foco em {str(top_area).title()}"
            else:
                name = base_name
                
        cluster_names[cid] = name
        used_names.add(name)

    # 6. Formata Saída
    output = pd.DataFrame({
        "CD_MUN": X_pct.index.astype(str).str.zfill(7),
        "cluster_id": cluster_ids,
        "cluster_name": [cluster_names[c] for c in cluster_ids]
    })
    
    return output
