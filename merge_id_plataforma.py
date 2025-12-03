import pandas as pd

# Carregar dados dos cursos


def merge_id_plataforma(cursos_df: pd.DataFrame, plataforma_df: pd.DataFrame) -> pd.DataFrame:
    """
    Função para mesclar o DataFrame de cursos com o DataFrame da plataforma
    com base na coluna 'ID Plataforma'.

    Parâmetros:
    cursos_df (pd.DataFrame): DataFrame contendo os dados dos cursos.
    plataforma_df (pd.DataFrame): DataFrame contendo os dados da plataforma.

    Retorna:
    pd.DataFrame: DataFrame resultante da mesclagem dos dois DataFrames.
    """
    
    
    cursos_df['sda_id'] = cursos_df['sda_id'].fillna(0).astype(int)
    plataforma_df = plataforma_df[['id', 'sda_id', 'name']]
    # Realiza a mesclagem dos DataFrames com base na coluna 'ID Plataforma'
    merged_df = pd.merge(cursos_df, plataforma_df, on='sda_id', how='left')
    
    merged_df.rename(columns={
        'name_y': 'plataforma_name', 
        'name_x': 'qualificacao_name'}, inplace=True)
    
    merged_df['id'] = merged_df['id'].fillna(0).astype(int)
    merged_df['Código Município Completo'] = merged_df['Código Município Completo'].fillna(0).astype(int)
    # merged_df['Código Município Completo'] = merged_df['Código Município Completo'].replace('.0', '', regex=True)
    
    return merged_df
