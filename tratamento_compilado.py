import pandas as pd

def tratamento_compilado(df_compilado: pd.DataFrame, 
                         df_lotes: pd.DataFrame, 
                         df_cozinhas_simp: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza o tratamento inicial do DataFrame compilado.

    Parâmetros:
    df (pd.DataFrame): DataFrame bruto com os dados compilados.

    Retorna:
    pd.DataFrame: DataFrame tratado.
    """
    
    
    df_compilado['LOTE UPP'] = df_compilado['LOTE'].str.upper()
    df_lotes['Nome_Município'] = df_lotes['Nome_Município'].replace('Ereré', 'Ererê')
    df_lotes['mun_upp'] = df_lotes['mun_upp'].replace('ERERÉ', 'ERERÊ')
    
    df_compilado_novos_lotes = pd.merge(df_compilado, df_lotes, left_on='CIDADE', right_on='Nome_Município', how='left')
    
    
    columns_to_convert = ['VAGAS OFERTADAS', 'INSCRITOS', 'DESISTENTES', 'CONCLUDENTES', 'NÚMERO DE ALUNOS BENEFICIÁRIOS']

    for col in columns_to_convert:
        df_compilado_novos_lotes[col] = pd.to_numeric(df_compilado_novos_lotes[col], errors='coerce').fillna(0).astype(int)
    
    
    df_cozinhas_simp['sda_id'] = df_cozinhas_simp['sda_id'].astype(str)
    df_compilado_novos_lotes_merge_nomes_cozinhas = pd.merge(df_compilado_novos_lotes, df_cozinhas_simp, left_on='ID COZINHA FOCAL', right_on='sda_id', how='left')
    df_compilado_novos_lotes_merge_nomes_cozinhas

    return df_compilado