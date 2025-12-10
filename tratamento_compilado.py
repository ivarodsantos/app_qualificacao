import pandas as pd
import warnings

def tratamento_compilado(df_compilado: pd.DataFrame, 
                         df_lotes: pd.DataFrame, 
                         df_cozinhas_simp: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza o tratamento inicial do DataFrame compilado.

    Parâmetros:
    df_compilado (pd.DataFrame): DataFrame bruto com os dados compilados do Google Sheets.
    df_lotes (pd.DataFrame): DataFrame com informações de lotes e municípios.
    df_cozinhas_simp (pd.DataFrame): DataFrame simplificado com dados das cozinhas (deve conter 'sda_id' e 'name').

    Retorna:
    pd.DataFrame: DataFrame tratado e enriquecido com informações de lotes e cozinhas.
    
    Raises:
    ValueError: Se algum DataFrame de entrada estiver vazio ou faltar colunas obrigatórias.
    """
    
    # Validação de entradas
    if df_compilado.empty:
        raise ValueError("df_compilado não pode estar vazio")
    
    if df_lotes.empty:
        raise ValueError("df_lotes não pode estar vazio")
    
    if df_cozinhas_simp.empty:
        raise ValueError("df_cozinhas_simp não pode estar vazio")
    
    # Validar colunas obrigatórias em df_compilado
    required_cols_compilado = ['LOTE', 'CIDADE', 'ID COZINHA FOCAL']
    missing_cols = [col for col in required_cols_compilado if col not in df_compilado.columns]
    if missing_cols:
        raise ValueError(f"Colunas obrigatórias ausentes em df_compilado: {missing_cols}")
    
    # Validar colunas obrigatórias em df_lotes
    required_cols_lotes = ['Nome_Município', 'mun_upp']
    missing_cols = [col for col in required_cols_lotes if col not in df_lotes.columns]
    if missing_cols:
        raise ValueError(f"Colunas obrigatórias ausentes em df_lotes: {missing_cols}")
    
    # Validar colunas obrigatórias em df_cozinhas_simp
    if 'sda_id' not in df_cozinhas_simp.columns:
        raise ValueError("Coluna 'sda_id' ausente em df_cozinhas_simp")
    
    # Criar cópia para não modificar os DataFrames originais
    df_compilado = df_compilado.copy()
    df_lotes = df_lotes.copy()
    df_cozinhas_simp = df_cozinhas_simp.copy()
    
    # Padronizar nome da coluna LOTE para maiúsculas
    df_compilado['LOTE UPP'] = df_compilado['LOTE'].str.upper()
    
    # Corrigir nome do município Ereré para Ererê
    df_lotes['Nome_Município'] = df_lotes['Nome_Município'].replace('Ereré', 'Ererê')
    df_lotes['mun_upp'] = df_lotes['mun_upp'].replace('ERERÉ', 'ERERÊ')
    
    # Merge com df_lotes
    df_compilado_novos_lotes = pd.merge(
        df_compilado, 
        df_lotes, 
        left_on='CIDADE', 
        right_on='Nome_Município', 
        how='left'
    )
    
    # Verificar se houve registros sem match no merge
    registros_sem_match = df_compilado_novos_lotes['Nome_Município'].isna().sum()
    if registros_sem_match > 0:
        warnings.warn(
            f"{registros_sem_match} registros não encontraram correspondência em df_lotes. "
            f"Verifique se todos os municípios em CIDADE existem em Nome_Município.",
            UserWarning
        )
    
    # Adicionar região de planejamento para Fortaleza
    df_compilado_novos_lotes.loc[
        df_compilado_novos_lotes['Nome_Município'] == 'Fortaleza', 
        'REGIÃO DE PLANEJAMENTO'
    ] = 'Grande Fortaleza'
    df_compilado_novos_lotes.loc[
        df_compilado_novos_lotes['Nome_Município'] == 'Fortaleza', 
        'regiao_upp'
    ] = 'GRANDE FORTALEZA'
    
    # Converter colunas numéricas para int
    columns_to_convert = [
        'VAGAS OFERTADAS', 
        'INSCRITOS', 
        'DESISTENTES', 
        'CONCLUDENTES', 
        'NÚMERO DE ALUNOS BENEFICIÁRIOS'
    ]
    
    for col in columns_to_convert:
        if col in df_compilado_novos_lotes.columns:
            df_compilado_novos_lotes[col] = pd.to_numeric(
                df_compilado_novos_lotes[col], 
                errors='coerce'
            ).fillna(0).astype(int)
        else:
            warnings.warn(f"Coluna '{col}' não encontrada no DataFrame. Pulando conversão.", UserWarning)
    
    # Converter sda_id para string
    df_cozinhas_simp['sda_id'] = df_cozinhas_simp['sda_id'].astype(str)
    
    # Merge com df_cozinhas_simp
    df_compilado_novos_lotes_merge_nomes_cozinhas = pd.merge(
        df_compilado_novos_lotes, 
        df_cozinhas_simp, 
        left_on='ID COZINHA FOCAL', 
        right_on='sda_id', 
        how='left'
    )
    
    # Verificar se houve registros sem match no segundo merge
    registros_sem_match_cozinhas = df_compilado_novos_lotes_merge_nomes_cozinhas['sda_id'].isna().sum()
    if registros_sem_match_cozinhas > 0:
        warnings.warn(
            f"{registros_sem_match_cozinhas} registros não encontraram correspondência em df_cozinhas_simp. "
            f"Verifique se todos os IDs em 'ID COZINHA FOCAL' existem em 'sda_id'.",
            UserWarning
        )
    
    return df_compilado_novos_lotes_merge_nomes_cozinhas