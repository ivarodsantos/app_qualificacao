import pandas as pd

# Link do Google Sheets
link = "https://docs.google.com/spreadsheets/d/1M2huy5RGW5D28zWRnBiHI4kSGWZNi5ejyygnxQjx7uo/edit?gid=0#gid=0"

# Nome da aba
nome_aba = "Compilado"

# Tentar carregar dados do Google Sheets
try:
    from google_sheets_api import carregar_google_sheet_por_aba
    
    print("Carregando dados do Google Sheets...")
    df_raw = carregar_google_sheet_por_aba(link, nome_aba, "A:AN")
    
    print('='*80)
    print('ANÁLISE DA COLUNA PROJETO - DADOS DO GOOGLE SHEETS')
    print('='*80)
    
    print('\n1. COLUNAS DISPONÍVEIS:')
    print('-'*80)
    for i, col in enumerate(df_raw.columns.tolist(), 1):
        print(f"{i}. {col}")
    
    print(f'\n2. TOTAL DE REGISTROS: {len(df_raw):,}')
    
    print('\n3. VERIFICANDO COLUNA PROJETO...')
    print('-'*80)
    
    # Procurar por colunas que contenham "PROJ"
    colunas_projeto = [col for col in df_raw.columns if 'PROJ' in col.upper()]
    
    if colunas_projeto:
        print(f'[OK] Encontrada(s) {len(colunas_projeto)} coluna(s) relacionada(s) a PROJETO:')
        for col in colunas_projeto:
            print(f'\n  Coluna: "{col}"')
            print(f'  Valores unicos ({df_raw[col].nunique()}):')
            print(df_raw[col].value_counts(dropna=False).head(20))
            
            total = len(df_raw)
            nulos = df_raw[col].isna().sum()
            vazios = (df_raw[col].astype(str).str.strip() == '').sum()
            preenchidos = total - nulos - vazios
            
            print(f'\n  QUALIDADE DOS DADOS:')
            print(f'    Total: {total:,}')
            print(f'    Nulos: {nulos:,} ({(nulos/total*100):.2f}%)')
            print(f'    Vazios: {vazios:,} ({(vazios/total*100):.2f}%)')
            print(f'    Preenchidos: {preenchidos:,} ({(preenchidos/total*100):.2f}%)')
    
    else:
        print('[ERRO] Nenhuma coluna relacionada a PROJETO encontrada')
        print('\nBuscando colunas similares...')
        
        # Procurar outras possíveis colunas
        possíveis = [col for col in df_raw.columns if any(termo in col.upper() for termo in ['PROG', 'INIC', 'AÇÃO', 'ATIV'])]
        if possíveis:
            print('Colunas que podem ser relevantes:')
            for col in possíveis:
                print(f"  - {col}")
    
    print('\n' + '='*80)
    
except Exception as e:
    print(f"Erro ao carregar dados: {e}")
    print("\nTentando métod alterativo...")
