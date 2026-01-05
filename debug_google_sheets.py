import pandas as pd
import sys
import os

# Adiciona o diretório atual ao path para importar os módulos locais
sys.path.append(os.getcwd())

try:
    from google_sheets_api import carregar_google_sheet_por_aba
except ImportError as e:
    print(f"Erro ao importar google_sheets_api: {e}")
    sys.exit(1)

link = "https://docs.google.com/spreadsheets/d/1M2huy5RGW5D28zWRnBiHI4kSGWZNi5ejyygnxQjx7uo/edit?gid=0#gid=0"
nome_aba = "Compilado"
intervalo = "A:AN"

print(f"Tentando carregar planilha: {link}")
print(f"Aba: {nome_aba}")
print(f"Intervalo: {intervalo}")

try:
    # Desativa o cache do Streamlit passando ttl=0 e use_cache=False (se a funcao suportar, mas o codigo mostra que suporta ttl=0 para bypass)
    # Na verdade, a assinatura é: carregar_google_sheet_por_aba(link_planilha, nome_aba, intervalo, ttl=300, use_cache=True)
    # Se ttl=0, ele chama _carregar_google_sheet_cached diretamente sem o decorador st.cache_data
    
    df = carregar_google_sheet_por_aba(link, nome_aba, intervalo, ttl=0, use_cache=False)
    
    print("\n--- Resultado ---")
    if df.empty:
        print("AVISO: O DataFrame retornado está VAZIO.")
    else:
        print(f"SUCESSO! DataFrame carregado com {len(df)} linhas e {len(df.columns)} colunas.")
        print("\nColunas encontradas:")
        print(df.columns.tolist())
        print("\nPrimeiras 5 linhas:")
        print(df.head())

except Exception as e:
    print("\n--- ERRO CAPTURADO ---")
    print(f"Tipo do erro: {type(e).__name__}")
    print(f"Mensagem de erro: {e}")
    import traceback
    traceback.print_exc()
