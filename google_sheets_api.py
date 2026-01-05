import os
import pandas as pd

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# Import Streamlit for caching
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# Escopo de acesso: somente leitura da planilha
SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]


def get_sheets_service():
    """
    Cria e retorna um client da Google Sheets API autenticado via OAuth.
    Usa:
      - credentials.json  -> dados do app (baixado do Google Cloud)
      - token.json        -> token de acesso do usuário (criado na 1ª vez)
    """
    creds = None

    # 1. Tenta carregar dos Segredos do Streamlit (para Cloud)
    # 1. Tenta carregar dos Segredos do Streamlit (para Cloud)
    if STREAMLIT_AVAILABLE:
        try:
            # Verifica existência sem quebrar
            if "google_oauth" in st.secrets:
                creds = Credentials.from_authorized_user_info(dict(st.secrets["google_oauth"]), SCOPES)
        except Exception as e:
            # Ignora erro de secrets não encontrados e segue para método local
            pass
    
    # 2. Se não deu certo (ou é local), tenta arquivo local token.json
    if not creds and os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    # Se ainda não temos credenciais válidas, precisamos gerar/renovar
    if not creds or not creds.valid:
        # Caso as credenciais existam, mas estejam expiradas, tenta renovar
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                # Se falhar ao renovar (ex: token revogado), força nova autenticação
                creds = None

        if not creds:
            # Fluxo de autenticação baseado no arquivo credentials.json
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json",  # arquivo baixado do Google Cloud
                SCOPES               # permissões que estamos pedindo
            )
            # Abre um servidor local e o navegador para login na conta Google
            creds = flow.run_local_server(port=0)

        # Salva o token para uso futuro (não precisar logar de novo)
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    # Cria o serviço (client) da API do Google Sheets
    service = build("sheets", "v4", credentials=creds)
    return service


def extrair_id_planilha(link_planilha: str) -> str:
    """
    Recebe o link completo da planilha do Google Sheets e extrai o ID.

    Exemplo de link:
    https://docs.google.com/spreadsheets/d/ID_DA_PLANILHA/edit?gid=0#gid=0
    """
    try:
        # Quebra o link em "/d/" e pega a parte depois disso
        parte = link_planilha.split("/d/")[1]
        # Da parte restante, pegamos tudo até a próxima "/"
        planilha_id = parte.split("/")[0]
        return planilha_id
    except Exception as e:
        raise ValueError(f"Não consegui extrair o ID da planilha. Verifique o link. Erro: {e}")


# Wrapper interno para caching condicional
def _carregar_google_sheet_cached(
    link_planilha: str,
    nome_aba: str,
    intervalo: str = "A:Z",
    ttl: int = 300
) -> pd.DataFrame:
    """
    Lê dados de uma ABA específica da planilha com cache Streamlit.
    
    Parâmetros:
        link_planilha (str): link completo da planilha do Google Sheets
        nome_aba (str): nome EXATO da aba (como aparece lá embaixo no Sheets)
        intervalo (str): intervalo no formato A1 (ex.: 'A:Z', 'A1:K500', etc.)
        ttl (int): tempo de vida do cache em segundos (padrão: 300 = 5 minutos)

    Retorno:
        pd.DataFrame com os dados lidos.
    """
    # 1) Extrai o ID da planilha a partir do link
    spreadsheet_id = extrair_id_planilha(link_planilha)

    # 2) Cria o serviço autenticado
    service = get_sheets_service()

    # 3) Monta o range no formato "NomeAba!Intervalo"
    range_planilha = f"{nome_aba}!{intervalo}"

    # 4) Faz a requisição para a API
    result = (
        service.spreadsheets()
        .values()
        .get(spreadsheetId=spreadsheet_id, range=range_planilha)
        .execute()
    )

    # 5) Extrai os valores retornados
    values = result.get("values", [])

    if not values:
        print("Nenhum dado encontrado nesse intervalo/aba.")
        return pd.DataFrame()

    # 6) Assume que a primeira linha é o cabeçalho
    header = values[0]   # nomes das colunas
    rows = values[1:]    # dados

    # 7) Cria o DataFrame
    df = pd.DataFrame(rows, columns=header)
    
    # Remove linhas onde CURSO é nulo, vazio ou apenas espaços
    df_clean = df.dropna(subset=['CURSO'])  # Remove NaN
    df_clean = df_clean[df_clean['CURSO'].str.strip() != '']  # Remove strings vazias/espaços
    
    return df_clean


def carregar_google_sheet_por_aba(
    link_planilha: str,
    nome_aba: str,
    intervalo: str = "A:Z",
    ttl: int = 300,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Lê dados de uma ABA específica da planilha do Google Sheets.
    
    Parâmetros:
        link_planilha (str): link completo da planilha do Google Sheets
        nome_aba (str): nome EXATO da aba (como aparece lá embaixo no Sheets)
        intervalo (str): intervalo no formato A1 (ex.: 'A:Z', 'A1:K500', etc.)
        ttl (int): tempo de vida do cache em segundos (padrão: 300 = 5 minutos)
                  Use 0 para desabilitar cache
        use_cache (bool): se True, usa cache do Streamlit (padrão: True)

    Retorno:
        pd.DataFrame com os dados lidos.
    """
    # Se Streamlit não está disponível ou cache desabilitado, chama função sem cache
    if not STREAMLIT_AVAILABLE or not use_cache or ttl == 0:
        return _carregar_google_sheet_cached(link_planilha, nome_aba, intervalo, ttl)
    
    # Aplica cache dinamicamente usando st.cache_data
    @st.cache_data(ttl=ttl, show_spinner=False)
    def _cached_load(link: str, aba: str, inter: str, _ttl: int):
        return _carregar_google_sheet_cached(link, aba, inter, _ttl)
    
    return _cached_load(link_planilha, nome_aba, intervalo, ttl)
