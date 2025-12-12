import pandas as pd
import numpy as np
import streamlit as st
from streamlit_folium import st_folium
import folium
import json
import geopandas as gpd
import branca.colormap as cm
import copy
import altair as alt
from folium.plugins import Draw
import base64

from branca.colormap import linear
from branca.element import MacroElement, Template, IFrame
import plotly.express as px
import urllib.parse as up # Import necessario para urlencode

# Configuração para evitar FutureWarnings do Pandas (Downcasting)
pd.set_option('future.no_silent_downcasting', True)
from ml_utils import gerar_clusters_municipios

@st.cache_data
def load_jornada_data():
    # Load Trilha
    try:
        trilha_df = pd.read_csv("data2/compilado_trilha_sebrae.csv", header=0)
        # Drop PII (Personal Identifiable Information)
        cols_drop = ["ALUNO", "CONTATO", "PONTO FOCAL", "ENDEREÇO", "CPF", "TELEFONE"]
        trilha_df = trilha_df[[c for c in trilha_df.columns if not any(x in c.upper() for x in cols_drop)]]
        
        # Numeric conversion
        cols_num = ["INSCRITOS TRILHA", "CONCLUDENTES TRILHA", "PESSOAS SENSIBILIZAÇÃO"]
        for col in cols_num:
            if col in trilha_df.columns:
                trilha_df[col] = pd.to_numeric(trilha_df[col], errors='coerce').fillna(0)
    except Exception as e:
        trilha_df = pd.DataFrame()
        st.error(f"Erro ao carregar Trilha: {e}")

    # Load Mentoria
    try:
        mentoria_df = pd.read_csv("data2/compilado_mentoria_sebrae.csv", header=0)
        # Drop PII
        cols_drop = ["ALUNO", "CONTATO", "PONTO FOCAL", "ENDEREÇO", "CPF", "TELEFONE"]
        mentoria_df = mentoria_df[[c for c in mentoria_df.columns if not any(x in c.upper() for x in cols_drop)]]
        
        # For Mentoria, usually 1 row = 1 student. So we count rows with Status = Concluido
    except Exception as e:
        mentoria_df = pd.DataFrame()
        st.error(f"Erro ao carregar Mentoria: {e}")
        
    return trilha_df, mentoria_df
from merge_id_plataforma import merge_id_plataforma
import acesso_planilha
from acesso_planilha import carregar_google_sheet_aba
from google_sheets_api import carregar_google_sheet_por_aba
from tratamento_compilado import tratamento_compilado

link = "https://docs.google.com/spreadsheets/d/1M2huy5RGW5D28zWRnBiHI4kSGWZNi5ejyygnxQjx7uo/edit?gid=0#gid=0"

# Coloque aqui o NOME EXATO da aba, como aparece no Google Sheets
nome_aba = "Compilado"  # exemplo; troque pelo nome real da aba
intervalo = "A:AN"       # lê todas as colunas da aba; ajuste se quiser

df = carregar_google_sheet_por_aba(link, nome_aba, intervalo)

# Configurações iniciais do Streamlit
# Configurações iniciais do Streamlit
st.set_page_config(layout="wide", page_title="Ceará Sem Fome - Qualificação")


# Função para carregar CSS externo
def load_css(file_name):
    with open(file_name, encoding="utf-8") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Carrega o CSS global
load_css("styles.css")

# Função para converter imagem para base64
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception as e:
        return ""

# Carregar imagens em base64
img_gov_b64 = get_base64_of_bin_file("icons/govce_hor_neg.png")
img_qualifica_b64 = get_base64_of_bin_file("icons/neg_fundo azul.png")

# =============================================================================
# HEADER INSTITUCIONAL (Baseado no Slide 1)
# =============================================================================
st.markdown(f"""
<div style="background: linear-gradient(135deg, var(--azul-gov) 0%, #1a4b8c 100%); padding: 2.5rem 3rem; border-radius: 0 0 20px 20px; margin-bottom: 2rem; color: white; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 20px;">
        <div style="display: flex; flex-direction: column; gap: 15px;">
            <div style="font-family: 'Space Grotesk', sans-serif; font-size: 2.2rem; font-weight: 700; line-height: 1.1; letter-spacing: -0.5px; text-transform: uppercase; border-bottom: 4px solid var(--amarelo-destaque); padding-bottom: 8px; width: fit-content;">
                Painel de Monitoramento
            </div>
            <div style="display: flex; align-items: center; gap: 15px; margin-top: 5px;">
                 <img src="data:image/png;base64,{img_qualifica_b64}" style="max-height: 90px; width: auto; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.1));" alt="+ Qualificação e Renda">
            </div>
        </div>
        <div style="display: flex; align-items: center; height: 100%; padding-left: 30px; border-left: 1px solid rgba(255, 255, 255, 0.2);">
             <img src="data:image/png;base64,{img_gov_b64}" style="max-height: 70px; width: auto; opacity: 0.95;" alt="Governo do Ceará">
        </div>
    </div>
</div>
<div class="custom-divider" style="margin-top: -1rem; margin-bottom: 2rem;"></div>
""", unsafe_allow_html=True)


#-------------- Carregamento dos dados --------------#
@st.cache_data
def load_data():
    """
    Carrega e processa todos os dados necessários para a aplicação.
    
    Returns:
        tuple: (cursos_df, qtd_beneficiarios_cozinhas_df, qtd_cozinhas_df, df_kitchen)
    """
    # Carregar dados brutos do Google Sheets (df_compilado)
    # Nota: 'df' já foi carregado globalmente na linha 65
    df_compilado = df.copy()
    
    # Carregar df_lotes (planilha de referência dos municípios)
    df_lotes = pd.read_csv(
        "data2/planilha de referência dos municipios com codigo do ibge - planilha de referência dos municipios com codigo do ibge.csv",
        encoding="utf-8",
        sep=",",
    )
    
    # Carregar df_kitchen
    df_kitchen = pd.read_csv(
        "data2/data-1762178638816_kitchen.csv"
    )
    
    # Criar df_cozinhas_simp com apenas as colunas necessárias
    df_cozinhas_simp = df_kitchen[['sda_id', 'name']].copy()
    
    # Processar os dados usando a função de tratamento
    cursos_df = tratamento_compilado(df_compilado, df_lotes, df_cozinhas_simp)
    
    # Carregar dados quantidade de beneficiários e cozinhas por lote, região e município
    qtd_beneficiarios_cozinhas_df = pd.read_csv(
        "data2/quantidade_beneficiarios_e_cozinhas_lote_regiao_municipio_03112025.csv",
        encoding="utf-8",
        sep=",",
    )
    
    # Carregar dados de quantidade de cozinhas por lote, região e município
    qtd_cozinhas_df = pd.read_csv(
        "data2/quantidade_beneficiarios_e_cozinhas_lote_regiao_municipio_03112025.csv",
        encoding="utf-8",
        sep=",",
    )
    
    return cursos_df, qtd_beneficiarios_cozinhas_df, qtd_cozinhas_df, df_kitchen


cursos_df, qtd_beneficiarios_cozinhas_df, qtd_cozinhas_df, df_kitchen = load_data()

# Substituir 'Certificado entregue' por 'Concluído'
if "STATUS" in cursos_df.columns:
    cursos_df["STATUS"] = cursos_df["STATUS"].astype(str).str.strip()
    cursos_df.loc[cursos_df["STATUS"] == "Certificado entregue", "STATUS"] = "Concluído"

# Substituir 'Maria da Hora' por 'Instituto Maria da Hora' na coluna EXECUTORA
if "EXECUTORA" in cursos_df.columns:
    cursos_df["EXECUTORA"] = cursos_df["EXECUTORA"].astype(str).str.strip()
    cursos_df.loc[cursos_df["EXECUTORA"] == "Maria da Hora", "EXECUTORA"] = "Instituto Maria da Hora"

# Normalizar colunas utilizadas nos filtros
# Remove espaços extras, padroniza NaN e garante consistência
if "Nome_Município" in cursos_df.columns:
    cursos_df["Nome_Município"] = cursos_df["Nome_Município"].astype(str).str.strip()
    cursos_df.loc[cursos_df["Nome_Município"] == "nan", "Nome_Município"] = None

if "EXECUTORA" in cursos_df.columns:
    # Já foi feito strip acima, mas garantir que não há "nan" como string
    cursos_df.loc[cursos_df["EXECUTORA"] == "nan", "EXECUTORA"] = None

if "CURSO" in cursos_df.columns:
    cursos_df["CURSO"] = cursos_df["CURSO"].astype(str).str.strip()
    cursos_df.loc[cursos_df["CURSO"] == "nan", "CURSO"] = None

if "ÁREA DO CURSO\n(automático)" in cursos_df.columns:
    cursos_df["ÁREA DO CURSO\n(automático)"] = (
        cursos_df["ÁREA DO CURSO\n(automático)"]
        .astype(str)
        .str.strip()
    )
    cursos_df.loc[cursos_df["ÁREA DO CURSO\n(automático)"] == "nan", "ÁREA DO CURSO\n(automático)"] = None

if "PROJETO" in cursos_df.columns:
    cursos_df["PROJETO"] = cursos_df["PROJETO"].astype(str).str.strip()
    cursos_df.loc[cursos_df["PROJETO"] == "nan", "PROJETO"] = None

if "PARCEIRO" in cursos_df.columns:
    cursos_df["PARCEIRO"] = cursos_df["PARCEIRO"].astype(str).str.strip().str.upper()
    cursos_df.loc[cursos_df["PARCEIRO"] == "NAN", "PARCEIRO"] = None

# Filtro de status indesejados solicitado pelo usuário
status_indesejados = ["Turma Cancelada", "Não iniciado", "Adiado"]
if "STATUS" in cursos_df.columns:
    # Remove espaços em branco extras para garantir o match
    cursos_df["STATUS"] = cursos_df["STATUS"].astype(str).str.strip()
    cursos_df = cursos_df[~cursos_df["STATUS"].isin(status_indesejados)]



geojsons = [
    "data/municipios_latlon.geojson",
    "data2/cozinhas_geo_ipece_01122025_simplificado.geojson",
    "data2/municipios_com_qualificacao_simplificado.geojson",
]

@st.cache_data
def load_geojson_files(file_list):
    """Carrega vários arquivos GeoJSON e retorna uma lista de objetos Python."""
    geojsons = []  # lista onde vamos guardar cada geojson carregado

    for file_path in file_list:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)   # carrega o arquivo geojson
            geojsons.append(data) # adiciona à lista

    return geojsons
    
municipios_geojson, cozinhas_geojson, municipios_com_qualificacao = load_geojson_files(geojsons)

# ícones personalizados para o mapa
cozinha_focal_icon = folium.CustomIcon(
    icon_image="icons/icone_csf_4.png",
    icon_size=(35, 42),
)
cozinha_csf_icon = folium.CustomIcon(
    icon_image="icons/icone_csf_1.png",
    icon_size=(35, 42),
)

#----------------------------------------------------#




#-------------- Pré processamento dos dados --------------#


# Mesclar dados dos cursos com dados da plataforma
merged_df = merge_id_plataforma(cursos_df, df_kitchen)


# -------------------------------------------------------------
# Cálculo da taxa relativa de conclusão (GLOBAL)
# -------------------------------------------------------------
if "VAGAS OFERTADAS" in merged_df.columns and "CONCLUDENTES" in merged_df.columns:
    merged_df["TAXA_CONCLUSAO"] = (
        merged_df["CONCLUDENTES"] / merged_df["VAGAS OFERTADAS"]
    ) * 100
    # Evitar valores >100% ou negativos
    merged_df["TAXA_CONCLUSAO"] = merged_df["TAXA_CONCLUSAO"].clip(0, 100)
    merged_df["TAXA_CONCLUSAO"] = merged_df["TAXA_CONCLUSAO"].fillna(0)
    
# Quantidade de cursos concluidos
# merged_df[merged_df['STATUS'] == "Concluído"]['CURSO'].count()

merged_df_agg = merged_df.groupby(
    ["Código Município Completo", "Nome_Município", 'Nº LOTE 2025']
).agg(
    total_turmas=pd.NamedAgg(column="CURSO", aggfunc="count"),
    total_vagas_ofertadas=pd.NamedAgg(column="VAGAS OFERTADAS", aggfunc="sum"),
    total_inscritos=pd.NamedAgg(column="INSCRITOS", aggfunc="sum"),
    total_desistentes=pd.NamedAgg(column="DESISTENTES", aggfunc="sum"),
    total_concludentes=pd.NamedAgg(column="CONCLUDENTES", aggfunc="sum"),
    percentual_conclusao=pd.NamedAgg(
        column="CONCLUDENTES", 
        aggfunc=lambda x: round((x.sum() / merged_df.loc[x.index, "VAGAS OFERTADAS"].sum()) * 100, 2) 
        if merged_df.loc[x.index, "VAGAS OFERTADAS"].sum() > 0 else 0
    ),
).reset_index()

total_geral_turmas = merged_df_agg["total_turmas"].sum()
total_geral_vagas = merged_df_agg["total_vagas_ofertadas"].sum()
total_geral_inscritos = merged_df_agg["total_inscritos"].sum()
total_geral_desistentes = merged_df_agg["total_desistentes"].sum()
total_geral_concludentes = merged_df_agg["total_concludentes"].sum()
percentual_geral_conclusao = round(
    (total_geral_concludentes / total_geral_inscritos) * 100, 2
) if total_geral_inscritos > 0 else 0



# ----------------- Enriquecer GeoJSON com indicadores (DINÂMICO) ----------------- #

@st.cache_data
def get_base_geodataframe(geojson_data):
    """Converte o GeoJSON bruto em GeoDataFrame para facilitar o merge."""
    gdf = gpd.GeoDataFrame.from_features(geojson_data["features"])
    # Padronizar CD_MUN
    if "CD_MUN" in gdf.columns:
         gdf["CD_MUN"] = gdf["CD_MUN"].astype(str).str.zfill(7)
    return gdf

# Carrega o GDF base (agora cached)
gdf_base_municipios = get_base_geodataframe(municipios_com_qualificacao)

def preparar_dados_mapa(df_input, gdf_base):
    """
    Agrega o dataframe filtrado por município e faz o merge com o GeoDataFrame.
    Retorna o GeoJSON pronto para o Folium.
    """
    if df_input.empty:
        # Se não há dados, retorna o mapa "vazio" mas com as geometrias
        # Podemos adicionar colunas zeradas para não quebrar o tooltip
        gdf_vazio = gdf_base.copy()
        cols_metricas = ["total_turmas", "total_concludentes", "percentual_conclusao", "has_qualif"]
        for c in cols_metricas:
            gdf_vazio[c] = 0
        return json.loads(gdf_vazio.to_json())

    # 1. Agrega métricas por município baseado no DF FILTRADO
    df_agg = df_input.groupby(
        ["Código Município Completo", "Nome_Município"]
    ).agg(
        total_turmas=pd.NamedAgg(column="CURSO", aggfunc="count"),
        total_vagas=pd.NamedAgg(column="VAGAS OFERTADAS", aggfunc="sum"),
        total_concludentes=pd.NamedAgg(column="CONCLUDENTES", aggfunc="sum"),
        percentual_conclusao=pd.NamedAgg(
            column="CONCLUDENTES", 
            # aggfunc pode ser complexo, melhor calcular pós-agg se for soma/soma
            aggfunc="sum" 
        ),
        # Nota: percentual_conclusao aqui ficou como soma de concludentes temporariamente
        # Ajustaremos abaixo
    ).reset_index()

    # Recalcula percentual correto (Soma Concludentes / Soma Vagas) por município
    # Precisa recalcular vagas também? Sim, total_vagas está ali.
    df_agg["percentual_conclusao"] = df_agg.apply(
        lambda row: round((row["total_concludentes"] / row["total_vagas"] * 100), 2) 
        if row["total_vagas"] > 0 else 0, 
        axis=1
    )

    # Prepara coluna de junção
    df_agg["CD_MUN"] = df_agg["Código Município Completo"].astype(str).str.zfill(7)
    df_agg["has_qualif"] = 1 # Se está no agg, tem qualificação no filtro atual

    # 2. Merge com GeoDataFrame base
    # Left join no GDF para manter geometria de quem não tem curso (opcional: inner se quiser sumir)
    
    # Remove colunas que podem dar conflito se já existirem no GeoJSON base
    cols_para_remover = ["total_turmas", "total_concludentes", "percentual_conclusao", "has_qualif"]
    gdf_base_clean = gdf_base.drop(columns=[c for c in cols_para_remover if c in gdf_base.columns])
    
    gdf_merged = gdf_base_clean.merge(
        df_agg[["CD_MUN", "total_turmas", "total_concludentes", "percentual_conclusao", "has_qualif"]],
        on="CD_MUN",
        how="left"
    )
    
    # Preenche NaNs resultantes do merge (municípios sem cursos no filtro atual)
    gdf_merged["total_turmas"] = gdf_merged["total_turmas"].fillna(0)
    gdf_merged["total_concludentes"] = gdf_merged["total_concludentes"].fillna(0)
    gdf_merged["percentual_conclusao"] = gdf_merged["percentual_conclusao"].fillna(0)
    gdf_merged["has_qualif"] = gdf_merged["has_qualif"].fillna(0)
    
    # 3. Simplifica (opcional, já pode estar simplificado no source, mas bom garantir se for pesado)
    # gdf_merged["geometry"] = gdf_merged["geometry"].simplify(0.01)

    return json.loads(gdf_merged.to_json())


# ---------------- Filtros sincronizados ---------------- #


# ---------------- Filtros sincronizados ---------------- #

base_df = merged_df.copy()

col_area = "ÁREA DO CURSO\n(automático)"

@st.cache_data(show_spinner=False)
def filtrar_dados(df, municipios, executoras, cursos, areas, projetos, parceiros, taxa_range):
    """Filtra o DataFrame com base nas listas de opções selecionadas e range de taxa."""
    df_result = df.copy()
    if municipios:
        df_result = df_result[df_result["Nome_Município"].isin(municipios)]
    if executoras:
        df_result = df_result[df_result["EXECUTORA"].isin(executoras)]
    if cursos:
        df_result = df_result[df_result["CURSO"].isin(cursos)]
    if areas:
        df_result = df_result[df_result[col_area].isin(areas)]
    if projetos:
        df_result = df_result[df_result["PROJETO"].isin(projetos)]
    if parceiros:
        df_result = df_result[df_result["PARCEIRO"].isin(parceiros)]
    
    # Filtro por faixa de conclusão
    if taxa_range:
        min_t, max_t = taxa_range
        if "TAXA_CONCLUSAO" in df_result.columns:
            df_result = df_result[
                (df_result["TAXA_CONCLUSAO"] >= min_t) & 
                (df_result["TAXA_CONCLUSAO"] <= max_t)
            ]
            
    return df_result

# 1) Ler seleções atuais do session_state (antes de criar os widgets)
sel_mun_prev   = st.session_state.get("f_mun", [])
sel_exec_prev  = st.session_state.get("f_exec", [])
sel_curso_prev = st.session_state.get("f_curso", [])
sel_area_prev  = st.session_state.get("f_area", [])
sel_projeto_prev = st.session_state.get("f_projeto", [])
sel_parceiro_prev = st.session_state.get("f_parceiro", [])
sel_taxa_prev  = st.session_state.get("f_taxa", (0, 100))

# 2) Montar df_opcoes "Context-Aware"
# Para cada widget, filtramos pelos OUTROS critérios, mas ignoramos a seleção DO PRÓPRIO widget.
# Isso evita que as opções desapareçam ao selecionar um item.

# Verifica se TODOS os filtros estão vazios (estado inicial)
todos_vazios = (
    not sel_mun_prev and 
    not sel_exec_prev and 
    not sel_curso_prev and 
    not sel_area_prev and 
    not sel_projeto_prev and 
    not sel_parceiro_prev and 
    sel_taxa_prev == (0, 100)
)

# Se todos estão vazios, usa base_df diretamente (evita dependências circulares)
if todos_vazios:
    mun_options = sorted(base_df["Nome_Município"].dropna().unique().tolist())
    exec_options = sorted(base_df["EXECUTORA"].dropna().unique().tolist())
    curso_options = sorted(base_df["CURSO"].dropna().unique().tolist())
    area_options = sorted(base_df[col_area].dropna().unique().tolist())
    projeto_options = sorted(base_df["PROJETO"].dropna().unique().tolist())
    parceiro_options = sorted(base_df["PARCEIRO"].dropna().unique().tolist())
else:
    # Opções de Municípios (respeita exec, curso, area, projeto, parceiro, taxa; ignora mun)
    df_mun_ops = filtrar_dados(base_df, [], sel_exec_prev, sel_curso_prev, sel_area_prev, sel_projeto_prev, sel_parceiro_prev, sel_taxa_prev)
    mun_options = sorted(df_mun_ops["Nome_Município"].dropna().unique().tolist())
    
    # Opções de Executoras (respeita mun, curso, area, projeto, parceiro, taxa; ignora exec)
    df_exec_ops = filtrar_dados(base_df, sel_mun_prev, [], sel_curso_prev, sel_area_prev, sel_projeto_prev, sel_parceiro_prev, sel_taxa_prev)
    exec_options = sorted(df_exec_ops["EXECUTORA"].dropna().unique().tolist())
    
    # Opções de Cursos (respeita mun, exec, area, projeto, parceiro, taxa; ignora curso)
    df_curso_ops = filtrar_dados(base_df, sel_mun_prev, sel_exec_prev, [], sel_area_prev, sel_projeto_prev, sel_parceiro_prev, sel_taxa_prev)
    curso_options = sorted(df_curso_ops["CURSO"].dropna().unique().tolist())
    
    # Opções de Áreas (respeita mun, exec, curso, projeto, parceiro, taxa; ignora area)
    df_area_ops = filtrar_dados(base_df, sel_mun_prev, sel_exec_prev, sel_curso_prev, [], sel_projeto_prev, sel_parceiro_prev, sel_taxa_prev)
    area_options = sorted(df_area_ops[col_area].dropna().unique().tolist())
    
    # Opções de Projetos (respeita mun, exec, curso, area, parceiro, taxa; ignora projeto)
    df_projeto_ops = filtrar_dados(base_df, sel_mun_prev, sel_exec_prev, sel_curso_prev, sel_area_prev, [], sel_parceiro_prev, sel_taxa_prev)
    projeto_options = sorted(df_projeto_ops["PROJETO"].dropna().unique().tolist())
    
    # Opções de Parceiros (respeita mun, exec, curso, area, projeto, taxa; ignora parceiro)
    df_parceiro_ops = filtrar_dados(base_df, sel_mun_prev, sel_exec_prev, sel_curso_prev, sel_area_prev, sel_projeto_prev, [], sel_taxa_prev)
    parceiro_options = sorted(df_parceiro_ops["PARCEIRO"].dropna().unique().tolist())

#-------------- Layout do Streamlit --------------#
st.sidebar.image('icons/neg_color.png', use_container_width=True)
st.sidebar.header("Filtros de Análise")

# Botão para limpar todos os filtros
if st.sidebar.button("🔄 Limpar Todos os Filtros", use_container_width=True):
    # Reseta todas as chaves de filtro no session state
    for key in ["f_mun", "f_exec", "f_curso", "f_area", "f_projeto", "f_parceiro", "f_taxa"]:
        if key in st.session_state:
            if key == "f_taxa":
                st.session_state[key] = (0, 100)
            else:
                st.session_state[key] = []
    st.rerun()

# Botão para forçar refresh dos dados (limpar cache)
if st.sidebar.button("🔄 Atualizar Dados", use_container_width=True, help="Limpa o cache e recarrega os dados do Google Sheets"):
    st.cache_data.clear()
    st.success("✅ Cache limpo! Recarregando dados...")
    st.rerun()

st.sidebar.markdown("---")

# 4) Widgets usando essas opções + default mantendo seleções válidas

selected_municipios = st.sidebar.multiselect(
    "Selecione os municípios:",
    options=mun_options,
    default=[v for v in sel_mun_prev if v in mun_options],
    key="f_mun",
)

selected_executoras = st.sidebar.multiselect(
    "Selecione as entidades executoras:",
    options=exec_options,
    default=[v for v in sel_exec_prev if v in exec_options],
    key="f_exec",
)

selected_cursos = st.sidebar.multiselect(
    "Selecione os cursos:",
    options=curso_options,
    default=[v for v in sel_curso_prev if v in curso_options],
    key="f_curso",
)


selected_areas_qualificacao = st.sidebar.multiselect(
    "Selecione as áreas de qualificação:",
    options=area_options,
    default=[v for v in sel_area_prev if v in area_options],
    key="f_area",
)

selected_projetos = st.sidebar.multiselect(
    "Selecione os projetos:",
    options=projeto_options,
    default=[v for v in sel_projeto_prev if v in projeto_options],
    key="f_projeto",
)

selected_parceiros = st.sidebar.multiselect(
    "Selecione os parceiros:",
    options=parceiro_options,
    default=[v for v in sel_parceiro_prev if v in parceiro_options],
    key="f_parceiro",
)

# Slider de Taxa de Conclusão
st.sidebar.markdown("<br/>", unsafe_allow_html=True)
selected_taxa_range = st.sidebar.slider(
    "Faixa de Taxa de Conclusão (%):",
    min_value=0,
    max_value=100,
    value=st.session_state.get("f_taxa", (0, 100)),
    key="f_taxa"
)

# 5) Verifica se algum filtro foi aplicado
algum_filtro_ativo = any([
    selected_municipios,
    selected_executoras,
    selected_cursos,
    selected_areas_qualificacao,
    selected_projetos,
    selected_parceiros,
    selected_taxa_range != (0, 100)
])

# 6) Aplica as seleções ATUAIS a toda a base para montar df_filtrado


st.sidebar.markdown("---")
usar_ia = st.sidebar.checkbox("🧠 Análise de Inteligência Artificial", value=False, help="Ativa a clusterização automática de municípios baseada em desempenho.")
    
if algum_filtro_ativo:
    df_filtrado = filtrar_dados(
        base_df, 
        selected_municipios, 
        selected_executoras, 
        selected_cursos, 
        selected_areas_qualificacao,
        selected_projetos,
        selected_parceiros,
        selected_taxa_range
    )
else:
    df_filtrado = base_df.copy()
#----------------------------------------------------#



# -------------------------------------------------------------
# Cálculo da taxa relativa de conclusão
# -------------------------------------------------------------
# (Previamente calculado globalmente para permitir filtro)
if "TAXA_CONCLUSAO" not in df_filtrado.columns:
     st.warning("Coluna TAXA_CONCLUSAO não calculada.")



    
# -------------------------------------------------------------------
# Tratamento de datas e filtro temporal
# -------------------------------------------------------------------

# Garante que as colunas de data existem antes de mexer nelas
col_data_inicio = "DATA INÍCIO"
col_data_termino = "DATA TÉRMINO"

if col_data_inicio in df_filtrado.columns and col_data_termino in df_filtrado.columns:
    # Converter para datetime (uma vez por rerun) e remover hora (normalize)
    df_filtrado[col_data_inicio] = pd.to_datetime(
        df_filtrado[col_data_inicio],
        dayfirst=True,
        errors="coerce",
    ).dt.normalize()
    
    df_filtrado[col_data_termino] = pd.to_datetime(
        df_filtrado[col_data_termino],
        dayfirst=True,
        errors="coerce",
    ).dt.normalize()

    # Criar campos derivados
    df_filtrado["DURACAO_DIAS"] = (
        df_filtrado[col_data_termino] - df_filtrado[col_data_inicio]
    ).dt.days

    df_filtrado["ANO_INICIO"] = df_filtrado[col_data_inicio].dt.year
    df_filtrado["MES_INICIO"] = df_filtrado[col_data_inicio].dt.month
    df_filtrado["ANO_MES_INICIO"] = df_filtrado[col_data_inicio].dt.to_period("M")

    # ------------------- Filtro temporal (Data Início / Data Fim) na sidebar -------------------
    datas_validas = df_filtrado[col_data_inicio].dropna()

    if not datas_validas.empty:
        # Definir limites baseados nos dados
        min_dataset = datas_validas.min().date()
        max_dataset = datas_validas.max().date()
        
        st.sidebar.markdown("**Período de Início dos Cursos:**")
        c_data1, c_data2 = st.sidebar.columns(2)
        
        with c_data1:
            data_inicial = st.date_input(
                "De:",
                value=min_dataset,
                min_value=min_dataset,
                max_value=max_dataset,
                format="DD/MM/YYYY"
            )
            
        with c_data2:
            data_final = st.date_input(
                "Até:",
                value=max_dataset,
                min_value=min_dataset, # Permite selecionar desde o inicio
                max_value=max_dataset,
                format="DD/MM/YYYY"
            )
            
        # Converter para Timestamp normalizado (00:00:00) para comparação
        ts_inicial = pd.Timestamp(data_inicial).normalize()
        ts_final = pd.Timestamp(data_final).normalize()
        
        # Validar se o usuário inverteu as datas (UX: corrige ou avisa? melhor filtrar direito)
        if ts_inicial > ts_final:
            st.sidebar.error("Data inicial maior que a final!")
            # Fallback seguro: não filtra nada ou inverte? 
            # Vamos travar o filtro para não retornar nada ou retornar vazio
            mascara_periodo = [False] * len(df_filtrado)
        else:
            mascara_periodo = (
                (df_filtrado[col_data_inicio] >= ts_inicial) & 
                (df_filtrado[col_data_inicio] <= ts_final)
            )

        # Checkbox para incluir sem data
        incluir_sem_data_chk = st.sidebar.checkbox(
            "Incluir cursos sem data?",
            value=True,
            help="Mantém cursos sem data de início cadastrada."
        )

        if incluir_sem_data_chk:
             mascara_final = mascara_periodo | df_filtrado[col_data_inicio].isna()
        else:
             mascara_final = mascara_periodo
             
        df_filtrado = df_filtrado[mascara_final]
else:
    st.warning(
        "Colunas de data não encontradas no df_filtrado. "
        "Verifique os nomes das colunas de DATA INÍCIO e DATA TÉRMINO."
    )

    
    

tab1, tab2 = st.tabs(['Qualificação Profissional', 'Jornada Empreendedora'])

with tab1:
    # (Bloco de conversão redundante removido pois já foi tratado acima)
    
        
        
    #-------------- Criar métricas a partir dos municípios --------------#
    df_metrics = df_filtrado.groupby(
        ["Código Município Completo", "Nome_Município", 'Nº LOTE 2025']
    ).agg(
        total_turmas=pd.NamedAgg(column="CURSO", aggfunc="count"),
        total_vagas_ofertadas=pd.NamedAgg(column="VAGAS OFERTADAS", aggfunc="sum"),
        total_inscritos=pd.NamedAgg(column="INSCRITOS", aggfunc="sum"),
        total_desistentes=pd.NamedAgg(column="DESISTENTES", aggfunc="sum"),
        total_concludentes=pd.NamedAgg(column="CONCLUDENTES", aggfunc="sum"),
        percentual_conclusao=pd.NamedAgg(
            column="CONCLUDENTES",
            aggfunc=lambda x: round(
                (x.sum() / df_filtrado.loc[x.index, "VAGAS OFERTADAS"].sum()) * 100, 2
            ) if df_filtrado.loc[x.index, "VAGAS OFERTADAS"].sum() > 0 else 0
        ),
    ).reset_index()
    
    
    # Cálculo direto do df_filtrado para evitar perda de dados por NaN no groupby
    total_geral_turmas = df_filtrado.shape[0] # Considera cada linha uma turma
    total_geral_vagas = df_filtrado["VAGAS OFERTADAS"].sum()
    total_geral_inscritos = df_filtrado["INSCRITOS"].sum()
    total_geral_desistentes = df_filtrado["DESISTENTES"].sum()
    total_geral_concludentes = df_filtrado["CONCLUDENTES"].sum()
    
    # Cálculo direto (sem limitação, conforme solicitado pelo usuário)
    concludentes_ajustados = total_geral_concludentes
    
    percentual_geral_conclusao = round(
        (concludentes_ajustados / total_geral_vagas) * 100, 2
    ) if total_geral_vagas > 0 else 0
    
    # ============ EXPORTAR RELATÓRIO PDF ============
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📄 Exportar Relatório")
    
    # Botão para gerar PDF
    if st.sidebar.button("📥 Gerar Relatório PDF", use_container_width=True):
        try:
            from gerar_relatorio_pdf import gerar_relatorio
            
            # Calcular taxa de desistência (sem usar INSCRITOS)
            taxa_desistencia = round(
                (total_geral_desistentes / total_geral_vagas) * 100, 2
            ) if total_geral_vagas > 0 else 0
            
            # Preparar KPIs estendidos (sem INSCRITOS conforme solicitado)
            kpis = {
                'total_turmas': total_geral_turmas,
                'total_vagas': int(total_geral_vagas),
                'total_concludentes': int(total_geral_concludentes),
                'total_desistentes': int(total_geral_desistentes),
                'taxa_conclusao': percentual_geral_conclusao,
                'taxa_desistencia': taxa_desistencia
            }
            
            # Preparar informações de status operacional
            status_info = {
                'em_execucao': turmas_em_execucao if 'turmas_em_execucao' in dir() else 0,
                'concluidas': turmas_concluidas if 'turmas_concluidas' in dir() else 0,
                'num_municipios': df_filtrado['Nome_Município'].nunique() if 'Nome_Município' in df_filtrado.columns else 0,
                'num_executoras': df_filtrado['EXECUTORA'].nunique() if 'EXECUTORA' in df_filtrado.columns else 0
            }
            
            # Preparar informações dos filtros
            filtros_info = {}
            if selected_municipios:
                filtros_info['Municípios'] = selected_municipios
            if selected_executoras:
                filtros_info['Executoras'] = selected_executoras
            if selected_cursos:
                filtros_info['Cursos'] = selected_cursos
            if selected_areas_qualificacao:
                filtros_info['Áreas'] = selected_areas_qualificacao
            if selected_projetos:
                filtros_info['Projetos'] = selected_projetos
            if selected_parceiros:
                filtros_info['Parceiros'] = selected_parceiros
            if selected_taxa_range != (0, 100):
                filtros_info['Taxa de Conclusão'] = f"{selected_taxa_range[0]}% - {selected_taxa_range[1]}%"
            
            # Carregar e processar dados da Jornada Empreendedora
            jornada_dados = None
            try:
                df_trilha_report, df_mentoria_report = load_jornada_data()
                if not df_trilha_report.empty or not df_mentoria_report.empty:
                    jornada_dados = compute_jornada_metrics(
                        df_filtrado, 
                        df_trilha_report, 
                        df_mentoria_report, 
                        selected_municipios
                    )
            except Exception as e:
                st.sidebar.warning(f"⚠️ Dados da Jornada não disponíveis: {str(e)}")
            
            # Gerar PDF com todos os novos parâmetros
            pdf_bytes = gerar_relatorio(
                df_filtrado, 
                kpis, 
                filtros_info,
                jornada_dados=jornada_dados,
                status_info=status_info
            )
            
            # Criar nome do arquivo com data/hora
            from datetime import datetime
            nome_arquivo = f"relatorio_csf_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            
            # Botão de download
            st.sidebar.download_button(
                label="⬇️ Baixar PDF",
                data=pdf_bytes,
                file_name=nome_arquivo,
                mime="application/pdf",
                use_container_width=True
            )
            st.sidebar.success("✅ Relatório gerado com sucesso!")
            
        except ImportError as e:
            st.sidebar.error("❌ Biblioteca fpdf2 não encontrada. Instale com: pip install fpdf2")
        except Exception as e:
            st.sidebar.error(f"❌ Erro ao gerar relatório: {str(e)}")

    
    st.sidebar.markdown("---")
    
    # ================================================
    
    # Métricas de Status
    # Robustez para encoding: verificar startsWith ou contains se necessário, 
    # mas assumindo utf-8 correto:
    turmas_em_execucao = df_filtrado[
        df_filtrado["STATUS"].astype(str).str.strip() == "Em execução"
    ].shape[0]
    
    turmas_concluidas = df_filtrado[
        df_filtrado["STATUS"].astype(str).str.strip() == "Concluído"
    ].shape[0]
    
    
    df_filtrado.rename(
        columns={"ÁREA DO CURSO\n(automático)": "ÁREA DO CURSO (automático)"},
        inplace=True
    )
    
    
    df_filtrado["ÁREA DO CURSO (automático)"] = (
        df_filtrado["ÁREA DO CURSO (automático)"]
        .str.strip()
        .str.upper()
    )
    
    #----------------------------------------------------#
    
    #-------------- Criar métricas a partir das executoras --------------#
    df_metrics_exec = df_filtrado.groupby(
        ["EXECUTORA"]
    ).agg(
        total_turmas=pd.NamedAgg(column="CURSO", aggfunc="count"),
        total_vagas_ofertadas=pd.NamedAgg(column="VAGAS OFERTADAS", aggfunc="sum"),
        total_inscritos=pd.NamedAgg(column="INSCRITOS", aggfunc="sum"),
        total_desistentes=pd.NamedAgg(column="DESISTENTES", aggfunc="sum"),
        total_concludentes=pd.NamedAgg(column="CONCLUDENTES", aggfunc="sum"),
        percentual_conclusao=pd.NamedAgg(
            column="CONCLUDENTES",
            aggfunc=lambda x: round(
                (x.sum() / df_filtrado.loc[x.index, "VAGAS OFERTADAS"].sum()) * 100, 2
            ) if df_filtrado.loc[x.index, "VAGAS OFERTADAS"].sum() > 0 else 0
        ),
    ).reset_index()
    
    total_geral_turmas_exec = df_metrics_exec["total_turmas"].sum()
    total_geral_vagas_exec = df_metrics_exec["total_vagas_ofertadas"].sum()
    total_geral_inscritos_exec = df_metrics_exec["total_inscritos"].sum()
    total_geral_desistentes_exec = df_metrics_exec["total_desistentes"].sum()
    total_geral_concludentes_exec = df_metrics_exec["total_concludentes"].sum()
    
    # Cálculo seguro também para as métricas agregadas por executora
    # Nota: O df_metrics_exec já está agregado, então para ser preciso precisaríamos ajustar na agragação. 
    # Mas como estamos olhando para os totais gerais aqui:
    percentual_geral_conclusao_exec = percentual_geral_conclusao # Reutiliza a métrica global correta
    # Ou se quisermos manter independente (mas consistente):
    # percentual_geral_conclusao_exec = round(
    #     (concludentes_ajustados / total_geral_inscritos) * 100, 2
    # ) if total_geral_inscritos > 0 else 0
    #----------------------------------------------------#
    
        
        
    # filtrar o geojson das cozinhas para manter apenas as cozinhas presentes no dataframe mesclado (cozinhas focais)
    if algum_filtro_ativo:
        df_ids = df_filtrado
    else:
        df_ids = base_df  # estado original
    
    codigos_cozinhas_focais = set(
        df_ids["id"].dropna().astype(int).astype(str)
    )
    
    features_focais_filtradas = []
    for feature in cozinhas_geojson['features']:
        feature_id = feature['properties'].get('ID0')
        if feature_id and str(feature_id) in codigos_cozinhas_focais:
            features_focais_filtradas.append(feature)
    
    geojson_cozinhas_focais_filtrado = {
        'type': cozinhas_geojson['type'],
        'name': cozinhas_geojson['name'] + '_focais_filtrado',
        'crs': cozinhas_geojson['crs'],
        'features': features_focais_filtradas
    }
    
    # filtrar cozinhas CSF pelos municípios que sobraram no df_filtrado
    if algum_filtro_ativo:
        mun_filtrados = df_filtrado["mun_upp"].dropna().unique().tolist()
    else:
        mun_filtrados = []
    
    features_csf_filtradas = []
    for feature in cozinhas_geojson['features']:
        mun_feat = feature["properties"].get("MUNICIPI9")
        if not mun_filtrados or mun_feat in mun_filtrados:
            features_csf_filtradas.append(feature)
    
    geojson_cozinhas_csf_filtrado = {
        'type': cozinhas_geojson['type'],
        'name': cozinhas_geojson['name'] + '_csf_filtrado',
        'crs': cozinhas_geojson['crs'],
        'features': features_csf_filtradas
    }
    
    
    
    # verificar se algum filtro está ativo para filtrar os municípios no mapa
    # e retornar ao estado original se não houver filtro
    if algum_filtro_ativo:
        mun_filtrados = df_filtrado["Nome_Município"].dropna().unique().tolist()
    else:
        mun_filtrados = []
    
    # função para filtrar municípios no geojson
    def filtra_municipios_geojson(geojson, lista_mun):
        if not lista_mun:
            return geojson # retorna o geojson original se a lista estiver vazia
        
        # Filtra as feições baseadas na propriedade 'has_qualif' (mais seguro que nome)
        # Se 'has_qualif' não existir (caso de erro), faz fallback para o nome
        feats = []
        for f in geojson["features"]:
            props = f.get("properties", {})
            # Se temos a marcação de qualificação (vinda do merge por ID), usamos ela
            if "has_qualif" in props:
                if props["has_qualif"] == 1:
                    feats.append(f)
            # Fallback: se não tiver a propriedade, tenta filtrar por nome (legado)
            elif props.get("NM_MUN") in lista_mun:
                feats.append(f)
                
        return { # retorna o geojson filtrado
            "type": geojson["type"],
            "name": geojson.get("name", "") + "_filtrado",
            "crs": geojson.get("crs"),
            "features": feats, # lista de feições filtradas
        }
    
    # aplica a filtragem aos geojsons de municípios com qualificação e choropleth
    # Recalcula o GeoJSON Dinâmico base nos filtros
    geo_data_filtrada = preparar_dados_mapa(df_filtrado, gdf_base_municipios)
    
    # Filtro visual (se quisermos esconder as geometrias que não tem cursos)
    # O código original usava "filtra_municipios_geojson" baseado em nomes.
    # Como agora temos "has_qualif" dinâmico, podemos usar isso se quisermos.
    # Mas para consistência com o pedido do usuário (filtrar o mapa), vamos manter a lógica de SOMENTE mostrar municípios do filtro?
    # O código anterior filtrava geometry...
    # Vamos aplicar a função filtra_municipios_geojson no resultado dinâmico.
    
    # Ajuste: A função filtra_municipios_geojson trabalha com properties['NM_MUN']. Precisa gartantir que o merge manteve essa coluna.
    # (GPD merge maintaint columns of left df by default).
    
    municipios_com_qualificacao_filtrado = filtra_municipios_geojson(
        geo_data_filtrada, 
        mun_filtrados,
    )
    
    municipios_choropleth_filtrado = filtra_municipios_geojson(
        geo_data_filtrada,
        mun_filtrados,
    )

    # ---------------- Lógica de Clusterização (IA) ---------------- #
    dict_cluster_cores = {} # Para legendas ou mapa
    
    if usar_ia:
        # Gera clusters com base no DF completo (Global) para manter perfis estáveis
        df_clusters = gerar_clusters_municipios(base_df)
        
        if not df_clusters.empty:
            # Merge clusters no GeoJSON existente
            # O GeoJSON já está em 'municipios_choropleth_filtrado' (dict)
            # Vamos iterar e adicionar as propriedades
            
            # Cria dicionário para busca rápida: CD_MUN -> {cluster_id, cluster_name}
            dict_clusters = df_clusters.set_index("CD_MUN")[["cluster_id", "cluster_name"]].to_dict(orient="index")
            
            # Cores para os clusters (0 a 4, suporta até 5)
            # Cores categóricas distintas
            dict_cluster_cores = {
                0: "#d95f02", # Laranja escuro
                1: "#7570b3", # Roxo
                2: "#e7298a", # Rosa choque
                3: "#66a61e", # Verde
                4: "#1b9e77"  # Turquesa
            }
            
            # Mapeia cluster_id -> cluster_name para a legenda
            dict_cluster_names = df_clusters.drop_duplicates(subset=["cluster_id"])[["cluster_id", "cluster_name"]].set_index("cluster_id")["cluster_name"].to_dict()
            
            for feat in municipios_choropleth_filtrado["features"]:
                props = feat.get("properties", {})
                cd_mun = props.get("CD_MUN")
                
                # Se não tiver CD_MUN no feature, tenta inferir ou pular
                # O merge anterior já garantiu CD_MUN nas properties vindas do df_data
                if cd_mun in dict_clusters:
                    c_info = dict_clusters[cd_mun]
                    feat["properties"]["cluster_id"] = int(c_info["cluster_id"])
                    feat["properties"]["cluster_name"] = str(c_info["cluster_name"])
                else:
                    feat["properties"]["cluster_id"] = -1 # Sem cluster
                    feat["properties"]["cluster_name"] = "Sem dados"

            st.sidebar.info("ℹ️ Clusters gerados com base em Volume de Vagas e Taxa de Conclusão.")

    
    # Atualiza colormap dinamicamente baseado nos dados visíveis?
    # Ou mantemos estático 0-100? Melhor estático 0-100 para consistência visual.
    # Escala personalizada: Amarelo -> Verde (Identidade Visual)
    colormap_conclusao = cm.LinearColormap(
        colors=['#FDD835', '#3F8E4D'], 
        vmin=0, vmax=100
    )
    colormap_conclusao.caption = "Percentual de conclusão (%)"
    
    # Se nenhum filtro ativo, reseta o estado do mapa para o centro e zoom iniciais
    if not algum_filtro_ativo:
        st.session_state.map_state = {"center": [-5.3159, -39.2129], "zoom": 7}
    
    #----------------------------------------------------#
    
    
    # Mapa interativo
    # Mapa interativo
    # st.markdown(
    #     "<h2 style='color:#6c91c8; font-weight:600; margin:0'>"
    #     "Mapa Interativo de Cursos por Município"
    #     "</h2>",
    #     unsafe_allow_html=True,
    # )
    
    # Ajustar a grade de layout
    col_mapa, col_metricas = st.columns([1.3, 1])
    
    with col_mapa:
        st.markdown(
            "<h4>"
            "Visualização Geoespacial"
            "</h4>",
            unsafe_allow_html=True,
        )
        
        # Chamada do fragmento (somente ele reroda nos cliques do mapa)
        # @st.fragment
        # Inserir o fragmento do mapa

        # 🔹 Chave da API do Google Maps (Fornecida pelo usuário)
        API_KEY = "AIzaSyCzCCCEss2JnTKtBsGWwpvCJMFg19svQpU"

        # 🔹 Função para gerar a URL da imagem do Street View
        def streetview_url(lat, lon, width=400, height=250, fov=85, heading=90, pitch=0):
            params = {
                "size": f"{width}x{height}",
                "location": f"{lat},{lon}",
                "fov": str(fov),
                "pitch": str(pitch),
                "heading": str(heading),
                "key": API_KEY
            }
            return f"https://maps.googleapis.com/maps/api/streetview?{up.urlencode(params)}"

        def make_kitchen_popup(props, lat, lon):
            nome_coz = props.get('NOME_USP1', 'Sem Nome')
            ender    = f"{props.get('ENDERECO5', '')}, {props.get('NUMERO6', '')}"
            bairro   = props.get('BAIRRO_L7', '')
            lote     = props.get('LOTE4', '')
            
            # URL da Imagem Estática (Street View)
            sv_img_url = streetview_url(lat, lon, width=360, height=200)
            
            # Links Externos
            rotas_car   = f"https://www.google.com/maps/dir/?api=1&destination={lat},{lon}&travelmode=driving"
            rotas_ape   = f"https://www.google.com/maps/dir/?api=1&destination={lat},{lon}&travelmode=walking"
            sv_link     = f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat},{lon}"

            html = f'''
            <div style="font-family: 'Space Grotesk', sans-serif; font-size:13px; width:360px; color:#333;">
              <div style="margin-bottom:8px; border-radius:8px; overflow:hidden; border: 1px solid #ddd;">
                <img src="{sv_img_url}" alt="Street View" style="width:100%; display:block;">
              </div>
              
              <div style="font-weight:700; font-size:15px; margin-bottom:4px; color: #2D68C4;">{nome_coz}</div>

              <div style="margin-bottom:4px;"><b>Endereço:</b> {ender}</div>
              <div style="margin-bottom:4px;"><b>Bairro:</b> {bairro}</div>
              <div style="margin-bottom:8px;">
                  <b>Lote:</b> <span style="background:#EBF3FA; border:1px solid #D6DDF1; padding:2px 8px; border-radius:12px; color: #2D68C4; font-size: 11px;">{lote}</span>
              </div>

              <div style="display:flex; flex-wrap:wrap; gap:10px; margin-top:10px; border-top: 1px solid #eee; padding-top: 8px;">
                <a href="{rotas_car}" target="_blank" style="text-decoration:none; color:#2D68C4; font-weight:600; font-size: 12px;">🚗 Carro</a>
                <a href="{rotas_ape}" target="_blank" style="text-decoration:none; color:#2D68C4; font-weight:600; font-size: 12px;">🚶 A pé</a>
                <a href="{sv_link}" target="_blank" style="text-decoration:none; color:#444; font-weight:600; font-size: 12px;">📷 Street View</a>
              </div>
            </div>
            '''
            
            return folium.Popup(IFrame(html=html, width=380, height=380), max_width=380)

        @st.fragment
        def mapa_fragment(
            municipios_geojson,
            geojson_cozinhas_csf_filtrado,
            geojson_cozinhas_focais_filtrado,
            municipios_com_qualificacao_filtrado, 
            municipios_choropleth_filtrado, 
            colormap_conclusao,
            usar_ia=False,
            dict_cluster_cores=None,
            dict_cluster_names=None
        ):
            
            # Elementos do mapa
            tooltip_municipios = folium.GeoJsonTooltip(
                fields=[
                "NM_MUN",
                "total_turmas",
                "total_concludentes",
                "percentual_conclusao",
                ],
                aliases=[
                    "Município:",
                    "Total de turmas:",
                    "Total de concluintes:",
                    "% de conclusão:",
                ],
                localize=True,
            )
    
            if "map_state" not in st.session_state:
                st.session_state.map_state = {"center": [-5.3159, -39.2129], "zoom": 7}
    
            m = folium.Map(
                location=st.session_state.map_state["center"],
                zoom_start=st.session_state.map_state["zoom"]
            )
            
            # plugins folium
            Draw(export=False, position='bottomleft').add_to(m)
            
            folium.plugins.Fullscreen(
                position="topleft",
                title="Expand me",
                title_cancel="Exit me",
                force_separate_button=True,
                force_separate_button_title="Expandir",
            ).add_to(m)
            
            
    
            # Controle de camadas
            folium.TileLayer("OpenStreetMap").add_to(m)
            
            
            municipios_qualif_feature_group = folium.FeatureGroup(name="Municípios com Qualificação").add_to(m)
            folium.GeoJson(
                municipios_com_qualificacao_filtrado,
                name="Municípios com Qualificação",
                style_function=lambda feature: {
                    "fillColor": "#4e90cc" 
                    if feature["properties"]["has_qualif"] == 1 
                    else '#f7e350',
                    'color': 'red',
                    'weight': 1,
                    'dashArray': '5, 5',
                    'fillOpacity': 0.6,
                },
                tooltip=tooltip_municipios,
            ).add_to(municipios_qualif_feature_group)
            
           
            # -------- NOVA CAMADA: Choropleth % conclusão --------
            choropleth_feature_group_indicadores = folium.FeatureGroup(
                name="Percentual de conclusão",
                show=False,
            ).add_to(m)
    
            if usar_ia and dict_cluster_cores:
                 # Estilo baseado em Cluster
                def style_cluster(feature):
                    c_id = feature["properties"].get("cluster_id", -1)
                    color = dict_cluster_cores.get(c_id, "#cccccc")
                    return {
                        "fillColor": color,
                        "color": "black",
                        "weight": 0.5,
                        "fillOpacity": 0.7,
                    }
                
                style_func = style_cluster
                
                # Tooltip adaptado para mostrar o Cluster
                tooltip_obj = folium.GeoJsonTooltip(
                    fields=["NM_MUN", "cluster_name", "percentual_conclusao"],
                    aliases=["Município:", "Perfil (IA):", "Conclusão:"],
                    localize=True,
                )
                
            else:
                 # Estilo original (Percentual)
                def style_conclusao(feature):
                    # pega o percentual da propriedade; se não houver, usa 0
                    valor = feature["properties"].get("percentual_conclusao")
                    if valor is None:
                        return {
                            "fillColor": "#cccccc",   # cinza para sem dado
                            "color": "black",
                            "weight": 0.5,
                            "fillOpacity": 0.4,
                        }
                    return {
                        "fillColor": colormap_conclusao(valor),
                        "color": "black",
                        "weight": 0.5,
                        "fillOpacity": 0.7,
                    }
                style_func = style_conclusao
                tooltip_obj = folium.GeoJsonTooltip(
                    fields=["NM_MUN", "percentual_conclusao"],
                    aliases=["Município:", "Percentual conclusão:"],
                    localize=True,
                )
    
            folium.GeoJson(
                municipios_choropleth_filtrado,
                name="Análise de Municípios" if usar_ia else "Percentual de conclusão",
                style_function=style_func,
                tooltip=tooltip_obj,
            ).add_to(choropleth_feature_group_indicadores)
    
            # adiciona a legenda (colormap ou manual)
            if usar_ia and dict_cluster_cores:
                # Legenda customizada html para clusters
                legend_html = """
                <div style="position: fixed; 
                            bottom: 50px; left: 50px; width: 180px; height: auto; 
                            border:2px solid grey; z-index:9999; font-size:12px;
                            background-color:white; padding: 10px; border-radius: 5px; opacity: 0.9;">
                  <b>Perfis de Municípios (IA)</b><br>
                """
                # Usa os nomes reais dos clusters vindos do ML
                for cid, color in dict_cluster_cores.items():
                    # Pega o nome real do cluster se disponível
                    cluster_label = dict_cluster_names.get(cid, f"Cluster {cid}") if dict_cluster_names else f"Cluster {cid}"
                    legend_html += f'<i style="background:{color};width:10px;height:10px;display:inline-block;border-radius:50%;"></i> {cluster_label}<br>'
                    
                legend_html += "</div>"
                m.get_root().html.add_child(folium.Element(legend_html))
            else:
                colormap_conclusao.add_to(m)
    
            
    
            cozinhas_csf_feature_group = folium.FeatureGroup(name="Cozinhas CSF", show=False).add_to(m)
            
            # Loop para criar Markers com Popups Customizados (Street View)
            if geojson_cozinhas_csf_filtrado:
                for feature in geojson_cozinhas_csf_filtrado["features"]:
                    props = feature["properties"]
                    geom = feature["geometry"]
                    
                    if geom["type"] == "Point":
                        lon, lat = geom["coordinates"] # GeoJSON é (Lon, Lat)
                        
                        # Cria ícone individual para evitar conflitos de referência
                        icon_obj = folium.CustomIcon(
                            icon_image="icons/icone_csf_1.png",
                            icon_size=(35, 42),
                        )
                        
                        folium.Marker(
                            location=[lat, lon],
                            icon=icon_obj,
                            popup=make_kitchen_popup(props, lat, lon),
                            tooltip=f"{props.get('NOME_USP1', 'Cozinha')}"
                        ).add_to(cozinhas_csf_feature_group)
    
            cozinhas_focais_feature_group = folium.FeatureGroup(name="Cozinhas Focais", show=False).add_to(m)
            folium.GeoJson(
                geojson_cozinhas_focais_filtrado,
                name="Cozinhas Focais",
                marker=folium.Marker(icon=cozinha_focal_icon),
                # tooltip=folium.GeoJsonTooltip(
                #     fields=["NOME_USP1", "ID0", "LOTE4"],
                #     aliases=["Nome da Cozinha: ", "ID da Cozinha: ", "Lote: "],
                #     localize=True,
                # ),
            ).add_to(cozinhas_focais_feature_group)
    
            folium.LayerControl().add_to(m)
    
            # >>> AQUI é onde a mágica acontece <<<
            st_data = st_folium(
                        m,
                        width=None,              # deixa o CSS controlar a largura
                        height=550,              # um pouco menor, pra equilibrar com os cards
                        key="mapa_qualificacao",
                        returned_objects=["last_object_clicked"],
                        center=st.session_state.map_state["center"],
                        zoom=st.session_state.map_state["zoom"],
                    )
    
    
            # Atualiza o estado global com as interações do mapa
            if st_data:
                if st_data.get("last_object_clicked"):
                    st.session_state.last_click = st_data["last_object_clicked"]
                if st_data.get("bounds"):
                    st.session_state.bounds = st_data["bounds"]
                    
                if st_data.get("center"):
                    st.session_state.map_state["center"] = st_data["center"]
                if st_data.get("zoom") is not None:
                    st.session_state.map_state["zoom"] = st_data["zoom"]
                
        
    
    
    
        # Chamada do fragmento (somente ele reroda nos cliques do mapa)
        mapa_fragment(
            municipios_geojson=municipios_geojson,
            geojson_cozinhas_csf_filtrado=geojson_cozinhas_csf_filtrado,
            geojson_cozinhas_focais_filtrado=geojson_cozinhas_focais_filtrado,
            municipios_com_qualificacao_filtrado=municipios_com_qualificacao_filtrado,
            municipios_choropleth_filtrado=municipios_choropleth_filtrado,
            # municipios_choropleth_filtrado=municipios_choropleth_filtrado,
            colormap_conclusao=colormap_conclusao,
            usar_ia=usar_ia,
            dict_cluster_cores=dict_cluster_cores,
            dict_cluster_names=dict_cluster_names if usar_ia else None
        )
    
    
    # --- MÉTRICAS À DIREITA ---
    
    # Custom CSS para estilizar os st.metrics
    
    
    
    
    def format_int_br(valor):
        """Formata inteiro com separador de milhar no padrão brasileiro."""
        if pd.isna(valor):
            return "-"
        return f"{int(valor):,}".replace(",", ".")
    
    def format_percent_br(valor, casas=2):
        """Formata percentual no padrão brasileiro (vírgula decimal)."""
        if pd.isna(valor):
            return "-"
        txt = f"{valor:.{casas}f}"
        return txt.replace(".", ",") + " %"
    
    
    with col_metricas:
        st.markdown(
            "<h4>"
            "Indicadores Gerais"
            "</h4>",
            unsafe_allow_html=True,
        )
    
        st.markdown(
            "<p style='color:#6b7280; font-size:0.85rem; margin-bottom:0.5rem;'>"
            "Os valores abaixo refletem o recorte atual dos filtros à esquerda."
            "</p>",
            unsafe_allow_html=True,
        )
    
        # Grupo 1 – Escala do Programa
        g1c1, g1c2, g1c3 = st.columns(3)
    
        with g1c1:
            st.metric("📘 Turmas", format_int_br(total_geral_turmas))
    
        with g1c2:
            # (Lógica de debug movida para seção de diagnóstico abaixo)
    
    
            # Se preferir contar por NOME (159), mude para 'Nome_Município'
            # Se preferir contar por CÓDIGO (160), mantenha 'Código Município Completo'
            st.metric(
                "🗺️ Municípios atendidos",
                format_int_br(df_filtrado['Código Município Completo'].nunique()),
            )
    
        with g1c3:
            st.metric("🎯 Vagas ofertadas", format_int_br(total_geral_vagas))
    
    
    
    
        # Espaço entre os grupos
        st.markdown("<br/>", unsafe_allow_html=True)
    
        # Grupo 2 – Participação e Conclusão
        g2c1, g2c2, g2c3 = st.columns(3)
    
        with g2c1:
            st.metric("👥 Inscritos", format_int_br(total_geral_inscritos))
    
        with g2c2:
            st.metric("🏅 Concludentes", format_int_br(concludentes_ajustados))
    
        with g2c3:
            st.metric(
                "📈 Conclusão geral (%)", 
                format_percent_br(percentual_geral_conclusao),
                help="Cálculo: (Total de Concludentes / Total de Vagas Ofertadas) * 100"
            )
            
        # Grupo 3 – Status das Turmas
        st.markdown("<br/>", unsafe_allow_html=True)
        g3c1, g3c2 = st.columns(2) 
        
        with g3c1:
            st.metric("⏳ Turmas em Execução", format_int_br(turmas_em_execucao))
            
        with g3c2:
            st.metric("✅ Turmas Concluídas", format_int_br(turmas_concluidas))
    
    #----------------------------------------------------#
    
    
    
    # -------------------------------------------------------------------
    # Análises Temporais dos Cursos
    # -------------------------------------------------------------------
    
    st.subheader("Análises Temporais dos Cursos")
    
    # Garantir que temos datas válidas depois de todos os filtros
    if (
        col_data_inicio in df_filtrado.columns
        and df_filtrado[col_data_inicio].notna().sum() > 0
    ):
    
        # 1) Série temporal de turmas iniciadas por mês
        df_temporal = (
            df_filtrado
            .dropna(subset=[col_data_inicio])
            .copy()
        )
        df_temporal["PERIODO_M"] = df_temporal[col_data_inicio].dt.to_period("M")
        df_temporal_group = (
            df_temporal
            .groupby("PERIODO_M")
            .size()
            .reset_index(name="qtd_turmas")
        )
        df_temporal_group["DATA"] = df_temporal_group["PERIODO_M"].dt.to_timestamp()
    
        chart_turmas_mes = (
            alt.Chart(df_temporal_group)
            .mark_line(point=True)
            .encode(
                x=alt.X("DATA:T", title="Mês de início"),
                y=alt.Y("qtd_turmas:Q", title="Quantidade de turmas"),
                tooltip=["DATA:T", "qtd_turmas:Q"],
            )
            .properties(
                height=300,
                title="Turmas iniciadas por mês",
            )
        )
    
        st.altair_chart(chart_turmas_mes, use_container_width=True)
    
        # 2) Turmas por trimestre (se a coluna existir)
        if "TRIMESTRE" in df_filtrado.columns:
            df_trim = (
                df_filtrado
                .groupby("TRIMESTRE")["CURSO"]
                .count()
                .reset_index(name="qtd_turmas")
            )
    
            chart_trim = (
                alt.Chart(df_trim)
                .mark_bar()
                .encode(
                    x=alt.X("TRIMESTRE:N", title="Trimestre"),
                    y=alt.Y("qtd_turmas:Q", title="Quantidade de turmas"),
                    tooltip=["TRIMESTRE", "qtd_turmas"],
                )
                .properties(
                    height=300,
                    title="Turmas por trimestre",
                )
            )
    
            st.altair_chart(chart_trim, use_container_width=True)
    
        # 3) Evolução de turmas por executora ao longo do tempo (se a coluna existir)
        if "EXECUTORA" in df_filtrado.columns:
            df_exec_tempo = (
                df_temporal
                .groupby(["PERIODO_M", "EXECUTORA"])
                .size()
                .reset_index(name="qtd_turmas")
            )
            df_exec_tempo["DATA"] = df_exec_tempo["PERIODO_M"].dt.to_timestamp()
    
            chart_exec = (
                alt.Chart(df_exec_tempo)
                .mark_line(point=True)
                .encode(
                    x=alt.X("DATA:T", title="Mês de início"),
                    y=alt.Y("qtd_turmas:Q", title="Turmas iniciadas"),
                    color=alt.Color("EXECUTORA:N", title="Executora"),
                    tooltip=["EXECUTORA", "DATA:T", "qtd_turmas:Q"],
                )
                .properties(
                    height=350,
                    title="Evolução das turmas iniciadas por executora",
                )
            )
    
            st.altair_chart(chart_exec, use_container_width=True)
    
    
    
    
    
    
    
    
    # -------------------------------------------------------------
    # Scatter Plot: Duração dos cursos × Taxa de conclusão (%)
    # -------------------------------------------------------------
    if "DURACAO_DIAS" in df_filtrado.columns and "TAXA_CONCLUSAO" in df_filtrado.columns:
    
        chart_taxa = (
            alt.Chart(df_filtrado)
            .mark_circle(size=70, opacity=0.7)
            .encode(
                x=alt.X("DURACAO_DIAS:Q", title="Duração do curso (dias)"),
                y=alt.Y("TAXA_CONCLUSAO:Q", title="Taxa de Conclusão (%)"),
                color=alt.Color("EXECUTORA:N", title="Executora"),
                tooltip=["CURSO", "EXECUTORA", "DURACAO_DIAS", "CONCLUDENTES", "VAGAS OFERTADAS", "TAXA_CONCLUSAO"],
            )
            .properties(
                title="Duração do Curso × Taxa de Conclusão (%)",
                height=350,
            )
        )
    
        st.altair_chart(chart_taxa, use_container_width=True)
    # -------------------------------------------------------------------
    
    
    # -------------------------------------------------------------
    # Taxa média de conclusão por área de qualificação
    # -------------------------------------------------------------
    if "ÁREA DO CURSO (automático)" in df_filtrado.columns and "TAXA_CONCLUSAO" in df_filtrado.columns:
    
        df_area = (
            df_filtrado
            .groupby("ÁREA DO CURSO (automático)")["TAXA_CONCLUSAO"]
            .mean()
            .reset_index()
        )
    
        chart_area = (
            alt.Chart(df_area)
            .mark_bar()
            .encode(
                x=alt.X("TAXA_CONCLUSAO:Q", title="Taxa média de conclusão (%)"),
                y=alt.Y("ÁREA DO CURSO (automático):N", title="Área de qualificação"),
                color=alt.Color("ÁREA DO CURSO (automático):N", legend=None),
                tooltip=["ÁREA DO CURSO (automático)", "TAXA_CONCLUSAO"],
            )
            .properties(
                height=350,
                title="Taxa Média de Conclusão por Área de Qualificação",
            )
        )
    
        st.altair_chart(chart_area, use_container_width=True)
    
    
    # -------------------------------------------------------------
    # Boxplot: Distribuição da taxa de conclusão por área
    # -------------------------------------------------------------
    chart_box_area = (
        alt.Chart(df_filtrado)
        .mark_boxplot(
            size=25,          # reduz largura para evitar sobreposição
            # extent="min-max"  # mostra todos os outliers
        )
        .encode(
            x=alt.X(
                "TAXA_CONCLUSAO:Q",
                title="Taxa de Conclusão (%)",
                # scale=alt.Scale(domain=[0, 100])
            ),
            y=alt.Y(
                "ÁREA DO CURSO (automático):N",
                title="Área de Qualificação",
                sort="-x"      # ordena por mediana descendente
            ),
            color=alt.Color(
                "ÁREA DO CURSO (automático):N",
                # legend=None
            ),
            tooltip=[
                alt.Tooltip("ÁREA DO CURSO (automático):N", title="Área"),
                alt.Tooltip("TAXA_CONCLUSAO:Q", format=".1f", title="Taxa (%)")
            ]
        )
        .properties(
            width=1100,   # largura real
            height=600,   # altura generosa
            title="Distribuição da Taxa de Conclusão por Área de Qualificação"
        )
    )
    
    st.altair_chart(chart_box_area, theme=None)
    
    
    
    
    
    
    
    
    
    
    # Gráficos
    
    col_grafico1, col_grafico2 = st.columns(2)
    
    with col_grafico1:
        # Gráfico de barras dos 10 municípios com mais concludentes
        st.markdown(
            "<h4>"
            "Top 10 Municípios por Concludentes"
            "</h4>",
            unsafe_allow_html=True,
        )
        top_mun = (
            df_metrics.sort_values("total_concludentes", ascending=False)
            .head(10)
        )
    
        chart = (
            alt.Chart(top_mun)
            .mark_bar()
            .encode(
                x=alt.X("total_concludentes:Q", title="Concludentes"),
                y=alt.Y("Nome_Município:N", sort="-x", title="Município"),
                tooltip=["Nome_Município", "total_concludentes"]
            )
            .properties(height=300)
        )
    
        st.altair_chart(chart, use_container_width=True)
        
    with col_grafico2:
        # Gráfico de barras dos 10 cursos com mais turmas
        st.markdown(
            "<h4>"
            "Top 10 Cursos por Número de Turmas"
            "</h4>",
            unsafe_allow_html=True,
        )
        top_cursos = (
            df_filtrado.groupby("CURSO")
            .agg(total_turmas=pd.NamedAgg(column="CURSO", aggfunc="count"))
            .reset_index()
            .sort_values("total_turmas", ascending=False)
            .head(10)
        )
        chart2 = (
            alt.Chart(top_cursos)
            .mark_bar()
            .encode(
                x=alt.X("total_turmas:Q", title="Número de Turmas"),
                y=alt.Y("CURSO:N", sort="-x", title="Curso"),
                tooltip=["CURSO", "total_turmas"]
            )
            .properties(height=300)
        )
        st.altair_chart(chart2, use_container_width=True)
        
        
        
    col_grafico3, col_grafico4 = st.columns(2)
    
    with col_grafico3:
        # Gráfico de barras dos 10 municípios com mais concludentes
        st.markdown(
            "<h4>"
            "Top 10 Concludentes por Executora"
            "</h4>",
            unsafe_allow_html=True,
        )
        top_exec = (
            df_metrics_exec.sort_values("total_concludentes", ascending=False)
            .head(10)
        )
    
        chart3 = (
            alt.Chart(top_exec)
            .mark_bar()
            .encode(
                x=alt.X("total_concludentes:Q", title="Concludentes"),
                y=alt.Y("EXECUTORA:N", sort="-x", title="Executora"),
                tooltip=["EXECUTORA", "total_concludentes"]
            )
            .properties(height=300)
        )
    
        st.altair_chart(chart3, use_container_width=True)
        
    with col_grafico4:
        # Gráfico de barras dos 10 cursos com mais turmas
        st.markdown(
            "<h4>"
            "Top 10 Turmas por Executora"
            "</h4>",
            unsafe_allow_html=True,
        )
        top_cursos = (
            df_filtrado.groupby("EXECUTORA")['CURSO'].count()
            .reset_index()
            .sort_values("CURSO", ascending=False)
        )
        chart4 = (
            alt.Chart(top_cursos)
            .mark_bar()
            .encode(
                x=alt.X("CURSO:Q", title="Número de Turmas"),
                y=alt.Y("EXECUTORA:N", sort="-x", title="Executora"),
                tooltip=["EXECUTORA", "CURSO"]
            )
            .properties(height=300)
        )
        st.altair_chart(chart4, use_container_width=True)
        
    # -------------------------------------------------------------------
    # Diagnóstico de Qualidade dos Dados
    # -------------------------------------------------------------------
    with st.expander("⚠️ Diagnóstico de Qualidade dos Dados (Clique para ver)", expanded=False):
        st.markdown("Verificação automática de inconsistências nos dados carregados.")
        
        # 1. Checagem de Concludentes > Inscritos
        inconsistent_rows = df_filtrado[df_filtrado["CONCLUDENTES"] > df_filtrado["INSCRITOS"]]
        if not inconsistent_rows.empty:
            st.error(f"Foram encontradas {len(inconsistent_rows)} turmas com mais Concludentes do que Inscritos.")
            st.markdown("**Aviso:** *Os indicadores acima exibem os valores reais inseridos na planilha, conforme solicitado, permitindo identificar inconsistências nos dados originais.*")
            st.dataframe(
                inconsistent_rows[["CURSO", "Nome_Município", "INSCRITOS", "CONCLUDENTES", "TAXA_CONCLUSAO"]],
                use_container_width=True
            )
        else:
            st.success("✅ Nenhuma inconsistência de 'Concludentes > Inscritos' encontrada no filtro atual.")
    
        # 2. Checagem de Múltiplos IDs para mesma cidade
        df_debug = df_filtrado[["Nome_Município", "Código Município Completo"]].copy()
        if "Nome_Município" in df_debug.columns:
            df_debug["Nome_Município"] = df_debug["Nome_Município"].astype(str).str.strip().str.upper()
            check_dups = df_debug.groupby("Nome_Município")["Código Município Completo"].nunique()
            dups = check_dups[check_dups > 1]
            
            if not dups.empty:
                st.warning(f"⚠️ Cidades com múltiplos códigos ID encontrados: {dups.index.tolist()}")
                st.dataframe(df_filtrado[
                    df_filtrado["Nome_Município"].astype(str).str.strip().str.upper().isin(dups.index)
                ][["Nome_Município", "Código Município Completo"]].drop_duplicates())
            else:
                st.success("✅ Nenhum conflito de ID de município encontrado.")
    
    #----------------------------------------------------#
    
    
    # -------------------------------------------------------------------
    # Dados Detalhados e Exportação
    # -------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Dados Detalhados")
    
    # Mostra dataframe filtrado
    # Mostra dataframe filtrado (removendo colunas de índice/unnamed se existirem)
    cols_to_show = [c for c in df_filtrado.columns if "Unnamed" not in c]
    
    st.dataframe(
        df_filtrado[cols_to_show], 
        use_container_width=True,
        hide_index=True, # Garante que o índice numérico do pandas também não apareça
        column_config={
            "TAXA_CONCLUSAO": st.column_config.NumberColumn(
                "Taxa Conclusão (%)",
                format="%.2f %%"
            )
        }
    )
    
    # Botão de download
    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False, sep=";").encode('utf-8')
    
    csv = convert_df(df_filtrado)
    
    st.download_button(
        label="📥 Baixar Dados Filtrados (CSV)",
        data=csv,
        file_name='dados_qualificacao_filtrados.csv',
        mime='text/csv',
        key='download-csv'
    )
    
        
    
    
#----------------------------------------------------#
# FUNÇÕES DE OTIMIZAÇÃO (CACHE) - JORNADA EMPREENDEDORA
#----------------------------------------------------#

@st.cache_data
def compute_jornada_metrics(df_filtrado, df_trilha, df_mentoria, selected_municipios):
    """
    Processa todos os indicadores e dataframes da aba Jornada Empreendedora.
    Retorna um dicionário com os resultados prontos.
    """
    # Copia para não alterar originais
    df_trilha_proc = df_trilha.copy()
    df_mentoria_proc = df_mentoria.copy()
    
    # --- Filtros Específicos da Aba ---
    if selected_municipios:
        if "CIDADE" in df_trilha_proc.columns:
             df_trilha_proc = df_trilha_proc[df_trilha_proc["CIDADE"].isin(selected_municipios)]
        if "MUNICÍPIO" in df_mentoria_proc.columns:
             df_mentoria_proc = df_mentoria_proc[df_mentoria_proc["MUNICÍPIO"].isin(selected_municipios)]

    # --- Agregação Qualificação (Nível 0) ---
    # df_filtrado já vem filtrado pela sidebar
    df_qualificacao_agg = df_filtrado.groupby("Nome_Município")["CONCLUDENTES"].sum().reset_index()
    df_qualificacao_agg.rename(columns={"Nome_Município": "NM_MUN", "CONCLUDENTES": "qtd_qualificacao"}, inplace=True)
    
    # --- KPIs ---
    col_sensib = next((c for c in df_trilha_proc.columns if "SENSIBILIZA" in c.upper()), None)
    kpi_sensib = 0
    if col_sensib:
        kpi_sensib = pd.to_numeric(df_trilha_proc[col_sensib], errors='coerce').fillna(0).sum()
    
    kpi_inscritos = df_trilha_proc["INSCRITOS TRILHA"].sum() if "INSCRITOS TRILHA" in df_trilha_proc.columns else 0
    kpi_concluintes_trilha = df_trilha_proc["CONCLUDENTES TRILHA"].sum() if "CONCLUDENTES TRILHA" in df_trilha_proc.columns else 0
    
    kpi_mentorados = 0
    if "STATUS" in df_mentoria_proc.columns:
         kpi_mentorados = df_mentoria_proc[df_mentoria_proc["STATUS"].astype(str).str.strip().str.lower() == "concluído"].shape[0]
         
    kpi_qualificados = df_qualificacao_agg["qtd_qualificacao"].sum()
    taxa_conversao_global = (kpi_mentorados / kpi_qualificados * 100) if kpi_qualificados > 0 else 0

    # --- Dados Funil ---
    funnel_data = pd.DataFrame({
        "Etapa": ["Sensibilização", "Inscrição Trilha", "Conclusão Trilha", "Mentoria Concluída"],
        "Quantidade": [kpi_sensib, kpi_inscritos, kpi_concluintes_trilha, kpi_mentorados],
        "Order": [1, 2, 3, 4]
    })
    
    # --- Dados Comparativos (Merge) ---
    # Agregação Trilha
    if "CIDADE" in df_trilha_proc.columns:
        agg_trilha = df_trilha_proc.groupby("CIDADE")["CONCLUDENTES TRILHA"].sum().reset_index()
        agg_trilha.rename(columns={"CIDADE": "NM_MUN", "CONCLUDENTES TRILHA": "qtd_trilha"}, inplace=True)
    else:
        agg_trilha = pd.DataFrame(columns=["NM_MUN", "qtd_trilha"])
        
    # Agregação Mentoria
    if "MUNICÍPIO" in df_mentoria_proc.columns:
        df_ment_concl = df_mentoria_proc[df_mentoria_proc["STATUS"].astype(str).str.strip().str.lower() == "concluído"]
        agg_mentoria = df_ment_concl.groupby("MUNICÍPIO").size().reset_index(name="qtd_mentoria")
        agg_mentoria.rename(columns={"MUNICÍPIO": "NM_MUN"}, inplace=True)
    else:
        agg_mentoria = pd.DataFrame(columns=["NM_MUN", "qtd_mentoria"])

    # Normalização de nomes para Merge
    df_qualificacao_agg["NM_MUN_UPPER"] = df_qualificacao_agg["NM_MUN"].astype(str).str.upper().str.strip()
    agg_trilha["NM_MUN_UPPER"] = agg_trilha["NM_MUN"].astype(str).str.upper().str.strip()
    agg_mentoria["NM_MUN_UPPER"] = agg_mentoria["NM_MUN"].astype(str).str.upper().str.strip()
    
    df_comparativo = df_qualificacao_agg.merge(agg_trilha[["NM_MUN_UPPER", "qtd_trilha"]], on="NM_MUN_UPPER", how="left")
    df_comparativo = df_comparativo.merge(agg_mentoria[["NM_MUN_UPPER", "qtd_mentoria"]], on="NM_MUN_UPPER", how="left").fillna(0)
    df_comparativo["total_empreend"] = df_comparativo["qtd_trilha"] + df_comparativo["qtd_mentoria"]
    
    # Dados para Gráfico de Barras (Top 10)
    df_top = df_comparativo.sort_values("qtd_qualificacao", ascending=False).head(10)
    df_long = df_top.melt(id_vars=["NM_MUN"], value_vars=["qtd_qualificacao", "qtd_mentoria"], 
                          var_name="Tipo", value_name="Quantidade")
    df_long["Tipo"] = df_long["Tipo"].map({"qtd_qualificacao": "Qualificados", "qtd_mentoria": "Mentorados"})

    return {
        "kpis": (kpi_qualificados, kpi_sensib, kpi_inscritos, kpi_concluintes_trilha, kpi_mentorados, taxa_conversao_global),
        "funnel_data": funnel_data,
        "df_comparativo": df_comparativo,
        "df_long": df_long
    }

@st.cache_data
def prepare_jornada_map_data(df_comparativo, _municipios_geojson_data):
    """
    Prepara o GeoJSON e dados de plotagem para o mapa da Jornada.
    O argumento _municipios_geojson_data inicia com _ para não ser hashado (assumido constante).
    """
    # Prepara GeoDataFrame Base
    gdf_base = gpd.GeoDataFrame.from_features(_municipios_geojson_data["features"])
    if "CD_MUN" in gdf_base.columns:
         gdf_base["CD_MUN"] = gdf_base["CD_MUN"].astype(str).str.zfill(7)
    
    gdf_base["NM_MUN_UPPER"] = gdf_base["NM_MUN"].astype(str).str.upper().str.strip()
    
    # Limpeza de colunas duplicadas
    cols_drop_gdf = ["total_turmas", "total_concludentes", "qtd_trilha", "qtd_mentoria", "total_empreend"]
    gdf_base_clean = gdf_base.drop(columns=[c for c in cols_drop_gdf if c in gdf_base.columns], errors='ignore')
    
    
    # Merge com dados comparativos
    # Remover NM_MUN de df_comparativo para evitar duplicação (gdf_base já possui via linha 1846)
    df_comp_clean = df_comparativo.drop(columns=["NM_MUN"], errors='ignore')
    gdf_jornada = gdf_base_clean.merge(df_comp_clean, on="NM_MUN_UPPER", how="left").fillna(0)
    
    # FILTRO: Remover Fortaleza e Municípios com Mentoria zerada do GeoJSON de plotagem
    gdf_jornada = gdf_jornada[
        (gdf_jornada["NM_MUN_UPPER"] != "FORTALEZA") & 
        (gdf_jornada["qtd_mentoria"] > 0)
    ]
    
    # Garantir que NM_MUN existe no GeoJSON final (APÓS os filtros)
    # Deve vir de gdf_base, mas criar se não existir para garantir tooltip funcione
    if "NM_MUN" not in gdf_jornada.columns:
        gdf_jornada["NM_MUN"] = gdf_jornada["NM_MUN_UPPER"].str.title()
    
    # Dados de Plotagem (Interior apenas e com Mentoria > 0)
    df_interior = df_comparativo[
        (df_comparativo["NM_MUN_UPPER"] != "FORTALEZA") &
        (df_comparativo["qtd_mentoria"] > 0)
    ]
    
    
    # Bins Dinâmicos
    dist_values = df_interior["qtd_mentoria"]
    interior_max = dist_values.max() if not dist_values.empty else 0
    interior_min = dist_values.min() if not dist_values.empty else 0

    bins = [0, 1, 2, 3] # default
    if interior_max > 0:
        quantiles = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        calculated_bins = list(dist_values.quantile(quantiles))
        calculated_bins[0] = interior_min
        calculated_bins[-1] = interior_max
        calculated_bins = sorted(list(set(calculated_bins)))
        
        if len(calculated_bins) < 4:
            if interior_min == interior_max:
                 calculated_bins = [interior_min, interior_min + 1, interior_min + 2, interior_min + 3]
            else:
                 calculated_bins = list(np.linspace(interior_min, interior_max, 4))
        
        bins = sorted(list(set(calculated_bins)))
        while len(bins) < 4:
             bins.append(bins[-1] + 1)
             
    return json.loads(gdf_jornada.to_json()), bins, df_interior

#----------------------------------------------------#

with tab2:
    st.markdown("## Jornada do Empreendedor")
    st.markdown("Acompanhamento das etapas de Pós-Qualificação: **Sensibilização**, **Trilha Empreendedora** e **Mentoria**.")

    # 1. Carregar Dados de Jornada (Cached)
    df_trilha, df_mentoria = load_jornada_data()

    if df_trilha.empty and df_mentoria.empty:
        st.info("Nenhum dado de Jornada disponível no momento.")
    else:
        # --- PROCESSAMENTO OTIMIZADO (CACHED) ---
        # Chama a função que processa tudo de uma vez
        with st.spinner("Processando indicadores da Jornada..."):
            jornada_data = compute_jornada_metrics(
                df_filtrado, 
                df_trilha, 
                df_mentoria, 
                selected_municipios
            )
        
        # Desempacota resultados
        kpis = jornada_data["kpis"] 
        # (kpi_qualificados, kpi_sensib, kpi_inscritos, kpi_concluintes_trilha, kpi_mentorados, taxa_conversao_global)
        kpi_qualificados, kpi_sensib, kpi_inscritos, kpi_concluintes_trilha, kpi_mentorados, taxa_conversao_global = kpis
        
        df_comparativo = jornada_data["df_comparativo"]
        funnel_data = jornada_data["funnel_data"]
        df_long = jornada_data["df_long"]

        # --- EXIBIÇÃO ---
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Qualificados (Fase 1)", f"{int(kpi_qualificados)}")
        col2.metric("Sensibilizados", f"{int(kpi_sensib)}")
        col3.metric("Inscritos Trilha", f"{int(kpi_inscritos)}")
        col4.metric("Concluintes Trilha", f"{int(kpi_concluintes_trilha)}")
        col5.metric("Mentorias Concluídas", f"{int(kpi_mentorados)}", delta=f"{taxa_conversao_global:.1f}% Conv.")
        st.divider()
        
        # --- Gráfico de Funil ---
        st.subheader("Funil de Conversão")
        chart_funnel = alt.Chart(funnel_data).mark_bar().encode(
            x=alt.X('Quantidade:Q', title='Participantes'),
            y=alt.Y('Etapa:N', sort=alt.EncodingSortField(field="Order", order="ascending"), title=None),
            color=alt.Color('Etapa:N', legend=None, scale=alt.Scale(scheme='tableau10')),
            tooltip=['Etapa', 'Quantidade']
        ).properties(height=300)
        
        text_funnel = chart_funnel.mark_text(align='left', baseline='middle', dx=3).encode(text='Quantidade:Q')
        st.altair_chart(chart_funnel + text_funnel, use_container_width=True)
        st.divider()
        
        # --- Gráfico Eficiência (Top 10) ---
        st.subheader("Eficiência por Município (Top 10)")
        chart_bars = alt.Chart(df_long).mark_bar().encode(
            x=alt.X('NM_MUN:N', sort='-y', title=None), 
            y=alt.Y('Quantidade:Q', title='Alunos'),
            color=alt.Color('Tipo:N', scale=alt.Scale(domain=['Qualificados', 'Mentorados'], range=['#6c91c8', '#ffaa00'])),
            xOffset='Tipo:N',
            tooltip=['NM_MUN', 'Tipo', 'Quantidade']
        ).properties(height=400)
        st.altair_chart(chart_bars, use_container_width=True)
        st.divider()

        # --- Mapa (Processamento Cached) ---
        st.subheader("Distribuição Geográfica (Empreendedorismo)")
        
        # Chama preparação do mapa cacheada
        geojson_jornada, bins_jornada, df_interior_plot = prepare_jornada_map_data(
            df_comparativo, 
            municipios_com_qualificacao
        )
        
        # Verificar se há dados para exibir no mapa
        # (GeoJSON pode estar vazio se município filtrado não tem mentorias)
        if not geojson_jornada or len(geojson_jornada.get('features', [])) == 0:
            st.info("ℹ️ Nenhum município com mentorias concluídas para exibir no mapa. Os municípios selecionados podem não ter dados de mentoria registrados.")
        else:
            # Fragmento do Mapa para evitar Reruns Globais
            @st.fragment
            def render_mapa_jornada(geojson, data_plot, bins):
                m_jornada = folium.Map(location=[-5.3159, -39.2129], zoom_start=7)
                
                folium.Choropleth(
                    geo_data=geojson,
                    data=data_plot,
                    columns=["NM_MUN_UPPER", "qtd_mentoria"],
                    key_on="feature.properties.NM_MUN_UPPER",
                    fill_color="YlOrRd", 
                    fill_opacity=0.7,
                    line_opacity=0.2,
                    legend_name="Total Mentorias (Interior)",
                    threshold_scale=bins,
                    highlight=True
                ).add_to(m_jornada)
                
                folium.GeoJson(
                    geojson,
                    style_function=lambda x: {'fillColor': '#00000000', 'color': '#00000000'},
                    tooltip=folium.GeoJsonTooltip(
                        fields=["NM_MUN", "qtd_qualificacao", "qtd_trilha", "qtd_mentoria"],
                        aliases=["Município:", "Qualificados:", "Trilha:", "Mentorados:"],
                        localize=True
                    )
                ).add_to(m_jornada)
                
                st_folium(m_jornada, width=700, height=500, key="mapa_jornada_frag", returned_objects=["last_object_clicked"])

            # Chama o fragmento
            render_mapa_jornada(geojson_jornada, df_interior_plot, bins_jornada)
        
        # --- Tabela ---
        st.subheader("Tabela de Efetividade")
        df_comparativo["conv_rate"] = (df_comparativo["qtd_mentoria"] / df_comparativo["qtd_qualificacao"] * 100).fillna(0)
        
        st.dataframe(
            df_comparativo[["NM_MUN", "qtd_qualificacao", "qtd_trilha", "qtd_mentoria", "conv_rate"]]
            .sort_values("qtd_qualificacao", ascending=False)
            .rename(columns={
                "NM_MUN": "Município",
                "qtd_qualificacao": "Total Qualificados",
                "qtd_trilha": "Concluintes Trilha",
                "qtd_mentoria": "Mentorias Concluídas",
                "conv_rate": "Taxa de Conversão (%)"
            }),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Taxa de Conversão (%)": st.column_config.ProgressColumn(
                    "Conversão (Qualif. -> Mentoria)",
                    format="%.1f%%",
                    min_value=0, max_value=100
                )
            }
        )


