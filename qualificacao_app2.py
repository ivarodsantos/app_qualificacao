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

from branca.colormap import linear
from branca.element import MacroElement, Template
from merge_id_plataforma import merge_id_plataforma

# Configurações iniciais do Streamlit
st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
    /* remove espaços extras entre componentes */
    div.block-container {
        padding-top: 0rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("""
<style>

    /* ======== 1. Fundo geral com textura geométrica suave ======== */
    body {
        background-color: #F7F9FB;
        background-image: url('https://i.imgur.com/5T0R07N.png'); /* textura similar à página 11 */
        background-size: cover;
        background-attachment: fixed;
    }

    /* Remove padding superior padrão do Streamlit */
    .block-container {
        padding-top: 1rem;
    }

    /* ======== 2. Títulos no azul institucional ======== */
    h1, h2, h3, h4 {
        color: #1C4D99;   /* azul das páginas 1 e 3 do PDF */
        font-weight: 700;
    }

    /* Subtítulos e labels */
    label, .stSelectbox, .stMultiselect {
        color: #4A7CC2 !important;  /* azul secundário */
        font-weight: 600;
    }

    /* ======== 3. Cards de métricas no estilo CSF ======== */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #EAF0FF 0%, #FFFFFF 90%);
        border-radius: 18px;
        padding: 1.5rem 1.3rem;
        min-height: 160px;
        border: 1px solid #D6DDF1;
        box-shadow: 0 6px 16px rgba(28, 77, 153, 0.15);
    }

    div[data-testid="stMetricLabel"] {
        font-size: 1.0rem;
        color: #1C4D99;
        font-weight: 600;
    }

    div[data-testid="stMetricValue"] {
        font-size: 2.3rem;
        font-weight: 700;
        color: #D33F3F;  /* vermelho do coração/logomarca */
    }

    /* ======== 4. Botões com tema institucional ======== */
    button[kind="primary"] {
        background-color: #3F8E4D !important; /* verde governo */
        color: white !important;
        border-radius: 10px !important;
        border: none !important;
    }

    button[kind="primary"]:hover {
        background-color: #2E6E3A !important;
    }

    /* ======== 5. Selects (multiselect / selectbox) ======== */
    section[data-testid="stSidebar"] {
        background-color: #F3F4F6;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label {
        color: #1C4D99;           /* azul institucional */
        font-weight: 600;
    }

    /* Caixa dos selects na sidebar (multiselect / selectbox) */
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background-color: #FFFFFF;
        border-radius: 10px;
        border: 1px solid #CBD5E1;              /* cinza suave */
        box-shadow: 0 1px 3px rgba(15,23,42,.08);
    }

    /* texto dentro do select */
    section[data-testid="stSidebar"] div[data-baseweb="select"] * {
        font-size: 0.90rem;
    }


    /* ======== 6. Painéis de seção ======== */
    .section-box {
        background-color: #FFFFFF80;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        border-left: 6px solid #4A7CC2;
        margin-top: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 3px 10px rgba(0,0,0,0.06);
    }

    /* ======== 7. Divisor temático ======== */
    .divider {
        height: 4px;
        background: linear-gradient(to right, #1C4D99, #F4CE3B, #D33F3F);
        margin: 1rem 0;
        border-radius: 2px;
    }

    /* ======== Mapa com borda institucional e largura total ======== */
    iframe, .folium-map {
        border: 3px solid #1C4D99 !important;
        border-radius: 12px !important;
        box-shadow: 0 6px 16px rgba(0,0,0,0.15);
        width: 100% !important;          /* ocupa toda a largura da coluna */
        max-width: 100% !important;
    }
    
    /* ====== CONTAINER BASE DO CARD (st.metric) ====== */
    div[data-testid="stMetric"] {
        border-radius: 18px;
        padding: 1.5rem 1.3rem;
        min-height: 150px;
        border: 1px solid #D6DDF1;
        box-shadow: 0 6px 16px rgba(15, 23, 42, 0.09);
        background: #FFFFFF;   /* base branca, vamos colorir por card abaixo */
    }

    /* Label dos cards */
    div[data-testid="stMetric"] label[data-testid="stMetricLabel"] {
        font-size: 1.0rem;
        color: #1C4D99;
        font-weight: 600;
    }

    /* Valor dos cards */
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 2.3rem;
        font-weight: 700;
        color: #111827;
    }
    
    /* 1º card – Turmas (azul claro) */
    div[data-testid="stMetric"]:nth-of-type(1) {
        background: linear-gradient(135deg, #EAF0FF 0%, #FFFFFF 70%);
    }

    /* 2º card – Municípios atendidos (verde claro) */
    div[data-testid="stMetric"]:nth-of-type(2) {
        background: linear-gradient(135deg, #E5F5EA 0%, #FFFFFF 70%);
    }

    /* 3º card – Vagas ofertadas (amarelo claro) */
    div[data-testid="stMetric"]:nth-of-type(3) {
        background: linear-gradient(135deg, #FFF7D6 0%, #FFFFFF 70%);
    }

    /* 4º card – Inscritos (azul médio) */
    div[data-testid="stMetric"]:nth-of-type(4) {
        background: linear-gradient(135deg, #E0ECFF 0%, #FFFFFF 70%);
    }

    /* 5º card – Concludentes (vermelho suave) */
    div[data-testid="stMetric"]:nth-of-type(5) {
        background: linear-gradient(135deg, #FFE3E0 0%, #FFFFFF 70%);
    }

    /* 6º card – Conclusão geral (%) (blend azul + amarelo) */
    div[data-testid="stMetric"]:nth-of-type(6) {
        background: linear-gradient(135deg, #EAF0FF 0%, #FFF7D6 60%, #FFFFFF 100%);
    }


    
    </style>
    """, 
    unsafe_allow_html=True)

    


# — Fonte global + Material Icons —
st.markdown(
    """
    
    <style>
    /* carrega a fonte com todos os pesos disponíveis */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

    /* aplica globalmente na app, mas NÃO nos spans de ícone */
    .stApp, .stAppViewContainer, .main, .block-container,
    h1, h2, h3, h4, h5, h6,
    p, label, li, a, button, input, textarea, select {
        font-family: 'Space Grotesk', sans-serif !important;
    }

    
    </style>
    """,
    unsafe_allow_html=True,
)




st.markdown(
    "<h1 style='color:#6c91c8; font-weight:700; margin:0'>"
    "Qualificação App - Análise de Cursos e Concludentes"
    "</h1>",
    unsafe_allow_html=True,
)


#-------------- Carregamento dos dados --------------#
@st.cache_data
def load_data():
    # Carregar dados dos cursos
    cursos_df = pd.read_csv(
        "data2/compilado_novos_lotes_merge_nomes_cozinhas_25112025.csv",
        encoding="utf-8",
        sep=";",
        # dtype={"Nº LOTE": "string", "Município": "string"},
    )
    
    # carregar dados quantidade de beneficiários e cozinhas por lote, região e município
    qtd_beneficiarios_cozinhas_df = pd.read_csv(
        "data2/quantidade_beneficiarios_e_cozinhas_lote_regiao_municipio_03112025.csv",
        encoding="utf-8",
        sep=",",
    )
    # carregar dados de quantidade de cozinhas por lote, região e município
    qtd_cozinhas_df = pd.read_csv(
        "data2/quantidade_beneficiarios_e_cozinhas_lote_regiao_municipio_03112025.csv",
        encoding="utf-8",
        sep=",",
    )
    
    df_kitchen = pd.read_csv(
        "data2/data-1762178638816_kitchen.csv"
    )
    
    return cursos_df, qtd_beneficiarios_cozinhas_df, qtd_cozinhas_df, df_kitchen


cursos_df, qtd_beneficiarios_cozinhas_df, qtd_cozinhas_df, df_kitchen = load_data()


geojsons = [
    "data/municipios_latlon.geojson",
    "data2/cozinhas_geo_ipece_01122025.geojson",
    "data2/municipios_com_qualificacao.geojson",
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
        aggfunc=lambda x: round((x.sum() / merged_df.loc[x.index, "INSCRITOS"].sum()) * 100, 2) 
        if merged_df.loc[x.index, "INSCRITOS"].sum() > 0 else 0
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



# ----------------- Enriquecer GeoJSON com indicadores ----------------- #

# 1. Preparar coluna de código no DataFrame agregado
merged_df_agg["CD_MUN"] = (
    merged_df_agg["Código Município Completo"]
    .astype(str)
    .str.zfill(7)          # garante 7 dígitos: 2300101, 2300200, etc.
)

# 2. Transformar o GeoJSON de municípios em GeoDataFrame
gdf_mun_qualif = gpd.GeoDataFrame.from_features(
    municipios_com_qualificacao["features"]
)

# 3. Padronizar a coluna de código no GeoDataFrame
gdf_mun_qualif["CD_MUN"] = (
    gdf_mun_qualif["CD_MUN"]
    .astype(str)
    .str.zfill(7)
)

# 4. Fazer o merge: municípios + indicadores de cursos/concludentes
gdf_mun_qualif_merged = gdf_mun_qualif.merge(
    merged_df_agg,
    on="CD_MUN",
    how="left",   # mantém todos os municípios do GeoJSON
)

# --- colormap para o percentual de conclusão ---
percentual_min = gdf_mun_qualif_merged["percentual_conclusao"].min()
percentual_max = gdf_mun_qualif_merged["percentual_conclusao"].max()

# usa um colormap contínuo de verde (pode trocar por YlOrRd, Blues, etc.)
colormap_conclusao = linear.YlGn_09.scale(percentual_min, percentual_max)
colormap_conclusao.caption = "Percentual de conclusão (%)"


# 5. Converter o GeoDataFrame merged de volta para um dict GeoJSON
municipios_com_qualificacao_merged = json.loads(
    gdf_mun_qualif_merged.to_json()
)

#----------------------------------------------------#


# ---------------- Filtros sincronizados ---------------- #

base_df = merged_df.copy()

# 1) Ler seleções atuais do session_state (antes de criar os widgets)
sel_mun_prev   = st.session_state.get("f_mun", [])
sel_exec_prev  = st.session_state.get("f_exec", [])
sel_curso_prev = st.session_state.get("f_curso", [])
sel_area_prev  = st.session_state.get("f_area", [])

# 2) Montar df_opcoes aplicando essas seleções
df_opcoes = base_df.copy()

if sel_mun_prev:
    df_opcoes = df_opcoes[df_opcoes["Nome_Município"].isin(sel_mun_prev)]
if sel_exec_prev:
    df_opcoes = df_opcoes[df_opcoes["EXECUTORA"].isin(sel_exec_prev)]
if sel_curso_prev:
    df_opcoes = df_opcoes[df_opcoes["CURSO"].isin(sel_curso_prev)]
if sel_area_prev:
    df_opcoes = df_opcoes[
        df_opcoes["ÁREA DO CURSO\n(automático)"].isin(sel_area_prev)
    ]

# 3) Opções disponíveis (AGORA a partir de df_opcoes → sincronizadas)
mun_options   = sorted(df_opcoes["Nome_Município"].dropna().unique().tolist())
exec_options  = sorted(df_opcoes["EXECUTORA"].dropna().unique().tolist())
curso_options = sorted(df_opcoes["CURSO"].dropna().unique().tolist())
area_options  = sorted(df_opcoes["ÁREA DO CURSO\n(automático)"].dropna().unique().tolist())

#-------------- Layout do Streamlit --------------#
st.sidebar.image('icons/neg_color.png', use_container_width=True)
st.sidebar.header("Filtros de Análise")

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

# 5) Verifica se algum filtro foi aplicado
algum_filtro_ativo = any([
    selected_municipios,
    selected_executoras,
    selected_cursos,
    selected_areas_qualificacao,
])

# 6) Aplica as seleções ATUAIS a toda a base para montar df_filtrado
if algum_filtro_ativo:
    df_filtrado = base_df.copy()

    if selected_municipios:
        df_filtrado = df_filtrado[
            df_filtrado["Nome_Município"].isin(selected_municipios)
        ]
    if selected_executoras:
        df_filtrado = df_filtrado[
            df_filtrado["EXECUTORA"].isin(selected_executoras)
        ]
    if selected_cursos:
        df_filtrado = df_filtrado[
            df_filtrado["CURSO"].isin(selected_cursos)
        ]
    if selected_areas_qualificacao:
        df_filtrado = df_filtrado[
            df_filtrado["ÁREA DO CURSO\n(automático)"].isin(
                selected_areas_qualificacao
            )
        ]
else:
    df_filtrado = base_df.copy()
    
    
# Criar indicadores a partir do df_filtrado
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
            (x.sum() / df_filtrado.loc[x.index, "INSCRITOS"].sum()) * 100, 2
        ) if df_filtrado.loc[x.index, "INSCRITOS"].sum() > 0 else 0
    ),
).reset_index()

total_geral_turmas = df_metrics["total_turmas"].sum()
total_geral_vagas = df_metrics["total_vagas_ofertadas"].sum()
total_geral_inscritos = df_metrics["total_inscritos"].sum()
total_geral_desistentes = df_metrics["total_desistentes"].sum()
total_geral_concludentes = df_metrics["total_concludentes"].sum()
percentual_geral_conclusao = round(
    (total_geral_concludentes / total_geral_inscritos) * 100, 2
) if total_geral_inscritos > 0 else 0

    
    
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
    feats = [ # filtra as feições
        f for f in geojson["features"]
        if f["properties"].get("NM_MUN") in lista_mun # verifica se o município está na lista
    ]
    return { # retorna o geojson filtrado
        "type": geojson["type"],
        "name": geojson.get("name", "") + "_filtrado",
        "crs": geojson.get("crs"),
        "features": feats, # lista de feições filtradas
    }

# aplica a filtragem aos geojsons de municípios com qualificação e choropleth
# usando a lista de municípios filtrados {mun_filtrados}
municipios_com_qualificacao_filtrado = filtra_municipios_geojson(
    municipios_com_qualificacao,
    mun_filtrados,
)

municipios_choropleth_filtrado = filtra_municipios_geojson(
    municipios_com_qualificacao_merged,
    mun_filtrados,
)

# Se nenhum filtro ativo, reseta o estado do mapa para o centro e zoom iniciais
if not algum_filtro_ativo:
    st.session_state.map_state = {"center": [-5.3159, -39.2129], "zoom": 7}

#----------------------------------------------------#


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
        "<h4 style='color:#6c91c8; font-weight:500; margin:0'>"
        "Visualização Geoespacial"
        "</h4>",
        unsafe_allow_html=True,
    )
    # Inserir o fragmento do mapa
    @st.fragment
    def mapa_fragment(
        municipios_geojson,
        geojson_cozinhas_csf_filtrado,
        geojson_cozinhas_focais_filtrado,
        municipios_com_qualificacao_filtrado, 
        municipios_choropleth_filtrado, 
        colormap_conclusao
    ):
        
        # Elementos do mapa
        tooltip_municipios = folium.GeoJsonTooltip(
            fields=[
            "NM_MUN",
            # "has_qualif",
            # "total_turmas",
            # "total_concludentes",
            # "percentual_conclusao",
            ],
            aliases=[
                "Município:",
                # "Possui Qualificação:",
                # "Total de turmas:",
                # "Total de concluintes:",
                # "% de conclusão:",
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

        folium.GeoJson(
            municipios_choropleth_filtrado,
            name="Percentual de conclusão",
            style_function=style_conclusao,
            tooltip=folium.GeoJsonTooltip(
                fields=["NM_MUN", "percentual_conclusao"],
                aliases=["Município:", "Percentual conclusão:"],
                localize=True,
            ),
        ).add_to(choropleth_feature_group_indicadores)

        # adiciona a legenda do colormap ao mapa
        colormap_conclusao.add_to(m)

        

        cozinhas_csf_feature_group = folium.FeatureGroup(name="Cozinhas CSF", show=False).add_to(m)
        folium.GeoJson(
            geojson_cozinhas_csf_filtrado,
            name="Cozinhas CSF",
            marker=folium.Marker(icon=cozinha_csf_icon),
            # tooltip=folium.GeoJsonTooltip(
            #     fields=["NOME_USP1", "ID0", "LOTE4"],
            #     aliases=["Nome da Cozinha: ", "ID da Cozinha: ", "Lote: "],
            #     localize=True,
            # ),
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
        colormap_conclusao=colormap_conclusao, 
    )


# --- MÉTRICAS À DIREITA ---

# Custom CSS para estilizar os st.metrics
st.markdown(
    """
    <style>

    /* Aumenta a largura útil da página */
    section.main > div.block-container {
        max-width: 1650px !important;
        padding-top: 1rem !important;
    }

    /* ====== CONTAINER DO CARD (st.metric) ====== */
    div[data-testid="stMetric"] {
        background: radial-gradient(circle at top left, #eef2ff, #f9fafb);
        border-radius: 18px;
        padding: 1.6rem 1.5rem;             /* altura e “respiro” do card */
        min-height: 160px !important;       /* deixa BEM alto */
        width: 100% !important;             /* ocupa toda a coluna */
        box-shadow: 0 7px 18px rgba(15, 23, 42, 0.10);
        border: 1px solid #d4ddff;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    /* ====== RÓTULO DA MÉTRICA ====== */
    div[data-testid="stMetric"] label[data-testid="stMetricLabel"] {
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        white-space: normal !important;      /* permite quebra de linha */
        text-overflow: unset !important;
        overflow: visible !important;
        word-break: break-word;              /* quebra se ficar muito longo */
        color: #4b5563;
        line-height: 1.25;
        margin-bottom: 0.55rem;
    }


    /* ====== VALOR PRINCIPAL ====== */
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 2.0rem !important;        /* um tiquinho menor */
        font-weight: 700;
        color: #111827;
        margin-top: 0.25rem;
        white-space: normal !important;
    }


    /* ====== DELTA (se vier a usar) ====== */
    div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {
        font-size: 0.9rem !important;
    }

    /* ====== espaçamento entre colunas de métricas ====== */
    div[data-testid="column"] {
        padding-left: 0.9rem !important;
        padding-right: 0.9rem !important;
    }


    </style>
    """,
    unsafe_allow_html=True,
)



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
        "<h4 style='color:#6c91c8; font-weight:500; margin:0'>"
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
        st.metric("🏅 Concludentes", format_int_br(total_geral_concludentes))

    with g2c3:
        st.metric("📈 Conclusão geral (%)", format_percent_br(percentual_geral_conclusao))



# Gráficos

col_grafico1, col_grafico2 = st.columns(2)

with col_grafico1:
    # Gráfico de barras dos 10 municípios com mais concludentes
    st.markdown(
        "<h4 style='color:#6c91c8; font-weight:500; margin:0'>"
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
        "<h4 style='color:#6c91c8; font-weight:500; margin:0'>"
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
    


#----------------------------------------------------#
st.write("merged_df")
st.write(merged_df.head())
st.write("merged_df_agg")
st.write(merged_df_agg.head())
st.write("municipios_com_qualificacao_merged")
st.write(municipios_com_qualificacao_merged["features"][0]["properties"])
st.write("df_opcoes")
st.write(df_opcoes.head())
st.write("geojson_cozinhas_csf_filtrado")
st.write(geojson_cozinhas_csf_filtrado)
st.write("features_csf_filtradas")
st.write(features_csf_filtradas)
st.write("df_filtrado")
st.write(df_filtrado.head())
st.write("cozinhas_geojson")
st.write(cozinhas_geojson)


st.write(session_state:=st.session_state)


mun_series = merged_df["mun_upp"]
st.write(mun_series.value_counts(dropna=False))

