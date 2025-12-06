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
import acesso_planilha
from acesso_planilha import carregar_google_sheet_aba
from google_sheets_api import carregar_google_sheet_por_aba

link = "https://docs.google.com/spreadsheets/d/1M2huy5RGW5D28zWRnBiHI4kSGWZNi5ejyygnxQjx7uo/edit?gid=0#gid=0"

# Coloque aqui o NOME EXATO da aba, como aparece no Google Sheets
nome_aba = "Compilado"  # exemplo; troque pelo nome real da aba
intervalo = "A:AN"       # lê todas as colunas da aba; ajuste se quiser

df = carregar_google_sheet_por_aba(link, nome_aba, intervalo)

# Configurações iniciais do Streamlit
st.set_page_config(layout="wide")


# Função para carregar CSS externo
def load_css(file_name):
    with open(file_name, encoding="utf-8") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Carrega o CSS global
load_css("styles.css")





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
        "data2/compilado_novos_lotes_merge_nomes_cozinhas_05122025.csv",
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


# ---------------- Filtros sincronizados ---------------- #

base_df = merged_df.copy()
col_area = "ÁREA DO CURSO\n(automático)"

def filtrar_dados(df, municipios, executoras, cursos, areas):
    """Filtra o DataFrame com base nas listas de opções selecionadas."""
    df_result = df.copy()
    if municipios:
        df_result = df_result[df_result["Nome_Município"].isin(municipios)]
    if executoras:
        df_result = df_result[df_result["EXECUTORA"].isin(executoras)]
    if cursos:
        df_result = df_result[df_result["CURSO"].isin(cursos)]
    if areas:
        df_result = df_result[df_result[col_area].isin(areas)]
    return df_result

# 1) Ler seleções atuais do session_state (antes de criar os widgets)
sel_mun_prev   = st.session_state.get("f_mun", [])
sel_exec_prev  = st.session_state.get("f_exec", [])
sel_curso_prev = st.session_state.get("f_curso", [])
sel_area_prev  = st.session_state.get("f_area", [])

# 2) Montar df_opcoes aplicando essas seleções
df_opcoes = filtrar_dados(base_df, sel_mun_prev, sel_exec_prev, sel_curso_prev, sel_area_prev)

# 3) Opções disponíveis (AGORA a partir de df_opcoes → sincronizadas)
mun_options   = sorted(df_opcoes["Nome_Município"].dropna().unique().tolist())
exec_options  = sorted(df_opcoes["EXECUTORA"].dropna().unique().tolist())
curso_options = sorted(df_opcoes["CURSO"].dropna().unique().tolist())
area_options  = sorted(df_opcoes[col_area].dropna().unique().tolist())

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
    df_filtrado = filtrar_dados(
        base_df, 
        selected_municipios, 
        selected_executoras, 
        selected_cursos, 
        selected_areas_qualificacao
    )
else:
    df_filtrado = base_df.copy()
#----------------------------------------------------#


# -------------------------------------------------------------
# Cálculo da taxa relativa de conclusão
# -------------------------------------------------------------
if "VAGAS OFERTADAS" in df_filtrado.columns and "CONCLUDENTES" in df_filtrado.columns:

    df_filtrado["TAXA_CONCLUSAO"] = (
        df_filtrado["CONCLUDENTES"] / df_filtrado["VAGAS OFERTADAS"]
    ) * 100

    # Evitar valores >100% ou negativos
    df_filtrado["TAXA_CONCLUSAO"] = df_filtrado["TAXA_CONCLUSAO"].clip(0, 100)

else:
    st.warning("Colunas VAGAS OFERTADAS ou CONCLUDENTES não foram encontradas no DataFrame.")



    
# -------------------------------------------------------------------
# Tratamento de datas e filtro temporal
# -------------------------------------------------------------------

# Garante que as colunas de data existem antes de mexer nelas
col_data_inicio = "DATA INÍCIO"
col_data_termino = "DATA TÉRMINO"

if col_data_inicio in df_filtrado.columns and col_data_termino in df_filtrado.columns:
    # Converter para datetime (uma vez por rerun)
    df_filtrado[col_data_inicio] = pd.to_datetime(
        df_filtrado[col_data_inicio],
        dayfirst=True,
        errors="coerce",
    )
    df_filtrado[col_data_termino] = pd.to_datetime(
        df_filtrado[col_data_termino],
        dayfirst=True,
        errors="coerce",
    )

    # Criar campos derivados
    df_filtrado["DURACAO_DIAS"] = (
        df_filtrado[col_data_termino] - df_filtrado[col_data_inicio]
    ).dt.days

    df_filtrado["ANO_INICIO"] = df_filtrado[col_data_inicio].dt.year
    df_filtrado["MES_INICIO"] = df_filtrado[col_data_inicio].dt.month
    df_filtrado["ANO_MES_INICIO"] = df_filtrado[col_data_inicio].dt.to_period("M")

    # ------------------- Filtro temporal na sidebar -------------------
    datas_validas = df_filtrado[col_data_inicio].dropna()

    if not datas_validas.empty:
        min_data = datas_validas.min().date()
        max_data = datas_validas.max().date()

        periodo = st.sidebar.date_input(
            "Período de início dos cursos:",
            value=(min_data, max_data),
            help="Filtra os cursos pelo intervalo de DATA INÍCIO.",
        )

        # O date_input pode retornar uma tupla (início, fim) ou uma única data
        if isinstance(periodo, tuple) and len(periodo) == 2:
            data_ini, data_fim = periodo
        else:
            data_ini = periodo
            data_fim = periodo

        data_ini = pd.to_datetime(data_ini)
        data_fim = pd.to_datetime(data_fim)

        df_filtrado = df_filtrado[
            (df_filtrado[col_data_inicio] >= data_ini)
            & (df_filtrado[col_data_inicio] <= data_fim)
        ]
else:
    st.warning(
        "Colunas de data não encontradas no df_filtrado. "
        "Verifique os nomes das colunas de DATA INÍCIO e DATA TÉRMINO."
    )

    
    
df_filtrado["DATA INÍCIO"] = pd.to_datetime(df_filtrado["DATA INÍCIO"], dayfirst=True, errors="coerce")
df_filtrado["DATA TÉRMINO"] = pd.to_datetime(df_filtrado["DATA TÉRMINO"], dayfirst=True, errors="coerce")
df_filtrado["ANO_INICIO"] = df_filtrado["DATA INÍCIO"].dt.year
df_filtrado["MES_INICIO"] = df_filtrado["DATA INÍCIO"].dt.month
df_filtrado["DURACAO_DIAS"] = (df_filtrado["DATA TÉRMINO"] - df_filtrado["DATA INÍCIO"]).dt.days

    
    
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
            (x.sum() / df_filtrado.loc[x.index, "INSCRITOS"].sum()) * 100, 2
        ) if df_filtrado.loc[x.index, "INSCRITOS"].sum() > 0 else 0
    ),
).reset_index()

total_geral_turmas_exec = df_metrics_exec["total_turmas"].sum()
total_geral_vagas_exec = df_metrics_exec["total_vagas_ofertadas"].sum()
total_geral_inscritos_exec = df_metrics_exec["total_inscritos"].sum()
total_geral_desistentes_exec = df_metrics_exec["total_desistentes"].sum()
total_geral_concludentes_exec = df_metrics_exec["total_concludentes"].sum()
percentual_geral_conclusao_exec = round(
    (total_geral_concludentes_exec / total_geral_inscritos_exec) * 100, 2
) if total_geral_inscritos_exec > 0 else 0
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


#-------------- Gráfico temporal --------------#

df_temporal = (
    df_filtrado.groupby(df_filtrado["DATA INÍCIO"].dt.to_period("M"))
    .size()
    .reset_index(name="qtd_turmas")
)

df_temporal["DATA"] = df_temporal["DATA INÍCIO"].dt.to_timestamp()

chart = (
    alt.Chart(df_temporal)
    .mark_line(point=True)
    .encode(
        x=alt.X("DATA:T", title="Período"),
        y=alt.Y("qtd_turmas:Q", title="Turmas iniciadas"),
        tooltip=["DATA", "qtd_turmas"]
    )
    .properties(height=300, title="Turmas iniciadas por mês")
)

st.altair_chart(chart, use_container_width=True)
#----------------------------------------------------#





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
    
    
    
col_grafico3, col_grafico4 = st.columns(2)

with col_grafico3:
    # Gráfico de barras dos 10 municípios com mais concludentes
    st.markdown(
        "<h4 style='color:#6c91c8; font-weight:500; margin:0'>"
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
        "<h4 style='color:#6c91c8; font-weight:500; margin:0'>"
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
    


#----------------------------------------------------#


