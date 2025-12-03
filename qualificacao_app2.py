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
import geopandas as gpd
from folium.plugins import Draw

from branca.colormap import linear
from branca.element import MacroElement, Template
from merge_id_plataforma import merge_id_plataforma

# Configurações iniciais do Streamlit
st.set_page_config(layout="wide")
    


# — Fonte global: Space Grotesk —
st.markdown(
    """
    <style>
    /* carrega a fonte com todos os pesos disponíveis */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

    /* aplica globalmente na app */
    .stApp, .stAppViewContainer, .main, .block-container,
    h1, h2, h3, h4, h5, h6,
    p, div, span, label, li, a, button, input, textarea, select {
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

# 1 - Separando dados para filtros

# 1.1 - lista de municípios
municipios = municipios_geojson["features"]
municipios_list = [municipio["properties"]["NM_MUN"] for municipio in municipios]
municipios_list.sort()

# 1.2 - lista de entidades executoras
executoras_list = cursos_df["EXECUTORA"].dropna().unique().tolist()
executoras_list.sort()

# 1.3 - Lista de Cursos
cursos_list = cursos_df["CURSO"].dropna().unique().tolist()
cursos_list.sort()

# 1.4 - Lista de áreas de qualificação
areas_qualificacao_list = cursos_df["ÁREA DO CURSO\n(automático)"].dropna().unique().tolist()
areas_qualificacao_list.sort()



# Mesclar dados dos cursos com dados da plataforma
merged_df = merge_id_plataforma(cursos_df, df_kitchen)


merged_df_agg = merged_df.groupby(
    ["Código Município Completo", "Nome_Município", 'Nº LOTE 2025']
).agg(
    total_turmas=pd.NamedAgg(column="CURSO", aggfunc="nunique"),
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




# filtrar o geojson das cozinhas para manter apenas as cozinhas presentes no dataframe mesclado
codigos_cozinhas_focais = set(merged_df["id"].dropna().astype(int).astype(str))
features_filtradas = []
for feature in cozinhas_geojson['features']:
    feature_id = feature['properties'].get('ID0')
    if feature_id and feature_id in codigos_cozinhas_focais:
        features_filtradas.append(feature)

geojson_filtrado = {
    'type': cozinhas_geojson['type'],
    'name': cozinhas_geojson['name'] + '_filtered',
    'crs': cozinhas_geojson['crs'],
    'features': features_filtradas
}

#----------------------------------------------------#





#-------------- Layout do Streamlit --------------#
# Filtros laterais
st.sidebar.header("Filtros de Análise")
# Filtro por município
selected_municipios = st.sidebar.multiselect(
    "Selecione os municípios:",
    options=municipios_list,
    default=None,
    placeholder ="Municípios"
)

# Filtro por entidade executora
selected_executoras = st.sidebar.multiselect(
    "Selecione as entidades executoras:",
    options=executoras_list,
    default=None,
    placeholder="Entidades Executoras"
)

# Filtro por curso
selected_cursos = st.sidebar.multiselect(
    "Selecione os cursos:",
    options=cursos_list,
    default=None,
    placeholder="Cursos"
)

# Filtro por área de qualificação
selected_areas_qualificacao = st.sidebar.multiselect(
    "Selecione as áreas de qualificação:",
    options=areas_qualificacao_list,
    default=None,
    placeholder="Áreas de Qualificação"
)

# Mapa interativo
st.markdown(
    "<h2 style='color:#6c91c8; font-weight:600; margin:0'>"
    "Mapa Interativo de Cursos por Município"
    "</h2>",
    unsafe_allow_html=True,
)




# Inserir o fragmento do mapa
@st.fragment
def mapa_fragment(
    municipios_geojson,
    cozinhas_geojson,
    geojson_filtrado,
    selected_municipios,
    municipios_com_qualificacao, 
    municipios_com_qualificacao_merged, 
    colormap_conclusao
):
    # --- opcional: filtrar municípios pelo multiselect ---
    # if selected_municipios and len(selected_municipios) < len(municipios_list):
    #     features_filtradas_mun = [
    #         f
    #         for f in municipios_geojson["features"]
    #         if f["properties"]["NM_MUN"] in selected_municipios
    #     ]
    #     municipios_geojson_filtrado = {
    #         **municipios_geojson,
    #         "features": features_filtradas_mun,
    #     }
    # else:
    #     municipios_geojson_filtrado = municipios_geojson

    # Elementos do mapa
    tooltip_municipios = folium.GeoJsonTooltip(
        fields=["NM_MUN", 'has_qualif'],
        aliases=["Município:", 'Possui Qualificação:'],
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

    # municipios_feature_group = folium.FeatureGroup(name="Municípios").add_to(m)
    # folium.GeoJson(
    #     municipios_geojson_filtrado,
    #     name="Municípios",
    #     tooltip=tooltip_municipios,
    # ).add_to(municipios_feature_group)
    
    
    municipios_qualif_feature_group = folium.FeatureGroup(name="Municípios com Qualificação").add_to(m)
    folium.GeoJson(
        municipios_com_qualificacao,
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
    
    
    # municipios_qualif_feature_group = folium.FeatureGroup(
    #     name="Municípios com Qualificação"
    # ).add_to(m)

    # folium.GeoJson(
    #     municipios_com_qualificacao,
    #     name="Municípios com Qualificação",
    #     style_function=lambda feature: {
    #         "fillColor": "#4e90cc" 
    #         if feature["properties"]["has_qualif"] == 1 
    #         else '#f7e350',
    #         'color': 'red',
    #         'weight': 1,
    #         'dashArray': '5, 5',
    #         'fillOpacity': 0.6,
    #     },
    #     tooltip=tooltip_municipios,
    # ).add_to(municipios_qualif_feature_group)

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
        municipios_com_qualificacao_merged,
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
        cozinhas_geojson,
        name="Cozinhas CSF",
        marker=folium.Marker(icon=cozinha_csf_icon),
        tooltip=folium.GeoJsonTooltip(
            fields=["NOME_USP1", "ID0", "LOTE4"],
            aliases=["Nome da Cozinha: ", "ID da Cozinha: ", "Lote: "],
            localize=True,
        ),
    ).add_to(cozinhas_csf_feature_group)

    cozinhas_focais_feature_group = folium.FeatureGroup(name="Cozinhas Focais", show=False).add_to(m)
    folium.GeoJson(
        geojson_filtrado,
        name="Cozinhas Focais",
        marker=folium.Marker(icon=cozinha_focal_icon),
        tooltip=folium.GeoJsonTooltip(
            fields=["NOME_USP1", "ID0", "LOTE4"],
            aliases=["Nome da Cozinha: ", "ID da Cozinha: ", "Lote: "],
            localize=True,
        ),
    ).add_to(cozinhas_focais_feature_group)

    folium.LayerControl().add_to(m)

    # >>> AQUI é onde a mágica acontece <<<
    st_data = st_folium(
        m,
        width=725,
        height=600,
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
    cozinhas_geojson=cozinhas_geojson,
    geojson_filtrado=geojson_filtrado,
    selected_municipios=selected_municipios,
    municipios_com_qualificacao=municipios_com_qualificacao,
    municipios_com_qualificacao_merged=municipios_com_qualificacao_merged,
    colormap_conclusao=colormap_conclusao, 
)

#----------------------------------------------------#
st.write(merged_df.head())
st.write(merged_df_agg.head())
st.write(municipios_com_qualificacao_merged["features"][0]["properties"])

