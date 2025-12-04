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
# municipios = municipios_geojson["features"]
# municipios_list = [municipio["properties"]["NM_MUN"] for municipio in municipios]
# municipios_list.sort()

# 1.2 - lista de entidades executoras
# executoras_list = cursos_df["EXECUTORA"].dropna().unique().tolist()
# executoras_list.sort()

# 1.3 - Lista de Cursos
# cursos_list = cursos_df["CURSO"].dropna().unique().tolist()
# cursos_list.sort()

# 1.4 - Lista de áreas de qualificação
# areas_qualificacao_list = cursos_df["ÁREA DO CURSO\n(automático)"].dropna().unique().tolist()
# areas_qualificacao_list.sort()



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
st.markdown(
    "<h2 style='color:#6c91c8; font-weight:600; margin:0'>"
    "Mapa Interativo de Cursos por Município"
    "</h2>",
    unsafe_allow_html=True,
)

# Ajustar a grade de layout
col_mapa, col_metricas = st.columns([2, 1])

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

        # municipios_feature_group = folium.FeatureGroup(name="Municípios").add_to(m)
        # folium.GeoJson(
        #     municipios_geojson_filtrado,
        #     name="Municípios",
        #     tooltip=tooltip_municipios,
        # ).add_to(municipios_feature_group)
        
        
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
        geojson_cozinhas_csf_filtrado=geojson_cozinhas_csf_filtrado,
        geojson_cozinhas_focais_filtrado=geojson_cozinhas_focais_filtrado,
        municipios_com_qualificacao_filtrado=municipios_com_qualificacao_filtrado,
        municipios_choropleth_filtrado=municipios_choropleth_filtrado,
        colormap_conclusao=colormap_conclusao, 
    )


# --- MÉTRICAS À DIREITA ---
with col_metricas:
    st.subheader("Indicadores")

    # Primeira linha com 3 métricas
    m1, m2, m3 = st.columns(3)
    m1.metric("Temperatura", "70 °F", "1.2 °F")
    m2.metric("Vento", "9 mph", "-8%")
    m3.metric("Umidade", "86%", "4%")

    # Segunda linha com 3 métricas
    m4, m5, m6 = st.columns(3)
    m4.metric("Pressão", "30.34 inHg", "-2 inHg")
    m5.metric("Visibilidade", "10 km", "1 km")
    m6.metric("Índice UV", "5", "1")


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

