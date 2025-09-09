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


# from branca.colormap import StepColormap
from branca.colormap import linear, StepColormap
from branca.element import MacroElement, Template

# df_qualificacao = pd.read_csv('data/agrupado_cursos_concluidos_em_execucao_sem_sebrae.csv')
st.set_page_config(layout="wide")
st.title("Qualificação App - Análise de Cursos e Concludentes")

@st.cache_data
def load_geojson():
    with open("data/municipios_latlon.geojson", "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_data():
    df = pd.read_csv(
        "data/agrupado_cursos_concluidos_em_execucao_sem_sebrae.csv",
        encoding="utf-8",
        sep=",",
        dtype={"Nº LOTE": "string", "Município": "string"},
    )
    # se vier uma coluna sem nome (índice exportado), descarte:
    if df.columns[0].startswith(("Unnamed", ",")) or df.columns[0] == "":
        df = df.drop(columns=[df.columns[0]])
    # normaliza município
    df["Município"] = df["Município"].str.strip().str.upper()
    # garante numéricos
    for col in ("qtd_concludentes", "qtd_turmas"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df

def add_binary_legend(m, title="Legenda", label_on="Com qualificação", label_off="Sem qualificação",
                      color_on="#238b45", color_off="#e0e0e0"):
    html = """
            {% macro html(this, kwargs) %}
            <div style="
                position: fixed; 
                top: 10px; right: 10px; z-index:9999;
                padding: 8px 10px; background: rgba(255,255,255,0.9);
                border: 1px solid #ccc; border-radius: 4px; font-size: 14px;">
                <b>Legenda</b>
                <div style="margin-top:6px;">
                    <div style="display:flex; align-items:center; margin-bottom:4px;">
                        <span style="display:inline-block; width:16px; height:12px; background:#238b45; border:1px solid #333; margin-right:6px;"></span>
                        <span>Com qualificação</span>
                    </div>
                    <div style="display:flex; align-items:center;">
                        <span style="display:inline-block; width:16px; height:12px; background:#e0e0e0; border:1px solid #333; margin-right:6px;"></span>
                        <span>Sem qualificação</span>
                    </div>
                </div>
            </div>
            {% endmacro %}
    """
    MacroElement._template = Template(html)
    m.get_root().add_child(MacroElement())


def build_map(geojson_data, camada, df):
    # m = folium.Map(location=[-5.3159, -39.2129], zoom_start=7, tiles="CartoDB positron")

    if "map_state" not in st.session_state:
        st.session_state.map_state = {"center": [-5.3159, -39.2129], "zoom": 7}

    m = folium.Map(
        location=st.session_state.map_state["center"],
        zoom_start=st.session_state.map_state["zoom"],
        tiles="CartoDB positron"
    )
    
    # ======= Métrica por camada =======
    if camada == "Cursos por Município":
        df_metric = df["Município"].value_counts().reset_index()
        df_metric.columns = ["NM_MUN", "valor"]
    elif camada == "Municípios com Qualificação":
        df_metric = df[["Município"]].drop_duplicates().rename(columns={"Município": "NM_MUN"})
        df_metric["valor"] = 1.0
    elif camada == "Concludentes por Município":
        df_metric = df.groupby("Município")["qtd_concludentes"].sum().reset_index()
        df_metric.columns = ["NM_MUN", "valor"]
    elif camada == "Turmas por Município":
        df_metric = df.groupby("Município")["qtd_turmas"].sum().reset_index()
        df_metric.columns = ["NM_MUN", "valor"]
    else:
        df_metric = pd.DataFrame(columns=["NM_MUN", "valor"])

    # Normalização
    df_metric["NM_MUN"] = df_metric["NM_MUN"].str.strip().str.upper()
    df_metric["valor"] = pd.to_numeric(df_metric["valor"], errors="coerce").fillna(0.0).astype(float)
    valores = dict(zip(df_metric["NM_MUN"], df_metric["valor"]))

    # ======= Escalas por camada (Greens) =======
    colormap = None
    grey = "#e0e0e0"

    if camada == "Municípios com Qualificação":
        # Binário: 1 = verde, 0/ausente = cinza
        # Legenda categórica:
        add_binary_legend(m, "Municípios com Qualificação", color_on="#238b45", color_off=grey)

    elif camada == "Cursos por Município":
        # Bins fixos (QGIS): 1-5, 5-10, 10-20, 20-max
        vmax = max(20.0, float(df_metric["valor"].max()))
        bins = [1.0, 2.0, 5.0, 10.0, 20.0, vmax]
        colormap = linear.Greens_05.to_step(index=bins)
        colormap.caption = "Cursos por Município"
        colormap.add_to(m)

    elif camada == "Concludentes por Município":
        # Bins fixos (QGIS): 0-100, 100-200, 200-500, 500-1000, 1000-max
        vmax = max(1000.0, float(df_metric["valor"].max()))
        bins = [0.0, 100.0, 150.0, 200.0, 500.0, 1000.0, vmax]
        colormap = linear.Greens_06.to_step(index=bins)
        colormap.caption = "Concludentes por Município"
        colormap.add_to(m)

    elif camada == "Turmas por Município":
        # Bins análogos aos de cursos: 1-5, 5-10, 10-20, 20-max
        vmax = max(20.0, float(df_metric["valor"].max()))
        bins = [1.0, 5.0, 10.0, 20.0, vmax]
        colormap = linear.Greens_05.to_step(index=bins)
        colormap.caption = "Turmas por Município"
        colormap.add_to(m)

    # ======= Injeção VALOR no GeoJSON =======
    for feature in geojson_data["features"]:
        nome_mun = feature["properties"]["NM_MUN"].strip().upper()
        feature["properties"]["VALOR"] = float(valores.get(nome_mun, 0.0))

    # ======= Estilo =======
    def style_fn(feat):
        nome = feat["properties"]["NM_MUN"].strip().upper()
        v = float(valores.get(nome, 0.0))

        if camada == "Municípios com Qualificação":
            fill = "#238b45" if v >= 1.0 else grey
        else:
            if colormap is None:
                fill = grey
            else:
                # Para Cursos/Turmas: 0 vira cinza para destacar ausência
                if camada in ("Cursos por Município", "Turmas por Município") and v < 1.0:
                    fill = grey
                else:
                    fill = colormap(v)

        return {"fillColor": fill, "color": "#333", "weight": 0.7, "fillOpacity": 0.75}

    geo = folium.GeoJson(
        geojson_data,
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(
            fields=["NM_MUN", "VALOR"],
            aliases=["Município:", f"{camada}:"],
            localize=True, labels=True, sticky=True, max_width=800
        ),
        popup=folium.GeoJsonPopup(             # <<-- ESSENCIAL
            fields=["NM_MUN", "VALOR"],
            aliases=["Município:", f"{camada}:"],
            localize=True, labels=True, max_width=400
        ),
        highlight_function=lambda f: {"weight": 2, "color": "#000", "fillOpacity": 0.85},
        popup_keep_highlighted=True,
    )
    geo.add_to(m)


    return m, geo



# Sidebar
st.sidebar.image('icons/neg_color.png', use_container_width=True)
camada = st.sidebar.radio("Selecione a camada:", 
                          ["Municípios com Qualificação", "Cursos por Município", "Concludentes por Município", "Turmas por Município"])

# Dados
geojson_raw = load_geojson()
geojson_data = copy.deepcopy(geojson_raw)  # trabalha numa cópia

# GeoDataFrame para lookup ponto-em-polígono (EPSG:4326)
gdf_muns = gpd.GeoDataFrame.from_features(geojson_raw, crs="EPSG:4326")[["NM_MUN", "geometry"]]
# índice espacial para acelerar
_ = gdf_muns.sindex

df_qualificacao = load_data()


df_municipios_unicos = df_qualificacao['Município'].unique()
df_lotes_unicos = df_qualificacao['Nº LOTE'].unique()

qtd_cursos_por_municipio = df_qualificacao['Município'].value_counts()
qtd_concludentes_por_municipio = df_qualificacao.groupby('Município')['qtd_concludentes'].sum().reset_index()
qtd_turmas_por_municipio = df_qualificacao.groupby('Município')['qtd_turmas'].sum().reset_index().sort_values(by='qtd_turmas', ascending=False)

@st.dialog("Detalhes do Município", width="large")
def show_municipio_dialog(municipio: str, df_base: pd.DataFrame, camada: str):
    # filtra base (seu df_qualificacao já está normalizado em UPPER)
    df_mun = df_base[df_base["Município"].str.upper().eq(str(municipio).upper())].copy()

    st.subheader(municipio.title())

    # KPIs rápidos
    c1, c2, c3 = st.columns(3)
    c1.metric("Cursos distintos", df_mun["CURSO"].nunique())
    c2.metric("Total de turmas", int(df_mun["qtd_turmas"].sum()))
    c3.metric("Total de concluintes", int(df_mun["qtd_concludentes"].sum()))

    # Agregações por curso
    agg = (df_mun.groupby("CURSO", as_index=False)
                 .agg(concludentes=("qtd_concludentes", "sum"),
                      turmas=("qtd_turmas", "sum"))
          )
    # ordenação por concludentes
    agg = agg.sort_values("concludentes", ascending=False)

    st.markdown("#### Concludentes por curso")
    ch1 = (alt.Chart(agg)
             .mark_bar()
             .encode(
                 x=alt.X("concludentes:Q", title="Concludentes"),
                 y=alt.Y("CURSO:N", sort="-x", title="Curso"),
                 tooltip=["CURSO", "concludentes", "turmas"]
             )
             .properties(height=400)
          )
    st.altair_chart(ch1, use_container_width=True)  # Altair no Streamlit :contentReference[oaicite:7]{index=7}

    st.markdown("#### Turmas por curso")
    ch2 = (alt.Chart(agg)
             .mark_bar()
             .encode(
                 x=alt.X("turmas:Q", title="Turmas"),
                 y=alt.Y("CURSO:N", sort="-x", title="Curso"),
                 tooltip=["CURSO", "turmas", "concludentes"]
             )
             .properties(height=400)
          )
    st.altair_chart(ch2, use_container_width=True)

    # Tabela detalhada (você pode escolher as colunas mais úteis)
    st.markdown("#### Detalhamento dos cursos (linhas originais)")
    cols = [c for c in df_mun.columns if c in ["Nº LOTE", "Município", "CURSO", "qtd_turmas", "qtd_inscritos", "qtd_vagas", "qtd_concludentes"]]
    st.dataframe(df_mun[cols], use_container_width=True)

    # Download do recorte
    csv_bytes = df_mun[cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Baixar CSV do município",
        data=csv_bytes,
        file_name=f"{municipio}_detalhamento.csv",
        mime="text/csv"
    )  # download_button doc :contentReference[oaicite:8]{index=8}

    # Botão fechar (opcional): st.rerun() fecha o modal programaticamente. :contentReference[oaicite:9]{index=9}
    if st.button("Fechar"):
        st.session_state.mun_clicked = None   # <<< zera a trava
        st.rerun()


# Mapa
st.markdown("### Mapa Interativo")
m, geo = build_map(geojson_data, camada, df_qualificacao)
st_data = st_folium(
    m,
    width="100%",
    height=600,
    use_container_width=True,
    returned_objects=["last_object_clicked"], # para evitar reruns desnecess
    feature_group_to_add=[geo]
)

# ---- captura do clique ----
if "mun_clicked" not in st.session_state:
    st.session_state.mun_clicked = None

evt = (st_data or {}).get("last_object_clicked")
# st.write("DEBUG - Retorno bruto do last_object_clicked:", evt)

mun = valor = None

if evt and isinstance(evt, dict):
    # 1) Tente obter diretamente das properties (se vierem)
    props = evt.get("properties") if "properties" in evt else evt
    mun = (props or {}).get("NM_MUN")
    valor = (props or {}).get("VALOR")

    # 2) Se NÃO veio NM_MUN, faça ponto-em-polígono com GeoPandas
    if not mun and "lat" in evt and "lng" in evt:
        pt = gpd.points_from_xy([evt["lng"]], [evt["lat"]], crs="EPSG:4326")[0]
        # pequena tolerância para cliques na borda (aprox. ~10m)
        poly = gdf_muns[gdf_muns.intersects(pt.buffer(0.0001))]
        if not poly.empty:
            mun = str(poly.iloc[0]["NM_MUN"]).strip().upper()
            # como você já injeta VALOR por camada, podemos recuperá-lo do dict 'valores'
            # mas aqui é mais simples computar pela sua df_metric/df_qualificacao:
            # use o mesmo cálculo do build_map, ou só ignore 'valor' no toast/modal se não precisar
            valor = None

# st.write("DEBUG - Município resolvido:", mun, " | Valor:", valor)

if mun and mun != st.session_state.mun_clicked:
    st.session_state.mun_clicked = mun
    st.toast(f"Município: {mun}" + (f" — {camada}: {valor}" if valor is not None else ""), icon="✅")
    show_municipio_dialog(mun, df_qualificacao, camada)



