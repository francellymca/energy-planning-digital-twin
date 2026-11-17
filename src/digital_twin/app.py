from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURES_PATH = (
    PROJECT_ROOT
    / "data/processed/modelagem/features_modelagem_2016_2025.csv"
)

SCENARIOS_PATH = (
    PROJECT_ROOT
    / "data/processed/cenarios/cenarios_consumo_2026_2030.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "data/processed/cenarios/resumo_cenarios_2026_2030.csv"
)

SARIMA_METRICS_PATH = (
    PROJECT_ROOT
    / "data/processed/modelagem/metricas_sarima_2024_2025.csv"
)

SARIMA_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "data/processed/modelagem/sarima_previsoes_2024_2025.csv"
)

STATE_SHAP_PATH = (
    PROJECT_ROOT
    / "data/processed/modelagem/shap_importancia_por_uf_xgboost.csv"
)

SHAP_PATH = (
    PROJECT_ROOT
    / "data/processed/modelagem/shap_importancia_global_xgboost.csv"
)

BRAZIL_STATES_PATH = (
    PROJECT_ROOT
    / "data/external/geospatial/"
    "estados_brasil_ibge.geojson"
)

UF_NAMES = {
    "BA": "Bahia",
    "MG": "Minas Gerais",
    "PR": "Paraná",
    "RJ": "Rio de Janeiro",
    "RS": "Rio Grande do Sul",
    "SP": "São Paulo",
}

UF_IBGE_CODES = {
    "BA": "29",
    "MG": "31",
    "PR": "41",
    "RJ": "33",
    "RS": "43",
    "SP": "35",
}


IBGE_STATE_NAMES = {
    "11": ("RO", "Rondônia"),
    "12": ("AC", "Acre"),
    "13": ("AM", "Amazonas"),
    "14": ("RR", "Roraima"),
    "15": ("PA", "Pará"),
    "16": ("AP", "Amapá"),
    "17": ("TO", "Tocantins"),
    "21": ("MA", "Maranhão"),
    "22": ("PI", "Piauí"),
    "23": ("CE", "Ceará"),
    "24": ("RN", "Rio Grande do Norte"),
    "25": ("PB", "Paraíba"),
    "26": ("PE", "Pernambuco"),
    "27": ("AL", "Alagoas"),
    "28": ("SE", "Sergipe"),
    "29": ("BA", "Bahia"),
    "31": ("MG", "Minas Gerais"),
    "32": ("ES", "Espírito Santo"),
    "33": ("RJ", "Rio de Janeiro"),
    "35": ("SP", "São Paulo"),
    "41": ("PR", "Paraná"),
    "42": ("SC", "Santa Catarina"),
    "43": ("RS", "Rio Grande do Sul"),
    "50": ("MS", "Mato Grosso do Sul"),
    "51": ("MT", "Mato Grosso"),
    "52": ("GO", "Goiás"),
    "53": ("DF", "Distrito Federal"),
}

SCENARIO_ORDER = [
    "Cenário de referência",
    "Menor demanda (P25)",
    "Maior demanda (P75)",
]


st.set_page_config(
    page_title="Gêmeo Digital para Planejamento Energético",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1680px;
            padding-top: 1.25rem;
            padding-bottom: 1rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid
                rgba(128, 128, 128, 0.22);
        }

        [data-testid="stSidebar"]
        .block-container {
            padding-top: 1.5rem;
        }

        h1 {
            letter-spacing: -0.025em;
        }

        [data-testid="stMetric"] {
            background:
                var(--secondary-background-color);
            border: 1px solid
                rgba(128, 128, 128, 0.22);
            border-radius: 12px;
            padding: 0.85rem 1rem;
            box-shadow:
                0 3px 12px
                rgba(0, 0, 0, 0.05);
        }

        [data-testid="stTabs"] button {
            font-weight: 650;
            padding-left: 1.15rem;
            padding-right: 1.15rem;
        }

        [data-testid="stPlotlyChart"] {
            border: 1px solid
                rgba(128, 128, 128, 0.22);
            border-radius: 12px;
            padding: 0.35rem;
        }

        div[data-testid="stAlert"] {
            border-radius: 10px;
        }

        hr {
            border-color:
                rgba(128, 128, 128, 0.22);
        }

        footer {
            visibility: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data
def load_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    history = pd.read_csv(
        FEATURES_PATH,
        parse_dates=["mes"],
    )

    scenarios = pd.read_csv(
        SCENARIOS_PATH,
        parse_dates=["mes"],
    )

    summary = pd.read_csv(
        SUMMARY_PATH
    )

    metrics = pd.read_csv(
        SARIMA_METRICS_PATH
    )

    predictions = pd.read_csv(
        SARIMA_PREDICTIONS_PATH,
        parse_dates=["mes"],
    )

    state_shap = pd.read_csv(
        STATE_SHAP_PATH
    )

    return (
        history,
        scenarios,
        summary,
        metrics,
        predictions,
        state_shap,
    )

@st.cache_data
def load_brazil_states_geojson() -> dict:
    return json.loads(
        BRAZIL_STATES_PATH.read_text(
            encoding="utf-8-sig"
        )
    )

(
    history,
    scenarios,
    summary,
    metrics,
    predictions,
    state_shap,
) = load_data()

brazil_states_geojson = (
    load_brazil_states_geojson()
)

st.title("⚡ Gêmeo Digital para Planejamento Energético")
st.caption(
    "Simulação, previsão e análise de cenários para apoiar "
    "decisões no planejamento energético estadual."
)

def build_study_area_map(
    geojson_data: dict,
    selected_uf: str,
    dark_mode: bool,
):
    if dark_mode:
        background_color = "#0E1724"
        unselected_color = "#263547"
        study_color = "#4285D4"
        selected_color = "#76B7FF"
        border_color = "#A8B6C5"
    else:
        background_color = "#FFFFFF"
        unselected_color = "#D6DEE8"
        study_color = "#4F9DE8"
        selected_color = "#123E73"
        border_color = "#FFFFFF"

    selected_code = str(UF_IBGE_CODES[selected_uf])

    study_codes = {
        str(code)
        for code in UF_IBGE_CODES.values()
    }

    fig, axis = plt.subplots(
        figsize=(3.0, 3.4),
        facecolor=background_color,
    )

    axis.set_facecolor(background_color)

    for feature in geojson_data["features"]:
        state_code = str(
            feature["properties"]["codarea"]
        )

        if state_code == selected_code:
            fill_color = selected_color
        elif state_code in study_codes:
            fill_color = study_color
        else:
            fill_color = unselected_color

        geometry = feature["geometry"]
        geometry_type = geometry["type"]
        coordinates = geometry["coordinates"]

        if geometry_type == "Polygon":
            polygons = [coordinates]
        elif geometry_type == "MultiPolygon":
            polygons = coordinates
        else:
            continue

        for polygon in polygons:
            exterior_ring = polygon[0]

            patch = MplPolygon(
                exterior_ring,
                closed=True,
                facecolor=fill_color,
                edgecolor=border_color,
                linewidth=0.55,
            )

            axis.add_patch(patch)

    axis.set_xlim(-74.5, -32.5)
    axis.set_ylim(-34.5, 6.0)
    axis.set_aspect("equal", adjustable="box")
    axis.axis("off")

    fig.subplots_adjust(
        left=0,
        right=1,
        top=1,
        bottom=0,
    )

    return fig

with st.sidebar:
    st.header("Configuração da simulação")

    selected_uf = st.selectbox(
        "Unidade da Federação",
        options=list(UF_NAMES),
        format_func=lambda uf: f"{UF_NAMES[uf]} ({uf})",
        index=list(UF_NAMES).index("RS"),
    )
    
    map_rows: list[dict[str, object]] = []

    selected_ibge_code = UF_IBGE_CODES[
        selected_uf
    ]

    analyzed_codes = set(
        UF_IBGE_CODES.values()
    )

    for ibge_code, (
        state_abbreviation,
        state_name,
    ) in IBGE_STATE_NAMES.items():
        if ibge_code == selected_ibge_code:
            highlight_level = 2
        elif ibge_code in analyzed_codes:
            highlight_level = 1
        else:
            highlight_level = 0

        map_rows.append(
            {
                "codigo_ibge": ibge_code,
                "uf": state_abbreviation,
                "estado": state_name,
                "destaque": highlight_level,
            }
        )

    theme_type = st.context.theme.type
    is_dark_mode = theme_type == "dark"

    map_figure = build_study_area_map(
        geojson_data=brazil_states_geojson,
        selected_uf=selected_uf,
        dark_mode=is_dark_mode,
    )

    st.pyplot(
        map_figure,
        use_container_width=True,        
    )

    plt.close(map_figure)

    st.caption(
        "Estados coloridos compõem o recorte "
        "territorial do estudo."
    )

    selected_scenario = st.radio(
        "Cenário de demanda",
        options=SCENARIO_ORDER,
        index=0,
    )
    
    selected_year = st.select_slider(
        "Ano de análise",
        options=list(range(2026, 2031)),
        value=2026,
    )
    
    scenario_descriptions = {
        "Cenário de referência": (
            "Trajetória central produzida pelo SARIMA."
        ),
        "Menor demanda (P25)": (
            "Trajetória de menor crescimento baseada "
            "no percentil 25 histórico."
        ),
        "Maior demanda (P75)": (
            "Trajetória de maior crescimento baseada "
            "no percentil 75 histórico."
        ),
    }

    st.caption(
        scenario_descriptions[selected_scenario]
    )

    st.divider()

    st.success("Dados carregados")
    st.caption("Período histórico: 2016–2025")
    st.caption("Horizonte de projeção: 2026–2030")


state_history = (
    history.loc[
        history["uf"].eq(selected_uf)
        & history["ano"].between(2022, 2025),
        ["mes", "consumo_mwh"],
    ]
    .sort_values("mes")
    .copy()
)

state_scenario = (
    scenarios.loc[
        scenarios["uf"].eq(selected_uf)
        & scenarios["cenario_rotulo"].eq(selected_scenario)
    ]
    .sort_values("mes")
    .copy()
)

state_summary = summary.loc[
    summary["uf"].eq(selected_uf)
    & summary["cenario_rotulo"].eq(
        selected_scenario
    )
    & summary["ano"].eq(selected_year)
].iloc[0]

state_metric = metrics.loc[
    metrics["uf"].eq(selected_uf)
].iloc[0]


def format_number_br(
    value: float,
    decimals: int = 1,
) -> str:
    formatted = f"{value:,.{decimals}f}"
    return (
        formatted
        .replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )


history_gwh = (
    state_history["consumo_mwh"] / 1_000
)

reference_gwh = (
    state_scenario["referencia_sarima_mwh"] / 1_000
)

scenario_gwh = (
    state_scenario["consumo_cenario_mwh"] / 1_000
)

lower_gwh = (
    state_scenario["limite_inferior_95_mwh"] / 1_000
)

upper_gwh = (
    state_scenario["limite_superior_95_mwh"] / 1_000
)


scenario_colors = {
    "Cenário de referência": "#16A6A1",
    "Menor demanda (P25)": "#8B98A1",
    "Maior demanda (P75)": "#F28C28",
}

active_color = scenario_colors[selected_scenario]

overview_tab, analysis_tab, data_tab = st.tabs(
    [
        "📈 Visão geral",
        "🧠 Desempenho e XAI",
        "🗂️ Dados e metodologia",
    ]
)


with overview_tab:
    
    st.subheader(
        f"Estado selecionado: "
        f"{UF_NAMES[selected_uf]} ({selected_uf})"
    )

    st.caption(
        "Consumo mensal observado entre 2022 e 2025 "
        "e projeções de referência e cenários entre "
        "2026 e 2030."
    )


    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=state_history["mes"],
            y=history_gwh,
            mode="lines+markers",
            name="Histórico observado",
            line={
                "color": "#0B4F8A",
                "width": 3,
            },
            marker={
                "size": 5,
                "color": "#0B4F8A",
            },
            hovertemplate=(
                "%{x|%b/%Y}<br>"
                "Observado: %{y:,.1f} GWh"
                "<extra></extra>"
            ),
        )
    )

    figure.add_trace(
        go.Scatter(
            x=state_scenario["mes"],
            y=upper_gwh,
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )

    figure.add_trace(
        go.Scatter(
            x=state_scenario["mes"],
            y=lower_gwh,
            mode="lines",
            name="Intervalo de confiança de 95%",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(73, 132, 180, 0.16)",
            hoverinfo="skip",
        )
    )

    figure.add_trace(
        go.Scatter(
            x=state_scenario["mes"],
            y=reference_gwh,
            mode="lines+markers",
            name="Referência SARIMA",
            line={
                "color": "#16A6A1",
                "width": 3,
            },
            marker={"size": 6},
            hovertemplate=(
                "%{x|%b/%Y}<br>"
                "Referência: %{y:,.1f} GWh"
                "<extra></extra>"
            ),
        )
    )

    if selected_scenario != "Cenário de referência":
        figure.add_trace(
            go.Scatter(
                x=state_scenario["mes"],
                y=scenario_gwh,
                mode="lines+markers",
                name=selected_scenario,
                line={
                    "color": active_color,
                    "width": 3,
                    "dash": "dash",
                },
                marker={"size": 6},
                hovertemplate=(
                    "%{x|%b/%Y}<br>"
                    f"{selected_scenario}: "
                    "%{y:,.1f} GWh"
                    "<extra></extra>"
                ),
            )
        )

    selected_year_start = pd.Timestamp(
        selected_year,
        1,
        1,
    )

    selected_year_end = pd.Timestamp(
        selected_year + 1,
        1,
        1,
    )

    figure.add_vrect(
        x0=selected_year_start,
        x1=selected_year_end,
        fillcolor="#1769AA",
        opacity=0.06,
        line_width=0,
        layer="below",
        annotation_text=f"Ano analisado: {selected_year}",
        annotation_position="top left",
    )
    
    figure.add_vline(
        x=pd.Timestamp("2026-01-01").timestamp() * 1_000,
        line_width=1,
        line_dash="dot",
        line_color="#7A8793",
    )

    figure.add_annotation(
        x=pd.Timestamp("2026-01-15"),
        y=1,
        yref="paper",
        text="Início da projeção",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font={
            "size": 12,
            "color": "#7A8793",
        },
    )

    figure.update_layout(
        height=400,
        margin={
            "l": 20,
            "r": 20,
            "t": 30,
            "b": 20,
        },
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
        xaxis_title=None,
        yaxis_title="Consumo mensal (GWh)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    figure.update_xaxes(
        showgrid=True,
        gridcolor="rgba(128,128,128,0.18)",
    )

    figure.update_yaxes(
        showgrid=True,
        gridcolor="rgba(128,128,128,0.18)",
        rangemode="tozero",
    )

    st.plotly_chart(
        figure,
        width="stretch",
        config={
            "displaylogo": False,
            "responsive": True,
        },
    )


    annual_scenario_gwh = (
        float(state_summary["consumo_cenario_mwh"])
        / 1_000
    )

    annual_reference_gwh = (
        float(state_summary["referencia_sarima_mwh"])
        / 1_000
    )

    difference_reference_pct = float(
        state_summary["diferenca_referencia_pct"]
    )

    observed_2025_gwh = (
        history.loc[
            history["uf"].eq(selected_uf)
            & history["ano"].eq(2025),
            "consumo_mwh",
        ].sum()
        / 1_000
    )

    variation_2025_pct = (
        annual_scenario_gwh / observed_2025_gwh - 1
    ) * 100

    mape = float(state_metric["mape_pct"])


    card_1, card_2, card_3, card_4 = st.columns(4)

    with card_1:
        st.metric(
            label=f"Consumo projetado em {selected_year}",
            value=(
                f"{format_number_br(annual_scenario_gwh, 0)} GWh"
            ),
            delta=(
                f"{format_number_br(variation_2025_pct, 2)}% "
                "em relação a 2025"
            ),
        )

    with card_2:
        st.metric(
            label=f"Referência SARIMA em {selected_year}",
            value=(
                f"{format_number_br(annual_reference_gwh, 0)} GWh"
            ),
        )

    with card_3:
        st.metric(
            label="Variação frente à referência",
            value=(
                f"{format_number_br(difference_reference_pct, 2)}%"
            ),
        )

    with card_4:
        st.metric(
            label="MAPE do SARIMA",
            value=f"{format_number_br(mape, 2)}%",
            help=(
                "Erro percentual absoluto médio obtido "
                "no teste temporal de 2024–2025."
            ),
        )

    st.caption(
        "Os cenários P25 e P75 representam trajetórias de demanda "
        "construídas a partir dos quantis das variações anuais "
        "históricas de cada estado. Eles não representam previsões "
        "probabilísticas nem efeitos climáticos causais."
    )

with analysis_tab:
    
    performance_column, shap_column = st.columns(
        [1.08, 0.92],
        gap="large",
    )


    with performance_column:
        st.subheader("Desempenho do modelo preditivo")

        st.caption(
            "Comparação entre os valores observados e "
            "previstos pelo SARIMA no teste temporal "
            "de 2024–2025."
        )

        state_predictions = (
            predictions.loc[
                predictions["uf"].eq(selected_uf)
            ]
            .sort_values("mes")
            .copy()
        )

        observed_gwh = (
            state_predictions["observado_mwh"]
            / 1_000
        )

        predicted_gwh = (
            state_predictions["previsto_sarima_mwh"]
            / 1_000
        )

        comparison_min = float(
            min(
                observed_gwh.min(),
                predicted_gwh.min(),
            )
        )

        comparison_max = float(
            max(
                observed_gwh.max(),
                predicted_gwh.max(),
            )
        )

        margin = (
            comparison_max - comparison_min
        ) * 0.08

        performance_figure = go.Figure()

        performance_figure.add_trace(
            go.Scatter(
                x=observed_gwh,
                y=predicted_gwh,
                mode="markers",
                name="Meses avaliados",
                marker={
                    "size": 9,
                    "color": state_predictions[
                        "erro_percentual_absoluto_pct"
                    ],
                    "colorscale": "Blues",
                    "showscale": True,
                    "colorbar": {
                        "title": "Erro (%)",
                        "thickness": 12,
                    },
                    "line": {
                        "color": "white",
                        "width": 0.7,
                    },
                },
                text=state_predictions[
                    "mes"
                ].dt.strftime("%m/%Y"),
                hovertemplate=(
                    "Mês: %{text}<br>"
                    "Observado: %{x:,.1f} GWh<br>"
                    "Previsto: %{y:,.1f} GWh"
                    "<extra></extra>"
                ),
            )
        )

        performance_figure.add_trace(
            go.Scatter(
                x=[
                    comparison_min - margin,
                    comparison_max + margin,
                ],
                y=[
                    comparison_min - margin,
                    comparison_max + margin,
                ],
                mode="lines",
                name="Previsão perfeita",
                line={
                    "color": "#80909E",
                    "width": 2,
                    "dash": "dash",
                },
                hoverinfo="skip",
            )
        )

        performance_figure.update_layout(
            height=370,
            margin={
                "l": 20,
                "r": 20,
                "t": 25,
                "b": 20,
            },
            xaxis_title="Observado (GWh)",
            yaxis_title="Previsto (GWh)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "left",
                "x": 0,
            },
        )

        performance_figure.update_xaxes(
            showgrid=True,
            gridcolor="rgba(128,128,128,0.18)",
        )

        performance_figure.update_yaxes(
            showgrid=True,
            gridcolor="rgba(128,128,128,0.18)",
            scaleanchor="x",
            scaleratio=1,
        )

        st.plotly_chart(
            performance_figure,
            width="stretch",
            config={"displaylogo": False},
        )

        mae_gwh = (
            float(state_metric["mae_mwh"])
            / 1_000
        )

        rmse_gwh = (
            float(state_metric["rmse_mwh"])
            / 1_000
        )

        metric_1, metric_2, metric_3 = st.columns(3)

        metric_1.metric(
            "MAE",
            f"{format_number_br(mae_gwh, 1)} GWh",
        )

        metric_2.metric(
            "RMSE",
            f"{format_number_br(rmse_gwh, 1)} GWh",
        )

        metric_3.metric(
            "MAPE",
            f"{format_number_br(mape, 2)}%",
        )


    with shap_column:
        st.subheader("Contribuição dos atributos")

        st.caption(
            "Importância relativa dos atributos no "
            "XGBoost de energia e clima para o estado "
            "selecionado."
        )

        attribute_labels = {
            "tendencia_mensal": "Tendência mensal",
            "mes_seno": "Sazonalidade (seno)",
            "mes_cosseno": "Sazonalidade (cosseno)",
            "consumo_mwh_lag_1": (
                "Consumo – defasagem de 1 mês"
            ),
            "consumo_mwh_lag_3": (
                "Consumo – defasagem de 3 meses"
            ),
            "consumo_mwh_lag_6": (
                "Consumo – defasagem de 6 meses"
            ),
            "consumo_mwh_lag_12": (
                "Consumo – defasagem de 12 meses"
            ),
            "umidade_media_pct_lag_12": (
                "Umidade média – defasagem de 12 meses"
            ),
            "uf_RJ": "Indicador estadual – RJ",
        }

        selected_shap = (
            state_shap.loc[
                state_shap["uf"].eq(selected_uf)
            ]
            .nlargest(
                7,
                "importancia_relativa_pct",
            )
            .sort_values(
                "importancia_relativa_pct",
                ascending=True,
            )
            .copy()
        )

        selected_shap["atributo_rotulo"] = (
            selected_shap["atributo"]
            .map(attribute_labels)
            .fillna(
                selected_shap["atributo"]
                .str.replace("_", " ")
                .str.capitalize()
            )
        )

        shap_figure = go.Figure(
            go.Bar(
                x=selected_shap[
                    "importancia_relativa_pct"
                ],
                y=selected_shap[
                    "atributo_rotulo"
                ],
                orientation="h",
                marker_color="#2A9D8F",
                text=selected_shap[
                    "importancia_relativa_pct"
                ].map(lambda value: f"{value:.2f}%"),
                textposition="outside",
                hovertemplate=(
                    "%{y}<br>"
                    "Importância relativa: %{x:.2f}%"
                    "<extra></extra>"
                ),
            )
        )

        shap_figure.update_layout(
            height=370,
            margin={
                "l": 20,
                "r": 55,
                "t": 25,
                "b": 20,
            },
            xaxis_title="Importância relativa (%)",
            yaxis_title=None,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )

        shap_figure.update_xaxes(
            showgrid=True,
            gridcolor="rgba(128,128,128,0.18)",
            range=[
                0,
                float(
                    selected_shap[
                        "importancia_relativa_pct"
                    ].max()
                )
                * 1.15,
            ],
        )

        st.plotly_chart(
            shap_figure,
            width="stretch",
            config={"displaylogo": False},
        )

        st.info(
            "O SARIMA constitui o núcleo preditivo do "
            "gêmeo digital. O XGBoost e os valores SHAP "
            "formam uma camada complementar de "
            "interpretabilidade, sem atribuição causal."
        )

with data_tab:

    st.subheader("Fontes e rastreabilidade dos dados")

    source_1, source_2, source_3, source_4 = st.columns(4)

    source_1.success("✓ EPE\n\nConsumo mensal de energia")
    source_2.success("✓ ONS\n\nCarga de energia")
    source_3.success("✓ IBGE\n\nIndicadores socioeconômicos")
    source_4.success("✓ INMET\n\nIndicadores climáticos")

    st.caption(
        "Protótipo acadêmico desenvolvido com dados públicos. "
        "As projeções e os cenários apoiam a análise de "
        "planejamento e não constituem garantia de demanda futura."
    )
    
    st.subheader("Arquitetura analítica")

    architecture_1, architecture_2, architecture_3 = (
        st.columns(3)
    )

    with architecture_1:
        st.info(
            "**Núcleo preditivo**\n\n"
            "Modelos SARIMA estaduais selecionados "
            "por validação temporal e reajustados com "
            "dados disponíveis até 2025."
        )

    with architecture_2:
        st.info(
            "**Motor de cenários**\n\n"
            "Trajetórias P25 e P75 construídas com "
            "quantis das variações anuais históricas "
            "de cada estado."
        )

    with architecture_3:
        st.info(
            "**Camada explicativa**\n\n"
            "XGBoost multivariado e valores SHAP "
            "utilizados para examinar a contribuição "
            "relativa dos atributos."
        )

    with st.expander(
        "Premissas e limitações metodológicas"
    ):
        st.markdown(
            """
            - O horizonte prospectivo compreende 2026–2030.
            - Os intervalos de confiança pertencem à projeção SARIMA.
            - P25 e P75 são trajetórias contrafactuais de demanda.
            - Os cenários não são probabilidades de ocorrência.
            - A análise SHAP não estabelece relações causais.
            - Os resultados apoiam o planejamento e devem ser
            atualizados quando novas observações estiverem disponíveis.
            """
        )

    st.subheader("Exportação dos resultados")

    export_summary = (
        summary.loc[
            summary["uf"].eq(selected_uf)
            & summary["cenario_rotulo"].eq(
                selected_scenario
            )
        ]
        .sort_values("ano")
        .copy()
    )

    download_1, download_2 = st.columns(2)

    with download_1:
        st.download_button(
            label=(
                "⬇️ Baixar cenário mensal selecionado"
            ),
            data=state_scenario.to_csv(
                index=False
            ).encode("utf-8-sig"),
            file_name=(
                f"cenario_{selected_uf}_"
                f"{selected_scenario.lower().replace(' ', '_')}_"
                "2026_2030.csv"
            ),
            mime="text/csv",
            width="stretch",
        )

    with download_2:
        st.download_button(
            label="⬇️ Baixar resumo anual selecionado",
            data=export_summary.to_csv(
                index=False
            ).encode("utf-8-sig"),
            file_name=(
                f"resumo_anual_{selected_uf}_"
                "2026_2030.csv"
            ),
            mime="text/csv",
            width="stretch",
        )