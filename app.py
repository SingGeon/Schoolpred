"""
Aplicatie BAC Romania - Statistici si Predictii ML
Streamlit + MongoDB + Scikit-learn
"""
import os
import sys
import time
import random
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from data.generate_bac_data import genereaza_date_bac
from database.mongodb_client import MongoDBClient
from models.predictor import BACPredictor

# ── Configurare pagina ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BAC Romania — Statistici & Predictii",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "bac_date_romania.csv")

MATERII_LABEL = {
    "nota_romana_oral": "Română Oral",
    "nota_romana_scris": "Română Scris",
    "nota_matematica": "Matematică",
    "nota_limba_straina": "Limbă Străină",
    "nota_specialitate": "Specialitate",
}
MATERII = list(MATERII_LABEL.keys())
CULORI_AN = px.colors.qualitative.Set2

# ── Incarcare date ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Se încarcă datele...")
def load_data() -> pd.DataFrame:
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    else:
        df = genereaza_date_bac(10000)
        df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    return df


def get_mongo_client() -> MongoDBClient:
    if "mongo_client" not in st.session_state:
        client = MongoDBClient()
        client.connect()
        st.session_state.mongo_client = client
    return st.session_state.mongo_client


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🎓 BAC Romania")
st.sidebar.caption("Statistici & Predictii ML")

PAGINI = [
    "🏠 Dashboard",
    "📅 Evoluție Anuală",
    "🗺️ Statistici pe Județ",
    "📚 Statistici pe Materie",
    "👥 Analiză Gen",
    "🏙️ Urban vs Rural",
    "🏫 Top Școli",
    "📊 Distribuție Note",
    "🔗 Corelații",
    "🤖 Model ML",
    "🔮 Predicție",
    "⚡ Live Dashboard",
]

pagina = st.sidebar.radio("Navigare", PAGINI, label_visibility="collapsed")
st.sidebar.divider()

# Status MongoDB
mongo = get_mongo_client()
if mongo.is_connected:
    st.sidebar.success("MongoDB: Conectat", icon="✅")
else:
    st.sidebar.warning("MongoDB: Offline — date din CSV", icon="⚠️")

df = load_data()

# Filtre globale in sidebar
st.sidebar.divider()
st.sidebar.subheader("Filtre globale")
ani_disponibili = sorted(df["an"].unique().tolist())
ani_selectati = st.sidebar.multiselect("An", ani_disponibili, default=ani_disponibili)
if ani_selectati:
    df = df[df["an"].isin(ani_selectati)]

mediu_selectat = st.sidebar.selectbox("Mediu", ["Toate", "Urban", "Rural"])
if mediu_selectat != "Toate":
    df = df[df["mediu"] == mediu_selectat]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if pagina == "🏠 Dashboard":
    st.title("🏠 Dashboard — BAC Romania")
    st.caption(f"Date: {len(df):,} elevi | Ani: {min(ani_selectati) if ani_selectati else '—'} – {max(ani_selectati) if ani_selectati else '—'}")

    # KPI cards
    c1, c2, c3, c4, c5 = st.columns(5)
    total = len(df)
    promovati = df["promovat"].sum()
    absenti = df["absent"].sum()
    rata = promovati / (total - absenti) * 100 if (total - absenti) > 0 else 0
    medie = df[df["absent"] == 0]["medie_generala"].mean()

    c1.metric("Total Elevi", f"{total:,}")
    c2.metric("Promovați", f"{int(promovati):,}")
    c3.metric("Absenți", f"{int(absenti):,}")
    c4.metric("Rată Promovare", f"{rata:.1f}%")
    c5.metric("Medie Generală", f"{medie:.2f}")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        rata_an = (
            df[df["absent"] == 0]
            .groupby("an")["promovat"]
            .mean()
            .reset_index()
            .rename(columns={"promovat": "Rată Promovare"})
        )
        rata_an["Rată Promovare"] *= 100
        fig = px.bar(
            rata_an, x="an", y="Rată Promovare",
            title="Rată Promovare pe An",
            labels={"an": "An", "Rată Promovare": "Rată (%)"},
            color="Rată Promovare", color_continuous_scale="Blues",
            text_auto=".1f",
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        medie_judet = (
            df[df["absent"] == 0]
            .groupby("judet")["medie_generala"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        fig2 = px.bar(
            medie_judet, x="medie_generala", y="judet",
            orientation="h",
            title="Top 10 Județe — Medie Generală",
            labels={"judet": "Județ", "medie_generala": "Medie"},
            color="medie_generala", color_continuous_scale="Greens",
        )
        fig2.update_layout(coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        pie_data = pd.DataFrame({
            "Categorie": ["Promovați", "Respinși", "Absenți"],
            "Număr": [
                int(promovati),
                int(total - promovati - absenti),
                int(absenti),
            ],
        })
        fig3 = px.pie(pie_data, values="Număr", names="Categorie",
                      title="Distribuție Rezultate", hole=0.4,
                      color_discrete_sequence=["#2ecc71", "#e74c3c", "#95a5a6"])
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        medii_materii = pd.DataFrame({
            "Materie": list(MATERII_LABEL.values()),
            "Medie": [df[m].mean() for m in MATERII],
        })
        fig4 = px.bar(
            medii_materii, x="Materie", y="Medie",
            title="Medie pe Materie (toți elevii)",
            color="Medie", color_continuous_scale="Oranges",
            text_auto=".2f",
        )
        fig4.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig4, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. EVOLUTIE ANUALA
# ═══════════════════════════════════════════════════════════════════════════════
elif pagina == "📅 Evoluție Anuală":
    st.title("📅 Evoluție Anuală BAC")

    df_nabs = df[df["absent"] == 0]

    rata_an = (
        df_nabs.groupby("an")["promovat"]
        .agg(["mean", "count", "sum"])
        .reset_index()
        .rename(columns={"mean": "rata", "count": "total", "sum": "promovati"})
    )
    rata_an["rata"] *= 100

    fig = px.line(
        rata_an, x="an", y="rata", markers=True,
        title="Evoluția Ratei de Promovare (2019–2024)",
        labels={"an": "An", "rata": "Rată Promovare (%)"},
        line_shape="spline",
    )
    fig.add_scatter(x=rata_an["an"], y=rata_an["rata"], mode="markers+text",
                    text=[f"{v:.1f}%" for v in rata_an["rata"]],
                    textposition="top center", showlegend=False,
                    marker=dict(size=10, color="#e74c3c"))
    fig.update_traces(line=dict(width=3, color="#3498db"), selector=dict(mode="lines"))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        medie_an = df_nabs.groupby("an")[MATERII].mean().reset_index()
        fig2 = go.Figure()
        for m in MATERII:
            fig2.add_trace(go.Scatter(
                x=medie_an["an"], y=medie_an[m],
                name=MATERII_LABEL[m], mode="lines+markers",
                line=dict(width=2),
            ))
        fig2.update_layout(
            title="Evoluția Mediei pe Materie",
            xaxis_title="An", yaxis_title="Medie",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        elevi_an = df.groupby("an").size().reset_index(name="nr_elevi")
        fig3 = px.bar(
            elevi_an, x="an", y="nr_elevi",
            title="Număr de Elevi pe An",
            labels={"an": "An", "nr_elevi": "Număr Elevi"},
            color="nr_elevi", color_continuous_scale="Purples", text_auto=True,
        )
        fig3.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Tabel Statistici Anuale")
    tabel = rata_an.copy()
    tabel["rata"] = tabel["rata"].round(1).astype(str) + "%"
    tabel.columns = ["An", "Rată Promovare", "Total Elevi", "Promovați"]
    st.dataframe(tabel, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. STATISTICI PE JUDET
# ═══════════════════════════════════════════════════════════════════════════════
elif pagina == "🗺️ Statistici pe Județ":
    st.title("🗺️ Statistici pe Județ")

    df_nabs = df[df["absent"] == 0]
    judet_stats = (
        df_nabs.groupby("judet")
        .agg(
            rata_promovare=("promovat", "mean"),
            medie_generala=("medie_generala", "mean"),
            nr_elevi=("id", "count"),
        )
        .reset_index()
    )
    judet_stats["rata_promovare"] *= 100
    judet_stats = judet_stats.sort_values("rata_promovare", ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            judet_stats.head(15), x="rata_promovare", y="judet",
            orientation="h", title="Top 15 Județe — Rată Promovare",
            labels={"judet": "Județ", "rata_promovare": "Rată (%)"},
            color="rata_promovare", color_continuous_scale="Greens",
            text_auto=".1f",
        )
        fig.update_layout(coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.bar(
            judet_stats.tail(15), x="rata_promovare", y="judet",
            orientation="h", title="Ultimele 15 Județe — Rată Promovare",
            labels={"judet": "Județ", "rata_promovare": "Rată (%)"},
            color="rata_promovare", color_continuous_scale="Reds",
            text_auto=".1f",
        )
        fig2.update_layout(coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig2, use_container_width=True)

    # Toate judetele
    fig3 = px.bar(
        judet_stats, x="judet", y="rata_promovare",
        title="Rată Promovare — Toate Județele",
        labels={"judet": "Județ", "rata_promovare": "Rată (%)"},
        color="rata_promovare",
        color_continuous_scale="RdYlGn",
        text_auto=".1f",
    )
    fig3.update_layout(xaxis_tickangle=-45, height=450)
    st.plotly_chart(fig3, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig4 = px.scatter(
            judet_stats, x="medie_generala", y="rata_promovare",
            size="nr_elevi", hover_name="judet",
            title="Medie vs Rată Promovare pe Județ",
            labels={"medie_generala": "Medie Generală", "rata_promovare": "Rată Promovare (%)"},
            color="rata_promovare", color_continuous_scale="Blues",
        )
        st.plotly_chart(fig4, use_container_width=True)

    with col4:
        st.subheader("Tabel Complet Județe")
        tabel_j = judet_stats.copy()
        tabel_j["rata_promovare"] = tabel_j["rata_promovare"].round(1).astype(str) + "%"
        tabel_j["medie_generala"] = tabel_j["medie_generala"].round(2)
        tabel_j.columns = ["Județ", "Rată Promovare", "Medie Generală", "Nr. Elevi"]
        st.dataframe(tabel_j, use_container_width=True, hide_index=True, height=400)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. STATISTICI PE MATERIE
# ═══════════════════════════════════════════════════════════════════════════════
elif pagina == "📚 Statistici pe Materie":
    st.title("📚 Statistici pe Materie")

    df_nabs = df[df["absent"] == 0]

    # Medii si std per materie
    stats_materii = pd.DataFrame({
        "Materie": list(MATERII_LABEL.values()),
        "Medie": [df_nabs[m].mean() for m in MATERII],
        "Std": [df_nabs[m].std() for m in MATERII],
        "Min": [df_nabs[m].min() for m in MATERII],
        "Max": [df_nabs[m].max() for m in MATERII],
        "Sub 5 (%)": [(df_nabs[m] < 5).mean() * 100 for m in MATERII],
    })

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            stats_materii, x="Materie", y="Medie",
            error_y="Std",
            title="Medie ± Deviație Standard pe Materie",
            color="Medie", color_continuous_scale="Blues",
            text_auto=".2f",
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.bar(
            stats_materii, x="Materie", y="Sub 5 (%)",
            title="Procent Note sub 5 pe Materie",
            color="Sub 5 (%)", color_continuous_scale="Reds",
            text_auto=".1f",
        )
        fig2.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    # Box plot
    df_melted = df_nabs[MATERII].rename(columns=MATERII_LABEL).melt(
        var_name="Materie", value_name="Notă"
    )
    fig3 = px.box(
        df_melted, x="Materie", y="Notă",
        title="Distribuție Note pe Materie (Box Plot)",
        color="Materie",
        points=False,
    )
    fig3.update_layout(showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

    # Evolutia mediei per materie si an
    medie_an_mat = df_nabs.groupby("an")[MATERII].mean().reset_index()
    fig4 = go.Figure()
    for m in MATERII:
        fig4.add_trace(go.Bar(
            x=medie_an_mat["an"].astype(str),
            y=medie_an_mat[m],
            name=MATERII_LABEL[m],
        ))
    fig4.update_layout(
        barmode="group",
        title="Medie pe Materie și An",
        xaxis_title="An", yaxis_title="Medie",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Statistici Detaliate pe Materie")
    stats_materii["Medie"] = stats_materii["Medie"].round(2)
    stats_materii["Std"] = stats_materii["Std"].round(2)
    stats_materii["Sub 5 (%)"] = stats_materii["Sub 5 (%)"].round(1).astype(str) + "%"
    st.dataframe(stats_materii, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ANALIZA GEN
# ═══════════════════════════════════════════════════════════════════════════════
elif pagina == "👥 Analiză Gen":
    st.title("👥 Analiză pe Gen")

    df_nabs = df[df["absent"] == 0]

    gen_stats = (
        df_nabs.groupby("gen")
        .agg(
            rata_promovare=("promovat", "mean"),
            medie_generala=("medie_generala", "mean"),
            nr_elevi=("id", "count"),
        )
        .reset_index()
    )
    gen_stats["rata_promovare"] *= 100
    gen_stats["gen_label"] = gen_stats["gen"].map({"F": "Fete", "M": "Băieți"})

    col1, col2, col3 = st.columns(3)
    with col1:
        fig = px.bar(
            gen_stats, x="gen_label", y="rata_promovare",
            title="Rată Promovare pe Gen",
            labels={"gen_label": "Gen", "rata_promovare": "Rată (%)"},
            color="gen_label",
            color_discrete_map={"Fete": "#e91e63", "Băieți": "#2196f3"},
            text_auto=".1f",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.bar(
            gen_stats, x="gen_label", y="medie_generala",
            title="Medie Generală pe Gen",
            labels={"gen_label": "Gen", "medie_generala": "Medie"},
            color="gen_label",
            color_discrete_map={"Fete": "#e91e63", "Băieți": "#2196f3"},
            text_auto=".2f",
        )
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    with col3:
        fig3 = px.pie(
            gen_stats, values="nr_elevi", names="gen_label",
            title="Distribuție Elevi pe Gen",
            color="gen_label",
            color_discrete_map={"Fete": "#e91e63", "Băieți": "#2196f3"},
            hole=0.4,
        )
        st.plotly_chart(fig3, use_container_width=True)

    # Medii per materie si gen
    medie_gen_mat = df_nabs.groupby("gen")[MATERII].mean().reset_index()
    medie_gen_mat["gen_label"] = medie_gen_mat["gen"].map({"F": "Fete", "M": "Băieți"})

    fig4 = go.Figure()
    for _, row in medie_gen_mat.iterrows():
        fig4.add_trace(go.Bar(
            x=list(MATERII_LABEL.values()),
            y=[row[m] for m in MATERII],
            name=row["gen_label"],
        ))
    fig4.update_layout(
        barmode="group",
        title="Medie pe Materie și Gen",
        xaxis_title="Materie", yaxis_title="Medie",
    )
    st.plotly_chart(fig4, use_container_width=True)

    # Evolutie rata promovare pe gen si an
    gen_an = (
        df_nabs.groupby(["an", "gen"])["promovat"]
        .mean()
        .reset_index()
    )
    gen_an["promovat"] *= 100
    gen_an["gen_label"] = gen_an["gen"].map({"F": "Fete", "M": "Băieți"})

    fig5 = px.line(
        gen_an, x="an", y="promovat", color="gen_label",
        title="Evoluție Rată Promovare pe Gen și An",
        labels={"an": "An", "promovat": "Rată (%)", "gen_label": "Gen"},
        markers=True, line_shape="spline",
        color_discrete_map={"Fete": "#e91e63", "Băieți": "#2196f3"},
    )
    st.plotly_chart(fig5, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. URBAN VS RURAL
# ═══════════════════════════════════════════════════════════════════════════════
elif pagina == "🏙️ Urban vs Rural":
    st.title("🏙️ Urban vs Rural")

    df_nabs = df[df["absent"] == 0]

    mediu_stats = (
        df_nabs.groupby("mediu")
        .agg(
            rata_promovare=("promovat", "mean"),
            medie_generala=("medie_generala", "mean"),
            nr_elevi=("id", "count"),
        )
        .reset_index()
    )
    mediu_stats["rata_promovare"] *= 100

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            mediu_stats, x="mediu", y="rata_promovare",
            title="Rată Promovare: Urban vs Rural",
            labels={"mediu": "Mediu", "rata_promovare": "Rată (%)"},
            color="mediu",
            color_discrete_map={"Urban": "#3498db", "Rural": "#27ae60"},
            text_auto=".1f",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.bar(
            mediu_stats, x="mediu", y="medie_generala",
            title="Medie Generală: Urban vs Rural",
            labels={"mediu": "Mediu", "medie_generala": "Medie"},
            color="mediu",
            color_discrete_map={"Urban": "#3498db", "Rural": "#27ae60"},
            text_auto=".2f",
        )
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # Medii per materie si mediu
    medie_mediu_mat = df_nabs.groupby("mediu")[MATERII].mean().reset_index()
    fig3 = go.Figure()
    for _, row in medie_mediu_mat.iterrows():
        culoare = "#3498db" if row["mediu"] == "Urban" else "#27ae60"
        fig3.add_trace(go.Bar(
            x=list(MATERII_LABEL.values()),
            y=[row[m] for m in MATERII],
            name=row["mediu"],
            marker_color=culoare,
        ))
    fig3.update_layout(
        barmode="group",
        title="Medie pe Materie: Urban vs Rural",
        xaxis_title="Materie", yaxis_title="Medie",
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Evolutie pe ani
    mediu_an = (
        df_nabs.groupby(["an", "mediu"])["promovat"]
        .mean()
        .reset_index()
    )
    mediu_an["promovat"] *= 100
    fig4 = px.line(
        mediu_an, x="an", y="promovat", color="mediu",
        title="Evoluție Rată Promovare: Urban vs Rural",
        labels={"an": "An", "promovat": "Rată (%)", "mediu": "Mediu"},
        markers=True, line_shape="spline",
        color_discrete_map={"Urban": "#3498db", "Rural": "#27ae60"},
    )
    st.plotly_chart(fig4, use_container_width=True)

    # Rata promovare per judet si mediu
    judet_mediu = (
        df_nabs.groupby(["judet", "mediu"])["promovat"]
        .mean()
        .reset_index()
    )
    judet_mediu["promovat"] *= 100
    fig5 = px.bar(
        judet_mediu, x="judet", y="promovat", color="mediu",
        barmode="group",
        title="Rată Promovare pe Județ și Mediu",
        labels={"judet": "Județ", "promovat": "Rată (%)", "mediu": "Mediu"},
        color_discrete_map={"Urban": "#3498db", "Rural": "#27ae60"},
    )
    fig5.update_layout(xaxis_tickangle=-45, height=420)
    st.plotly_chart(fig5, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. TOP SCOLI
# ═══════════════════════════════════════════════════════════════════════════════
elif pagina == "🏫 Top Școli":
    st.title("🏫 Top Școli")

    df_nabs = df[df["absent"] == 0]

    scoala_stats = (
        df_nabs.groupby(["scoala", "mediu"])
        .agg(
            rata_promovare=("promovat", "mean"),
            medie_generala=("medie_generala", "mean"),
            nr_elevi=("id", "count"),
        )
        .reset_index()
    )
    scoala_stats["rata_promovare"] *= 100
    scoala_stats = scoala_stats[scoala_stats["nr_elevi"] >= 5]

    col1, col2 = st.columns([2, 1])
    with col1:
        top20 = scoala_stats.nlargest(20, "rata_promovare")
        fig = px.bar(
            top20, x="rata_promovare", y="scoala",
            orientation="h",
            title="Top 20 Școli — Rată Promovare",
            labels={"scoala": "Școală", "rata_promovare": "Rată (%)"},
            color="mediu",
            color_discrete_map={"Urban": "#3498db", "Rural": "#27ae60"},
            text_auto=".1f",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=550)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Statistici Școli")
        st.metric("Total Școli", f"{len(scoala_stats):,}")
        st.metric("Cea mai bună rată", f"{scoala_stats['rata_promovare'].max():.1f}%")
        st.metric("Cea mai slabă rată", f"{scoala_stats['rata_promovare'].min():.1f}%")
        st.metric("Medie generală", f"{scoala_stats['rata_promovare'].mean():.1f}%")

    # Tabel top scoli
    st.subheader("Tabel Top 30 Școli")
    tabel = scoala_stats.nlargest(30, "rata_promovare")[
        ["scoala", "mediu", "rata_promovare", "medie_generala", "nr_elevi"]
    ].copy()
    tabel["rata_promovare"] = tabel["rata_promovare"].round(1).astype(str) + "%"
    tabel["medie_generala"] = tabel["medie_generala"].round(2)
    tabel.columns = ["Școală", "Mediu", "Rată Promovare", "Medie Generală", "Nr. Elevi"]
    st.dataframe(tabel, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. DISTRIBUTIE NOTE
# ═══════════════════════════════════════════════════════════════════════════════
elif pagina == "📊 Distribuție Note":
    st.title("📊 Distribuție Note")

    df_nabs = df[df["absent"] == 0]

    materie_sel = st.selectbox(
        "Alege materia",
        ["Toate materiile"] + list(MATERII_LABEL.values()),
    )

    if materie_sel == "Toate materiile":
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=list(MATERII_LABEL.values()) + ["Medie Generală"],
        )
        materii_all = MATERII + ["medie_generala"]
        labels_all = list(MATERII_LABEL.values()) + ["Medie Generală"]
        for i, (m, lbl) in enumerate(zip(materii_all, labels_all)):
            r, c = divmod(i, 3)
            fig.add_trace(
                go.Histogram(x=df_nabs[m], nbinsx=30, name=lbl, showlegend=False),
                row=r + 1, col=c + 1,
            )
        fig.update_layout(title="Distribuție Note — Toate Materiile", height=550)
        st.plotly_chart(fig, use_container_width=True)
    else:
        col_map = {v: k for k, v in MATERII_LABEL.items()}
        col_map["Medie Generală"] = "medie_generala"
        col = col_map.get(materie_sel, "medie_generala")

        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(
                df_nabs, x=col, nbins=40,
                title=f"Distribuție Note — {materie_sel}",
                labels={col: "Notă", "count": "Număr Elevi"},
                color_discrete_sequence=["#3498db"],
            )
            fig.add_vline(x=5, line_dash="dash", line_color="red",
                          annotation_text="Limita promovare (5)")
            fig.add_vline(x=df_nabs[col].mean(), line_dash="dot", line_color="green",
                          annotation_text=f"Medie: {df_nabs[col].mean():.2f}")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.violin(
                df_nabs, y=col, x="an",
                title=f"Distribuție {materie_sel} pe An",
                labels={col: "Notă", "an": "An"},
                box=True, points=False,
            )
            st.plotly_chart(fig2, use_container_width=True)

    # Histogram medie generala pe mediu
    fig_med = px.histogram(
        df_nabs, x="medie_generala", color="mediu",
        nbins=40, barmode="overlay", opacity=0.7,
        title="Distribuție Medie Generală: Urban vs Rural",
        labels={"medie_generala": "Medie Generală", "mediu": "Mediu"},
        color_discrete_map={"Urban": "#3498db", "Rural": "#27ae60"},
    )
    fig_med.add_vline(x=5, line_dash="dash", line_color="red",
                      annotation_text="Limita promovare")
    st.plotly_chart(fig_med, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. CORELATII
# ═══════════════════════════════════════════════════════════════════════════════
elif pagina == "🔗 Corelații":
    st.title("🔗 Corelații între Materii")

    df_nabs = df[df["absent"] == 0]
    cols_corr = MATERII + ["medie_generala"]
    labels_corr = list(MATERII_LABEL.values()) + ["Medie Generală"]

    corr_matrix = df_nabs[cols_corr].corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=labels_corr, y=labels_corr,
        colorscale="RdBu", zmid=0,
        text=corr_matrix.round(2).values,
        texttemplate="%{text}",
        textfont={"size": 11},
    ))
    fig.update_layout(
        title="Matrice de Corelație între Materii",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        m1 = st.selectbox("Materia X", list(MATERII_LABEL.values()), key="m1")
    with col2:
        m2 = st.selectbox("Materia Y", list(MATERII_LABEL.values()), index=2, key="m2")

    col_map = {v: k for k, v in MATERII_LABEL.items()}
    c1, c2 = col_map[m1], col_map[m2]

    sample = df_nabs.sample(min(2000, len(df_nabs)), random_state=42)
    fig2 = px.scatter(
        sample, x=c1, y=c2, color="promovat",
        title=f"Scatter: {m1} vs {m2}",
        labels={c1: m1, c2: m2, "promovat": "Promovat"},
        color_discrete_map={0: "#e74c3c", 1: "#2ecc71"},
        opacity=0.6, trendline="ols",
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Scatter matrix
    st.subheader("Scatter Matrix (eșantion 1000 elevi)")
    sample_sm = df_nabs[MATERII + ["promovat"]].sample(1000, random_state=42)
    sample_sm = sample_sm.rename(columns=MATERII_LABEL)
    sample_sm["Rezultat"] = sample_sm["promovat"].map({0: "Respins", 1: "Promovat"})
    fig3 = px.scatter_matrix(
        sample_sm,
        dimensions=list(MATERII_LABEL.values()),
        color="Rezultat",
        color_discrete_map={"Respins": "#e74c3c", "Promovat": "#2ecc71"},
        title="Scatter Matrix Materii",
        opacity=0.4,
    )
    fig3.update_traces(diagonal_visible=False, marker=dict(size=2))
    fig3.update_layout(height=600)
    st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 10. MODEL ML
# ═══════════════════════════════════════════════════════════════════════════════
elif pagina == "🤖 Model ML":
    st.title("🤖 Model Machine Learning")

    if "predictor" not in st.session_state:
        st.session_state.predictor = BACPredictor()
        st.session_state.predictor.load_model()

    predictor: BACPredictor = st.session_state.predictor

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Antrenare Model")
        st.write("**Algoritmi:**")
        st.write("- 🌲 Random Forest (clasificare promovat/respins)")
        st.write("- 📈 Gradient Boosting (predicție medie)")
        st.write(f"**Date antrenare:** {len(df[df['absent']==0]):,} elevi")

        if st.button("🚀 Antrenează Modelul", type="primary", use_container_width=True):
            with st.spinner("Antrenare în curs..."):
                rezultate = predictor.train(df)
                st.session_state.ml_rezultate = rezultate
                predictor.save_model()
            st.success("Model antrenat cu succes!")

        if predictor.is_trained:
            st.subheader("Performanță Model")
            st.metric("Acuratețe Clasificator", f"{predictor.accuracy:.1%}")
            st.metric("MAE Regressor", f"{predictor.mae:.3f}")
            st.metric("R² Score", f"{predictor.r2:.3f}")

    with col2:
        if predictor.is_trained:
            st.subheader("Importanța Feature-urilor")
            fi = predictor.get_feature_importance()
            fi_labels = {
                "nota_romana_oral": "Română Oral",
                "nota_romana_scris": "Română Scris",
                "nota_matematica": "Matematică",
                "nota_limba_straina": "Limbă Străină",
                "nota_specialitate": "Specialitate",
                "mediu_encoded": "Mediu (Urban/Rural)",
                "gen_encoded": "Gen",
                "an": "An",
            }
            fi_df = pd.DataFrame({
                "Feature": [fi_labels.get(k, k) for k in fi.keys()],
                "Importanță": list(fi.values()),
            }).sort_values("Importanță", ascending=True)
            fig = px.bar(
                fi_df, x="Importanță", y="Feature",
                orientation="h",
                title="Importanța Feature-urilor (Random Forest)",
                color="Importanță", color_continuous_scale="Blues",
            )
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Apasă 'Antrenează Modelul' pentru a vedea rezultatele.")

    if predictor.is_trained and predictor.conf_matrix is not None:
        st.subheader("Raport Clasificare & Matrice de Confuzie")
        col3, col4 = st.columns(2)

        with col3:
            cm = predictor.conf_matrix
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm, x=["Respins", "Promovat"], y=["Respins", "Promovat"],
                colorscale="Blues",
                text=cm, texttemplate="%{text}",
                textfont={"size": 16},
            ))
            fig_cm.update_layout(
                title="Matrice de Confuzie",
                xaxis_title="Predicție", yaxis_title="Real",
                height=350,
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        with col4:
            st.subheader("Raport Clasificare")
            if predictor.class_report:
                st.code(predictor.class_report)


# ═══════════════════════════════════════════════════════════════════════════════
# 11. PREDICTIE
# ═══════════════════════════════════════════════════════════════════════════════
elif pagina == "🔮 Predicție":
    st.title("🔮 Predicție Rezultat BAC")

    if "predictor" not in st.session_state:
        st.session_state.predictor = BACPredictor()
        st.session_state.predictor.load_model()

    predictor: BACPredictor = st.session_state.predictor

    if not predictor.is_trained:
        st.warning("Modelul nu este antrenat. Mergi la pagina 🤖 Model ML și antrenează modelul.")
        if st.button("Antrenează acum"):
            with st.spinner("Antrenare..."):
                predictor.train(df)
                predictor.save_model()
            st.success("Gata!")
            st.rerun()
    else:
        st.subheader("Introduce notele unui elev")
        col1, col2 = st.columns(2)

        with col1:
            nota_ro_oral = st.slider("Română Oral", 1.0, 10.0, 7.0, 0.1)
            nota_ro_scris = st.slider("Română Scris", 1.0, 10.0, 6.5, 0.1)
            nota_mat = st.slider("Matematică", 1.0, 10.0, 6.0, 0.1)
            nota_ls = st.slider("Limbă Străină", 1.0, 10.0, 7.0, 0.1)
            nota_spec = st.slider("Specialitate", 1.0, 10.0, 6.5, 0.1)

        with col2:
            mediu = st.radio("Mediu", ["Urban", "Rural"])
            gen = st.radio("Gen", ["M", "F"], format_func=lambda x: "Băiat" if x == "M" else "Fată")
            an = st.selectbox("An", [2024, 2023, 2022, 2021, 2020, 2019])

            st.divider()
            medie_calc = round(
                nota_ro_oral * 0.1 + nota_ro_scris * 0.3 +
                nota_mat * 0.3 + nota_ls * 0.15 + nota_spec * 0.15, 2
            )
            st.metric("Medie calculată", medie_calc)

        features = {
            "nota_romana_oral": nota_ro_oral,
            "nota_romana_scris": nota_ro_scris,
            "nota_matematica": nota_mat,
            "nota_limba_straina": nota_ls,
            "nota_specialitate": nota_spec,
            "mediu": mediu,
            "gen": gen,
            "an": an,
        }

        if st.button("🔮 Prezice Rezultatul", type="primary", use_container_width=True):
            pred_cls, confidence = predictor.predict_promovat(features)
            pred_medie = predictor.predict_medie(features)

            st.divider()
            col3, col4, col5 = st.columns(3)

            if pred_cls == 1:
                col3.success(f"✅ PROMOVAT\nîncredere: {confidence:.1%}")
            else:
                col3.error(f"❌ RESPINS\nîncredere: {confidence:.1%}")

            col4.metric("Medie Prezisă (ML)", f"{pred_medie:.2f}")
            col5.metric("Medie Calculată", f"{medie_calc:.2f}")

            # Gauge chart pentru probabilitate
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=confidence * 100,
                title={"text": "Probabilitate Promovare (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#2ecc71" if pred_cls == 1 else "#e74c3c"},
                    "steps": [
                        {"range": [0, 50], "color": "#ffebee"},
                        {"range": [50, 75], "color": "#fff9c4"},
                        {"range": [75, 100], "color": "#e8f5e9"},
                    ],
                    "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 50},
                },
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Salveaza in MongoDB
            mongo = get_mongo_client()
            pred_data = {**features, "pred_promovat": pred_cls, "pred_medie": pred_medie,
                         "confidence": confidence}
            mongo.insert_prediction(pred_data)


# ═══════════════════════════════════════════════════════════════════════════════
# 12. LIVE DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
elif pagina == "⚡ Live Dashboard":
    st.title("⚡ Live Dashboard — Procesare în Timp Real")
    st.caption("Simulare procesare rezultate BAC în timp real (actualizare la 3 secunde)")

    if "live_history" not in st.session_state:
        st.session_state.live_history = []
        st.session_state.live_total = 0
        st.session_state.live_promovati = 0

    # Generare batch nou de elevi simulati
    batch_size = random.randint(3, 8)
    nou_batch = df.sample(batch_size).to_dict(orient="records")

    for elev in nou_batch:
        st.session_state.live_history.append({
            "timp": time.strftime("%H:%M:%S"),
            "judet": elev["judet"],
            "mediu": elev["mediu"],
            "medie": elev["medie_generala"],
            "promovat": elev["promovat"],
            "gen": elev["gen"],
        })
        st.session_state.live_total += 1
        st.session_state.live_promovati += elev["promovat"]

    # Pastreaza ultimele 100 intrari
    if len(st.session_state.live_history) > 100:
        st.session_state.live_history = st.session_state.live_history[-100:]

    live_df = pd.DataFrame(st.session_state.live_history)

    # KPI live
    total_live = st.session_state.live_total
    prom_live = st.session_state.live_promovati
    rata_live = prom_live / total_live * 100 if total_live > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Procesați (sesiune)", f"{total_live:,}")
    c2.metric("Promovați", f"{int(prom_live):,}")
    c3.metric("Rată Promovare Live", f"{rata_live:.1f}%")
    c4.metric("Ultim Batch", f"{batch_size} elevi")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        # Grafic rolling rata de promovare
        if len(live_df) >= 2:
            live_df["promovat_int"] = live_df["promovat"].astype(int)
            window = min(20, len(live_df))
            live_df["rata_rolling"] = live_df["promovat_int"].rolling(window=window, min_periods=1).mean() * 100

            fig_live = go.Figure()
            fig_live.add_trace(go.Scatter(
                x=list(range(len(live_df))),
                y=live_df["rata_rolling"],
                mode="lines+markers",
                line=dict(color="#e74c3c", width=2),
                marker=dict(size=5),
                name="Rată Rolling",
                fill="tozeroy",
                fillcolor="rgba(231,76,60,0.15)",
            ))
            fig_live.add_hline(y=50, line_dash="dash", line_color="orange",
                               annotation_text="50%")
            fig_live.update_layout(
                title=f"Rată Promovare Rolling (fereastra {window})",
                xaxis_title="Nr. Elev",
                yaxis_title="Rată (%)",
                yaxis_range=[0, 100],
                height=320,
            )
            st.plotly_chart(fig_live, use_container_width=True)

    with col2:
        # Donut live
        if len(live_df) > 0:
            counts = live_df["promovat"].value_counts().reset_index()
            counts["promovat"] = counts["promovat"].map({1: "Promovat", 0: "Respins"})
            fig_donut = px.pie(
                counts, values="count", names="promovat",
                title="Distribuție Live (ultimii 100)",
                hole=0.5,
                color="promovat",
                color_discrete_map={"Promovat": "#2ecc71", "Respins": "#e74c3c"},
            )
            fig_donut.update_layout(height=320)
            st.plotly_chart(fig_donut, use_container_width=True)

    # Tabel ultimii elevi procesati
    if len(live_df) > 0:
        st.subheader("Ultimii Elevi Procesați")
        ultimii = live_df.tail(15).copy()[::-1]
        ultimii["promovat"] = ultimii["promovat"].map({1: "✅ Promovat", 0: "❌ Respins"})
        ultimii["medie"] = ultimii["medie"].round(2)
        ultimii.columns = ["Timp", "Județ", "Mediu", "Medie", "Rezultat", "Gen"]
        st.dataframe(ultimii, use_container_width=True, hide_index=True)

    # Bar chart judete procesate
    if len(live_df) >= 5:
        col3, col4 = st.columns(2)
        with col3:
            judet_live = live_df.groupby("judet").size().sort_values(ascending=False).head(10)
            fig_j = px.bar(
                x=judet_live.index, y=judet_live.values,
                title="Județe Procesate (top 10)",
                labels={"x": "Județ", "y": "Nr. Elevi"},
                color=judet_live.values, color_continuous_scale="Blues",
            )
            fig_j.update_layout(coloraxis_showscale=False, xaxis_tickangle=-30)
            st.plotly_chart(fig_j, use_container_width=True)

        with col4:
            medie_live_time = live_df.groupby("timp")["medie"].mean().reset_index()
            fig_mt = px.line(
                medie_live_time, x="timp", y="medie",
                title="Medie Generală în Timp",
                labels={"timp": "Timp", "medie": "Medie"},
                markers=True,
            )
            fig_mt.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(fig_mt, use_container_width=True)

    # Auto-refresh
    time.sleep(3)
    st.rerun()
