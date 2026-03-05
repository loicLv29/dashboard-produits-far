import os
import sys
import subprocess
import hmac
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_EXPORT_DIR = os.path.join(BASE_DIR, "..", "exports")

def resolve_export_dir() -> str:
    candidate_dirs = [
        DEFAULT_EXPORT_DIR,
        os.path.join(tempfile.gettempdir(), "ipln-dashboard", "exports"),
    ]

    for directory in candidate_dirs:
        try:
            os.makedirs(directory, exist_ok=True)
            probe_file = os.path.join(directory, ".write_probe")
            with open(probe_file, "w", encoding="utf-8") as f:
                f.write("ok")
            os.remove(probe_file)
            return directory
        except OSError:
            continue

    raise RuntimeError("Aucun dossier d'export inscriptible n'est disponible.")

EXPORT_DIR = resolve_export_dir()
TABLE_PATH = os.path.join(EXPORT_DIR, "table_scoring.csv")

def run_dashboard_engine() -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["IPLN_EXPORT_DIR"] = EXPORT_DIR
    return subprocess.run(
        [sys.executable, "dashboard_engine.py"],
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
        env=env
    )

def ensure_table_exists() -> None:
    if os.path.exists(TABLE_PATH):
        return

    result = run_dashboard_engine()
    if result.returncode != 0 or not os.path.exists(TABLE_PATH):
        raise FileNotFoundError(
            "Le fichier exports/table_scoring.csv est introuvable et sa generation a echoue.\n"
            f"stderr:\n{result.stderr}"
        )

def require_authentication() -> None:
    configured_password = st.secrets.get("APP_PASSWORD", "")

    if not configured_password:
        st.error("Acces bloque: configure APP_PASSWORD dans les secrets Streamlit.")
        st.stop()

    if st.session_state.get("authenticated", False):
        return

    st.title("Connexion")
    input_password = st.text_input("Mot de passe", type="password")

    if st.button("Se connecter"):
        if hmac.compare_digest(input_password, configured_password):
            st.session_state["authenticated"] = True
            st.rerun()
        st.error("Mot de passe incorrect.")

    st.stop()

require_authentication()

st.sidebar.caption(f"Fichier actif: {TABLE_PATH}")
if os.path.exists(TABLE_PATH):
    st.sidebar.caption(
        "Derniere maj: "
        + pd.Timestamp(os.path.getmtime(TABLE_PATH), unit="s").strftime("%Y-%m-%d %H:%M:%S")
    )

# =========================
# LOAD DATA
# =========================

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

ensure_table_exists()
df = load_data(TABLE_PATH)


# =========================
# NORMALISATION
# =========================

# EAN
if "ean" not in df.columns:
    for c in df.columns:
        if "ean" in c.lower():
            df = df.rename(columns={c: "ean"})
            break

df["ean"] = (
    df["ean"]
    .astype(str)
    .str.replace(".0","",regex=False)
    .str.strip()
)

# numériques
num_cols = [
    "prix_vente_ttc",
    "prix_achat_ht",
    "stock",
    "rotation_30j",
    "rotation_60j",
    "rotation_90j",
    "qte_30j",
    "qte_annee",
    "ca_annee_ttc",
    "marge_eur",
    "marge_pct",
    "score_acquisition",
    "score_fidelisation"
]

for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# année
if "annee" in df.columns:
    df["annee"] = pd.to_numeric(df["annee"], errors="coerce")


# =========================
# SIDEBAR
# =========================

st.sidebar.header("⚙ Paramètres")

# refresh
st.sidebar.subheader("🔄 Données")

if st.sidebar.button("Rafraichir les données"):

    with st.spinner("Recalcul en cours..."):

        result = run_dashboard_engine()

        if result.returncode == 0:
            st.cache_data.clear()
            st.success("Recalcul terminé.")
            st.rerun()

        else:
            st.error("Erreur lors du recalcul. Verifie les logs serveur.")
            print("dashboard_engine.py stderr:")
            print(result.stderr)


# =========================
# FILTRE ANNÉE
# =========================

selected_years = []

if "annee" in df.columns:

    years = sorted([int(x) for x in df["annee"].dropna().unique()])

    selected_years = st.sidebar.multiselect(
        "Année",
        years,
        default=[max(years)]
    )


# =========================
# RECHERCHE PRODUIT
# =========================

search_query = st.sidebar.text_input(
    "🔎 Recherche produit (ID / nom / EAN)"
)


# =========================
# PRIX
# =========================

min_price = float(df["prix_vente_ttc"].min())
max_price = float(df["prix_vente_ttc"].max())

prix_range = st.sidebar.slider(
    "💰 Plage de prix",
    min_price,
    max_price,
    (min_price,max_price)
)

prix_min,prix_max = prix_range

stock_min = st.sidebar.slider("Stock minimum",0,500,1)


# =========================
# FILTRES
# =========================

marques = st.sidebar.multiselect(
    "Marques",
    sorted(df["marque"].dropna().unique())
)

if marques:
    category_options = sorted(
        df[df["marque"].isin(marques)]["categorie"].dropna().unique()
    )
else:
    category_options = sorted(df["categorie"].dropna().unique())

categories = st.sidebar.multiselect(
    "Catégories",
    category_options
)


# =========================
# AGE MOYEN
# =========================

st.sidebar.subheader("👥 Age moyen")

if "age_moyen" in df.columns and df["age_moyen"].notna().any():

    age_min = float(df["age_moyen"].dropna().min())
    age_max = float(df["age_moyen"].dropna().max())

    age_range = st.sidebar.slider(
        "Plage âge moyen",
        min_value=age_min,
        max_value=age_max,
        value=(age_min, age_max)
    )

else:
    age_range = None


# =========================
# STATUT RISQUE
# =========================

st.sidebar.subheader("🚨 Statut risque")

statuts = st.sidebar.multiselect(
    "Filtrer par statut",
    ["🟢 Sain", "🟡 À surveiller", "🔴 À traiter"],
    default=["🟢 Sain", "🟡 À surveiller", "🔴 À traiter"]
)


# =========================
# TOP PRODUITS
# =========================

top_n = st.sidebar.slider(
    "Top N produits",
    10,
    2000,
    300
)


# =========================
# SCORE RISQUE
# =========================

df["p_stock"] = df["stock"].rank(pct=True)
df["p_rot60_low"] = 1 - df["rotation_60j"].rank(pct=True)
df["p_marge_low"] = 1 - df["marge_eur"].rank(pct=True)

df["score_risque"] = (
    0.45*df["p_stock"]+
    0.35*df["p_rot60_low"]+
    0.20*df["p_marge_low"]
)*100


def classify_risk(row):

    if row["rotation_60j"]==0 and row["stock"]>=5:
        return "🔴 À traiter"

    if row["rotation_60j"]<=0.15 and row["stock"]>=5:
        return "🟡 À surveiller"

    return "🟢 Sain"

df["statut_risque"] = df.apply(classify_risk,axis=1)


# =========================
# FILTRAGE
# =========================

df_filtered = df.copy()

if selected_years:
    df_filtered = df_filtered[df_filtered["annee"].isin(selected_years)]

if search_query:

    tokens = [t for t in search_query.lower().split() if t]
    search_cols = [
        c for c in [
            "id_product",
            "nom",
            "ean",
            "produit",
            "produit_declinaison",
            "marque",
            "categorie",
        ] if c in df_filtered.columns
    ]

    # Concat all searchable fields so multi-word queries can match across columns.
    search_text = (
        df_filtered[search_cols]
        .astype(str)
        .fillna("")
        .agg(" ".join, axis=1)
        .str.lower()
    )

    if tokens:
        mask = pd.Series(True, index=df_filtered.index)
        for token in tokens:
            mask &= search_text.str.contains(token, regex=False, na=False)
        df_filtered = df_filtered[mask]

df_filtered = df_filtered[
    (df_filtered["prix_vente_ttc"]>=prix_min)
    &
    (df_filtered["prix_vente_ttc"]<=prix_max)
]

df_filtered = df_filtered[df_filtered["stock"]>=stock_min]

if marques:
    df_filtered = df_filtered[df_filtered["marque"].isin(marques)]

if categories:
    df_filtered = df_filtered[df_filtered["categorie"].isin(categories)]

# filtre age
if age_range is not None:
    df_filtered = df_filtered[
        (df_filtered["age_moyen"].fillna(0) >= age_range[0]) &
        (df_filtered["age_moyen"].fillna(0) <= age_range[1])
    ]

# filtre statut risque
if statuts:
    df_filtered = df_filtered[
        df_filtered["statut_risque"].isin(statuts)
    ]

group_keys = [c for c in ["id_product", "id_product_attribute"] if c in df_filtered.columns] or ["id_product"]

# Keep full yearly rows for pivot/trend; dedup only after yearly features are computed.
df_yearly = df_filtered.copy()

if df_filtered.empty:
    st.warning(
        "Aucun resultat avec ces filtres. Essaie d'enlever une categorie ou une marque."
    )


# =========================
# PIVOT ANNÉES
# =========================

year_cols=[]

if "annee" in df_yearly.columns:

    pivot_qte = df_yearly.pivot_table(
        index=group_keys,
        columns="annee",
        values="qte_annee",
        aggfunc="sum"
    )

    pivot_ca = df_yearly.pivot_table(
        index=group_keys,
        columns="annee",
        values="ca_annee_ttc",
        aggfunc="sum"
    )

    pivot_qte.columns=[f"Qté {int(c)}" for c in pivot_qte.columns]
    pivot_ca.columns=[f"CA {int(c)}" for c in pivot_ca.columns]

    pivot = pd.concat([pivot_qte,pivot_ca],axis=1).reset_index()

    year_cols = [c for c in pivot.columns.tolist() if c not in group_keys]


# =========================
# TENDANCE
# =========================

if "annee" in df_yearly.columns:

    trend_df = (
        df_yearly
        .groupby(group_keys + ["annee"])["qte_annee"]
        .sum()
        .unstack(fill_value=0)
    )

    if trend_df.shape[1]>=2:

        last = trend_df.columns.max()
        prev = sorted(trend_df.columns)[-2]

        trend_df["trend"] = trend_df[last]-trend_df[prev]
        trend_df = trend_df["trend"].reset_index()
    else:
        trend_df = pd.DataFrame(columns=group_keys + ["trend"])
else:
    trend_df = pd.DataFrame(columns=group_keys + ["trend"])

# Keep one display row per product (latest selected year when available).
if "annee" in df_yearly.columns:
    df_display = df_yearly.sort_values("annee", ascending=False)
else:
    df_display = df_yearly.copy()

df_display = df_display.drop_duplicates(subset=group_keys, keep="first")
df_display = df_display.reset_index(drop=True)

if "annee" in df_yearly.columns:
    df_display = df_display.merge(pivot, on=group_keys, how="left")
    if not trend_df.empty:
        df_display = df_display.merge(trend_df, on=group_keys, how="left")

df_filtered = df_display


def trend_icon(x):

    if pd.isna(x):
        return ""

    if x>0:
        return "📈"

    if x<0:
        return "📉"

    return "➡️"


if "trend" in df_filtered.columns:
    df_filtered["trend"] = df_filtered["trend"].apply(trend_icon)
else:
    df_filtered["trend"] = ""


# =========================
# POTENTIEL CA
# =========================

df_filtered["potentiel_ca"] = (
    df_filtered["stock"]
    *
    df_filtered["prix_vente_ttc"]
    *
    0.30
)


# =========================
# STRATEGIE
# =========================

def strategie(row):

    if row["statut_risque"]=="🔴 À traiter":
        return "risque"

    if row["score_fidelisation"]>=row["score_acquisition"]:
        return "fidelisation"

    return "acquisition"

df_filtered["strategie"] = df_filtered.apply(strategie,axis=1)


# =========================
# IDEALO
# =========================

def build_idealo(e):

    if pd.isna(e):
        return None

    e=str(e)

    digits="".join(filter(str.isdigit,e))

    if len(digits)>=8:
        return f"https://www.idealo.fr/prechcat.html?q={digits}"

    return None

df_filtered["Idealo"] = df_filtered["ean"].apply(build_idealo)


# =========================
# KPI
# =========================

st.subheader("📌 Résumé du périmètre filtré")

k1,k2,k3,k4 = st.columns(4)

k1.metric("📸 Produits",len(df_filtered))

ca = df_filtered["ca_annee_ttc"].sum()

k2.metric("🚀 CA période",f"{ca:,.0f} €".replace(","," "))

marge = df_filtered["marge_pct"].mean()

k3.metric("👛 Marge moyenne",f"{marge:.1f} %")

stock = (df_filtered["stock"]*df_filtered["prix_achat_ht"]).sum()

k4.metric("💣 Valeur stock HT",f"{stock:,.0f} €".replace(","," "))


# =========================
# TABLES
# =========================

# Affichage: remplace les valeurs manquantes de marge par "N/A" pour eviter "None".
if "marge_pct" in df_filtered.columns:
    df_filtered["marge_pct_display"] = df_filtered["marge_pct"].apply(
        lambda x: "N/A" if pd.isna(x) else f"{x:.1f} %"
    )
else:
    df_filtered["marge_pct_display"] = "N/A"

base_cols=[
"id_product",
"ean",
"Idealo",
"nom",
"marque",
"categorie",
"trend",
"prix_vente_ttc",
"marge_pct_display",
"stock",
"potentiel_ca"
]

cols = base_cols + year_cols + [
"rotation_30j",
"rotation_60j",
"rotation_90j",
"score_acquisition",
"score_fidelisation",
"statut_risque"
]

cols=[c for c in cols if c in df_filtered.columns]


column_config={
"Idealo": st.column_config.LinkColumn(
    "Prix marché",
    display_text="Vérifier le prix"
),
"prix_vente_ttc": st.column_config.NumberColumn(
    "Prix TTC",
    format="%.2f €"
),
"potentiel_ca": st.column_config.NumberColumn(
    "Potentiel CA",
    format="%.0f €"
),
"marge_pct_display": st.column_config.TextColumn(
    "Marge %",
)
}


# =========================
# TABS
# =========================

tab1,tab2,tab3 = st.tabs([
"🚀 Acquisition",
"🔁 Fidélisation",
"🚨 Risque"
])


# =========================
# ACQUISITION
# =========================

with tab1:

    df_acq = df_filtered[df_filtered["strategie"]=="acquisition"]

    df_acq = df_acq.sort_values(
        "score_acquisition",
        ascending=False
    ).head(top_n)

    st.dataframe(
        df_acq[cols],
        hide_index=True,
        use_container_width=True,
        height=520,
        column_config=column_config
    )


# =========================
# FIDELISATION
# =========================

with tab2:

    df_fid = df_filtered[df_filtered["strategie"]=="fidelisation"]

    df_fid = df_fid.sort_values(
        "score_fidelisation",
        ascending=False
    ).head(top_n)

    st.dataframe(
        df_fid[cols],
        hide_index=True,
        use_container_width=True,
        height=520,
        column_config=column_config
    )


# =========================
# RISQUE
# =========================

with tab3:

    df_risk = df_filtered[df_filtered["strategie"]=="risque"]

    df_risk = df_risk.sort_values(
        "score_risque",
        ascending=False
    ).head(top_n)

    st.dataframe(
        df_risk[cols],
        hide_index=True,
        use_container_width=True,
        height=520,
        column_config=column_config
    )
