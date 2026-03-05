import os
import pandas as pd
import numpy as np

# =========================
# PARAMÈTRES
# =========================
SEUIL_PRIX_PROMO = 80
STOCK_MIN = 1
TOP_N = 200

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
DEFAULT_EXPORT_DIR = os.path.join(BASE_DIR, "..", "exports")
EXPORT_DIR = os.environ.get("IPLN_EXPORT_DIR", DEFAULT_EXPORT_DIR)

# ⚠️ Mets ici les noms EXACTS de tes fichiers (sans _2025 si tu as changé)
FILE_234 = "234_clients_categories.csv"
FILE_235 = "235_connaissance_produits.csv"
FILE_236 = "236_clients_produits.csv"

# =========================
# HELPERS
# =========================
def first_existing_path(*candidates: str) -> str:
    """Return the first existing file path within DATA_DIR."""
    for name in candidates:
        path = os.path.join(DATA_DIR, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Aucun fichier trouvé parmi: {candidates} dans {DATA_DIR}")

def normalize_ean(df: pd.DataFrame) -> pd.DataFrame:
    # Renomme ean si la colonne s'appelle ean13 / ean13_x / etc.
    if "ean" not in df.columns:
        for c in df.columns:
            if "ean" in c.lower():
                df = df.rename(columns={c: "ean"})
                break
    if "ean" in df.columns:
        df["ean"] = (
            df["ean"].astype(str)
            .str.replace(".0", "", regex=False)
            .str.strip()
        )
        df.loc[df["ean"].isin(["nan", "None", ""]), "ean"] = np.nan
    return df

def detect_tva_rate(row) -> float:
    # Règle demandée :
    # - NEUF => 20%
    # - catégorie TVA 5,5% si tva_5_5==True (si dispo) OU catégorie contient LIVRE
    # - OCCASION => 0%
    condition = str(row.get("condition_produit", "")).strip().lower()
    if condition == "occasion":
        return 0.0

    tva_5_5 = row.get("tva_5_5", None)
    if isinstance(tva_5_5, (bool, np.bool_)) and tva_5_5 is True:
        return 0.055

    cat = str(row.get("categorie", "")).upper()
    if "LIVRE" in cat:
        return 0.055

    return 0.20

# =========================
# CHARGEMENT
# =========================
os.makedirs(EXPORT_DIR, exist_ok=True)

path_234 = first_existing_path(FILE_234, "234_clients_categories_2025.csv")
path_235 = first_existing_path(FILE_235, "235_connaissance_produits.csv")
path_236 = first_existing_path(FILE_236, "236_clients_produits_2025.csv")

df_cat = pd.read_csv(path_234, sep=";")
df_prod = pd.read_csv(path_235, sep=";")
df_sales = pd.read_csv(path_236, sep=";")

df_prod = normalize_ean(df_prod)
df_sales = normalize_ean(df_sales)

# =========================
# TYPES NUMÉRIQUES PRODUITS
# =========================
for col in ["prix_vente_ht", "prix_achat_ht", "stock"]:
    if col in df_prod.columns:
        df_prod[col] = pd.to_numeric(df_prod[col], errors="coerce")

# =========================
# TVA => PRIX TTC
# =========================
if "prix_vente_ht" not in df_prod.columns:
    raise KeyError("La colonne 'prix_vente_ht' est absente du fichier 235.")

df_prod["tva_rate"] = df_prod.apply(detect_tva_rate, axis=1)

df_prod["prix_vente_ttc"] = np.where(
    df_prod["tva_rate"] == 0.0,
    df_prod["prix_vente_ht"],
    df_prod["prix_vente_ht"] * (1 + df_prod["tva_rate"])
)

# =========================
# MARGE (HT)
# =========================
df_prod["flag_pa_missing"] = df_prod["prix_achat_ht"].fillna(0) == 0

df_prod["marge_eur"] = np.where(
    df_prod["flag_pa_missing"],
    np.nan,
    df_prod["prix_vente_ht"] - df_prod["prix_achat_ht"]
)

df_prod["marge_pct"] = np.where(
    df_prod["flag_pa_missing"],
    np.nan,
    (df_prod["marge_eur"] / df_prod["prix_vente_ht"].replace(0, np.nan)) * 100
).round(2)

# =========================
# MERGE PRODUITS / VENTES
# =========================

merge_keys = ["id_product", "id_product_attribute"]

# conversion annee si elle existe
if "annee" in df_sales.columns:
    df_sales["annee"] = pd.to_numeric(df_sales["annee"], errors="coerce")

df = df_prod.merge(
    df_sales,
    on=["id_product", "id_product_attribute"],
    how="left"
)

# =========================
# GARDE-FOUS COLONNES
# =========================
for col, default in {
    "pourcentage_nouveaux_clients": 0.0,
    "taux_reachat_pourcent": 0.0,
    "qte_30j": 0.0,
    "rotation_30j": 0.0,
    "rotation_60j": 0.0,
    "rotation_90j": 0.0,
    "age_moyen": np.nan,
}.items():
    if col not in df.columns:
        df[col] = default

# conversions num
for col in [
    "pourcentage_nouveaux_clients", "taux_reachat_pourcent",
    "qte_30j", "rotation_30j", "rotation_60j", "rotation_90j", "age_moyen"
]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# =========================
# NORMALISATION (percentiles)
# =========================
def pct_rank(s: pd.Series) -> pd.Series:
    return s.rank(pct=True)

df["p_new"] = pct_rank(df["pourcentage_nouveaux_clients"].fillna(0))
df["p_stock"] = pct_rank(pd.to_numeric(df.get("stock", 0), errors="coerce").fillna(0))
df["p_qte30"] = pct_rank(df["qte_30j"].fillna(0))
df["p_marge"] = pct_rank(df["marge_pct"].fillna(0))
df["p_reachat"] = pct_rank(df["taux_reachat_pourcent"].fillna(0))

# =========================
# SCORES (base)
# =========================
df["score_acquisition"] = (
    0.40 * df["p_new"] +
    0.25 * df["p_qte30"] +
    0.20 * df["p_stock"] +
    0.15 * df["p_marge"]
) * 100

df["score_fidelisation"] = (
    0.35 * df["p_reachat"] +
    0.25 * df["p_marge"] +
    0.20 * df["p_stock"] +
    0.20 * df["p_qte30"]
) * 100

# =========================
# ELIGIBILITÉ
# =========================
df["eligible"] = (
    (pd.to_numeric(df["prix_vente_ttc"], errors="coerce").fillna(0) >= SEUIL_PRIX_PROMO) &
    (pd.to_numeric(df.get("stock", 0), errors="coerce").fillna(0) >= STOCK_MIN)
)

# =========================
# EXPORTS
# =========================
df.to_csv(os.path.join(EXPORT_DIR, "table_scoring.csv"), index=False)

df[df["eligible"]].sort_values("score_acquisition", ascending=False).head(TOP_N).to_csv(
    os.path.join(EXPORT_DIR, "export_acquisition.csv"),
    index=False
)

df[df["eligible"]].sort_values("score_fidelisation", ascending=False).head(TOP_N).to_csv(
    os.path.join(EXPORT_DIR, "export_fidelisation.csv"),
    index=False
)

print("Exports générés.")
