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
FILE_242 = "242_catalogue_snapshot.csv"

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

def optional_existing_path(*candidates: str) -> str | None:
    for name in candidates:
        path = os.path.join(DATA_DIR, name)
        if os.path.exists(path):
            return path
    return None

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

def build_yearly_snapshot_df(path_242: str, merge_keys: list[str]) -> pd.DataFrame:
    df_snap = pd.read_csv(path_242, sep=";")
    df_snap = normalize_ean(df_snap)

    if "snapshot_date" not in df_snap.columns:
        raise KeyError("La colonne 'snapshot_date' est absente du fichier 242.")

    df_snap["snapshot_date"] = pd.to_datetime(df_snap["snapshot_date"], errors="coerce")
    df_snap = df_snap[df_snap["snapshot_date"].notna()].copy()
    df_snap["annee"] = df_snap["snapshot_date"].dt.year.astype(int)

    if "active" in df_snap.columns:
        df_snap = df_snap[df_snap["active"] == 1].copy()

    for col in ["prix_vente_ht", "prix_achat_ht", "stock"]:
        if col in df_snap.columns:
            df_snap[col] = pd.to_numeric(df_snap[col], errors="coerce")

    snapshot_cols = [
        "annee", "snapshot_date", *merge_keys,
        "ean", "categorie", "marque", "nom",
        "stock", "prix_achat_ht", "prix_vente_ht",
    ]
    snapshot_cols = [c for c in snapshot_cols if c in df_snap.columns]

    df_snap = (
        df_snap[snapshot_cols]
        .sort_values("snapshot_date")
        .drop_duplicates(subset=["annee", *merge_keys], keep="last")
        .drop(columns=["snapshot_date"], errors="ignore")
    )

    return df_snap

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
path_242 = optional_existing_path(FILE_242, "242_catalogue_snapshot.csv")

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

merge_keys = ["id_product", "id_product_attribute"]

# conversion annee si elle existe
if "annee" in df_sales.columns:
    df_sales["annee"] = pd.to_numeric(df_sales["annee"], errors="coerce")

# =========================
# MERGE PRODUITS / VENTES
# =========================
if path_242:
    df_snapshot_yearly = build_yearly_snapshot_df(path_242, merge_keys)

    df = df_sales.merge(
        df_snapshot_yearly,
        on=["annee", *merge_keys],
        how="left"
    )

    current_cols = merge_keys + [
        c for c in [
            "ean", "categorie", "marque", "nom",
            "stock", "prix_achat_ht", "prix_vente_ht",
            "qte_30j", "rotation_30j", "rotation_60j", "rotation_90j",
        ] if c in df_prod.columns
    ]
    df_prod_current = df_prod[current_cols].drop_duplicates(subset=merge_keys)

    df = df.merge(
        df_prod_current,
        on=merge_keys,
        how="left",
        suffixes=("_snap", "_cur")
    )

    for col in ["ean", "categorie", "marque", "nom", "stock", "prix_achat_ht", "prix_vente_ht"]:
        snap_col = f"{col}_snap"
        cur_col = f"{col}_cur"
        if snap_col in df.columns and cur_col in df.columns:
            df[col] = df[snap_col].combine_first(df[cur_col])
        elif snap_col in df.columns:
            df[col] = df[snap_col]
        elif cur_col in df.columns:
            df[col] = df[cur_col]

    df = df.drop(
        columns=[f"{c}_snap" for c in ["ean", "categorie", "marque", "nom", "stock", "prix_achat_ht", "prix_vente_ht"]] +
                [f"{c}_cur" for c in ["ean", "categorie", "marque", "nom", "stock", "prix_achat_ht", "prix_vente_ht"]],
        errors="ignore"
    )

    sold_keys = df_sales[merge_keys].drop_duplicates()
    df_unsold = df_prod_current.merge(sold_keys, on=merge_keys, how="left", indicator=True)
    df_unsold = df_unsold[df_unsold["_merge"] == "left_only"].drop(columns=["_merge"])

    for col in df.columns:
        if col not in df_unsold.columns:
            df_unsold[col] = np.nan
    for col in df_unsold.columns:
        if col not in df.columns:
            df[col] = np.nan

    df = pd.concat([df, df_unsold[df.columns]], ignore_index=True, sort=False)
else:
    df = df_prod.merge(
        df_sales,
        on=merge_keys,
        how="left"
    )

# =========================
# TVA / MARGE (sur dataset final)
# =========================
for col in ["prix_vente_ht", "prix_achat_ht", "stock"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

if "prix_vente_ht" not in df.columns:
    raise KeyError("La colonne 'prix_vente_ht' est absente du dataset final.")

df["tva_rate"] = df.apply(detect_tva_rate, axis=1)

df["prix_vente_ttc"] = np.where(
    df["tva_rate"] == 0.0,
    df["prix_vente_ht"],
    df["prix_vente_ht"] * (1 + df["tva_rate"])
)

df["flag_pa_missing"] = df["prix_achat_ht"].fillna(0) == 0

df["marge_eur"] = np.where(
    df["flag_pa_missing"],
    np.nan,
    df["prix_vente_ht"] - df["prix_achat_ht"]
)

df["marge_pct"] = np.where(
    df["flag_pa_missing"],
    np.nan,
    (df["marge_eur"] / df["prix_vente_ht"].replace(0, np.nan)) * 100
).round(2)

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
