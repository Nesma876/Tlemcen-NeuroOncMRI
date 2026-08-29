# =============================================================================
# SOURCE : cellule 49 du notebook de reference (notebooks/full_pipeline_reference.ipynb)
# Script REEL, exécuté sur Kaggle, ayant produit les chiffres officiels du
# manuscrit :
#   TF-IDF sans feature a risque : AUC = 0.802 +/- 0.029 (6-seed mean)
#   TF-IDF avec feature a risque : AUC = 0.967 +/- 0.011 (6-seed mean)
#   Delta AUC (fuite)            : +0.166 +/- 0.033
#   CamemBERT                    : AUC = 0.775 (deterministe)
#
# C'est la specification PRINCIPALE du masquage (pluriel/primitif pris en compte,
# verifie a 0 terme residuel), avec protocole multi-seed complet (6 seeds :
# 1, 2, 3, 4, 42, 777).
# =============================================================================

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import re
from functools import partial
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

SEED = 42
np.random.seed(SEED)

# ============================================================
# CONFIG + DONNEES
# ============================================================
EXCEL_PATH = Path(os.environ.get(
    "TNO_METADATA_FILE",
    "/kaggle/input/datasets/souaadrahmoun/datasetcac/ClasseurPFE1 (1).xlsx",
))
OUTPUT_DIR = Path(os.environ.get("TNO_OUTPUT_DIR", "/kaggle/working"))

if not EXCEL_PATH.exists():
    raise SystemExit(
        f"Metadata file not found: {EXCEL_PATH}\n"
        "Download the external dataset and pass --metadata-file to the wrapper "
        "script, or set TNO_METADATA_FILE. See DATA.md."
    )
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_excel(EXCEL_PATH, sheet_name="Feuil1")
for col in ["Origine_Tumeur", "Sexe", "Localisation", "Métastase_Origine", "ANNOTATION", "Rapport"]:
    df[col] = df[col].astype(str).fillna("").replace("nan", "")

y = (df["Origine_Tumeur"].str.strip() == "Secondaire").astype(int).values
patient_ids = df["ID_Patient"].values
print(f"Patients : {len(y)} (CP={sum(y==0)}, CS={sum(y==1)})")

# ============================================================
# SPECIFICATION PRINCIPALE DE MASQUAGE (regex fixe : \w* au lieu de rien, + primitif ajoute)
# ============================================================
CLASS_REVEALING_TERMS_FIXED = [
    r"primaire\w*", r"secondaire\w*", r"m[ée]tastase\w*", r"m[ée]tastatique\w*",
    r"primitif\w*", r"primitiv\w*",
]

def mask_class_revealing_terms_fixed(text_series):
    out = text_series.astype(str)
    for pat in CLASS_REVEALING_TERMS_FIXED:
        out = out.str.replace(pat, " CLASSE ", regex=True, flags=re.IGNORECASE)
    return out

text_raw_fixed = mask_class_revealing_terms_fixed(df["ANNOTATION"] + " " + df["Rapport"])
residual = text_raw_fixed.str.contains(r"secondaires?\b|primaires?\b|primitif\w*", case=False, regex=True).sum()
print(f"Verification masquage : {residual}/45 termes residuels (doit etre 0)")

# ============================================================
# FEATURES (specification de reference)
# ============================================================
def build_safe_manual_features(df):
    feat = pd.DataFrame(index=df.index)
    feat["Age"] = pd.to_numeric(df["Âge"] if "Âge" in df.columns else df["Age"], errors="coerce")
    feat["Sexe"] = LabelEncoder().fit_transform(df["Sexe"].astype(str))
    feat["Localisation"] = LabelEncoder().fit_transform(df["Localisation"].astype(str))
    annot = mask_class_revealing_terms_fixed(df["ANNOTATION"])
    nlp_patterns = {
        "nlp_necrose": r"n[ée]cros", "nlp_oedeme": r"[oœ]d[eè]me",
        "nlp_effet_masse": r"effet de masse", "nlp_rehaussement": r"rehaussement",
        "nlp_saignement": r"saignement|h[ée]morrag", "nlp_multiple": r"multiple|plusieurs",
        "nlp_unique": r"\bunique\b", "nlp_engagement": r"engagement|sous-falc",
        "nlp_spectro": r"spectroscop", "nlp_annulaire": r"annulaire",
        "nlp_corps_calleux": r"corps calleux", "nlp_jonction_cortico": r"jonction cortico|cortico-sous-cortical",
        "nlp_infiltrant": r"infiltrant\w*",
    }
    annot_lower = annot.str.lower()
    for name, pat in nlp_patterns.items():
        feat[name] = annot_lower.str.contains(pat, na=False).astype(int)
    return feat

def build_leaky_features(df):
    feat = pd.DataFrame(index=df.index)
    meta_vide = ["", "nan", "none", "non applicable", "non précisée", "-", "n/a"]
    feat["a_metastase"] = (~df["Métastase_Origine"].astype(str).str.strip().str.lower().isin(meta_vide)).astype(int)
    return feat

class DenseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return X.toarray() if hasattr(X, "toarray") else X

def run_loocv_text_model(manual_feat, text_raw, y, k_best=20, seed=SEED):
    n = len(y)
    loo = LeaveOneOut()
    oof_proba = np.zeros(n)
    manual_arr = manual_feat.values.astype(float)
    text_arr = text_raw.values
    for train_idx, test_idx in loo.split(manual_arr):
        try:
            tfidf = TfidfVectorizer(max_features=150, ngram_range=(1, 2), min_df=2, max_df=0.90, sublinear_tf=True)
            X_tfidf_train = tfidf.fit_transform(text_arr[train_idx])
        except ValueError:
            tfidf = TfidfVectorizer(max_features=150, ngram_range=(1, 1), min_df=1, max_df=1.0, sublinear_tf=True)
            X_tfidf_train = tfidf.fit_transform(text_arr[train_idx])
        X_tfidf_test = tfidf.transform(text_arr[test_idx])
        dense = DenseTransformer()
        X_tfidf_train = dense.transform(X_tfidf_train)
        X_tfidf_test = dense.transform(X_tfidf_test)
        med = np.nanmedian(manual_arr[train_idx], axis=0)
        Xm_train = np.where(np.isnan(manual_arr[train_idx]), med, manual_arr[train_idx])
        Xm_test = np.where(np.isnan(manual_arr[test_idx]), med, manual_arr[test_idx])
        X_train_full = np.hstack([Xm_train, X_tfidf_train])
        X_test_full = np.hstack([Xm_test, X_tfidf_test])
        scaler = StandardScaler()
        X_train_full = scaler.fit_transform(X_train_full)
        X_test_full = scaler.transform(X_test_full)
        mi_scorer = partial(mutual_info_classif, random_state=seed)
        k = min(k_best, X_train_full.shape[1])
        selector = SelectKBest(mi_scorer, k=k)
        X_train_sel = selector.fit_transform(X_train_full, y[train_idx])
        X_test_sel = selector.transform(X_test_full)
        clf = RandomForestClassifier(n_estimators=300, max_depth=5, class_weight="balanced", random_state=seed)
        clf.fit(X_train_sel, y[train_idx])
        oof_proba[test_idx] = clf.predict_proba(X_test_sel)[:, 1]
    return oof_proba

def run_loocv_on_text_embeddings(embeddings, y, seed=SEED):
    n = len(y)
    loo = LeaveOneOut()
    oof_proba = np.zeros(n)
    for train_idx, test_idx in loo.split(embeddings):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(embeddings[train_idx])
        X_test = scaler.transform(embeddings[test_idx])
        clf = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0, random_state=seed)
        clf.fit(X_train, y[train_idx])
        oof_proba[test_idx] = clf.predict_proba(X_test)[:, 1]
    return oof_proba

def summarize(y, proba, thr=0.5):
    pred = (proba >= thr).astype(int)
    return {"AUC": roc_auc_score(y, proba), "Accuracy": accuracy_score(y, pred),
            "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred, zero_division=0),
            "F1": f1_score(y, pred, zero_division=0, average="macro")}

safe_feat = build_safe_manual_features(df)
leaky_feat = build_leaky_features(df)
combined_feat = pd.concat([safe_feat, leaky_feat], axis=1)

# ============================================================
# CAMEMBERT — embeddings extraits UNE FOIS (deterministe, pas besoin de multi-seed sur l'extraction)
# ============================================================
try:
    from transformers import AutoTokenizer, AutoModel
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'transformers'. Install project dependencies with "
        "`pip install -r requirements.txt` before running this benchmark."
    ) from exc

device = "cpu"
tokenizer = AutoTokenizer.from_pretrained("camembert-base")
camembert = AutoModel.from_pretrained("camembert-base").to(device)
camembert.eval()

def extract_camembert_embeddings(text_series, tokenizer, model, device, max_length=256):
    embeddings = []
    with torch.no_grad():
        for text in text_series:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, padding=True).to(device)
            outputs = model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            summed = (outputs.last_hidden_state * mask).sum(1)
            counts = mask.sum(1).clamp(min=1e-9)
            embeddings.append((summed / counts).squeeze(0).cpu().numpy())
    return np.stack(embeddings)

camembert_embeddings_fixed = extract_camembert_embeddings(text_raw_fixed, tokenizer, camembert, device)
print("Embeddings CamemBERT extraits (deterministe, une seule fois)")

# ============================================================
# MULTI-SEED (6 seeds), specification principale de masquage
# ============================================================
seeds_to_test = [1, 2, 3, 4, 42, 777]
results_multiseed = []

for s in seeds_to_test:
    print(f"\n{'='*20} SEED {s} {'='*20}")

    proba_safe_s = run_loocv_text_model(safe_feat, text_raw_fixed, y, seed=s)
    proba_leaky_s = run_loocv_text_model(combined_feat, text_raw_fixed, y, seed=s)
    proba_camembert_s = run_loocv_on_text_embeddings(camembert_embeddings_fixed, y, seed=s)

    auc_safe = summarize(y, proba_safe_s)["AUC"]
    auc_leaky = summarize(y, proba_leaky_s)["AUC"]
    auc_camembert = summarize(y, proba_camembert_s)["AUC"]

    print(f"Safe (sans feature a risque)   : AUC={auc_safe:.4f}")
    print(f"Leaky (avec feature a risque)  : AUC={auc_leaky:.4f}")
    print(f"CamemBERT                      : AUC={auc_camembert:.4f}")

    results_multiseed.append({
        "seed": s, "AUC_safe": auc_safe, "AUC_leaky": auc_leaky, "AUC_camembert": auc_camembert
    })

# ============================================================
# RESUME
# ============================================================
results_df = pd.DataFrame(results_multiseed)
print("\n" + "="*70)
print("RESUME MULTI-SEED (specification principale de masquage)")
print("="*70)
summary = results_df.drop(columns="seed").agg(["mean", "std", "min", "max"])
print(summary.to_string())

delta_per_seed = results_df["AUC_leaky"] - results_df["AUC_safe"]
print(f"\nDelta AUC (fuite) par seed : {delta_per_seed.tolist()}")
print(f"Delta AUC moyen : {delta_per_seed.mean():.4f} +/- {delta_per_seed.std():.4f}")

results_df.to_csv(OUTPUT_DIR / "multiseed_principal_masking.csv", index=False)
print(f"\nExporte -> {OUTPUT_DIR / 'multiseed_principal_masking.csv'}")
