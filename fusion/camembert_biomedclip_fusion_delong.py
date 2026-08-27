# =============================================================================
# SOURCE : cellule 46 du notebook original (notebooks/full_pipeline_original.ipynb)
# Script REEL, exécuté sur Kaggle, ayant produit les chiffres de fusion
# officiels du manuscrit (Table 5) :
#   CamemBERT seul   : AUC = 0.763 (reference-run, PRE-correction masquage)
#   BiomedCLIP seul  : AUC = 0.807
#   Fusion guidee    : AUC = 0.914, DeLong vs BiomedCLIP : delta=+0.107, p=0.099
#
# ATTENTION VERSIONING (documente dans le manuscrit, Section 5) : ce script
# utilise le masquage de texte NON corrige (CLASS_REVEALING_TERMS avec \b,
# pas \w*) -- c'est pourquoi CamemBERT donne ici 0.763 et non 0.775 (valeur
# corrigee, voir models/text/tfidf_camembert_corrected_multiseed.py). Les
# resultats de fusion sont donc rapportes comme EXPLORATOIRES dans le
# manuscrit, pas comme extension directe du benchmark texte corrige.
# =============================================================================

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import re, glob, random, warnings, gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from scipy import stats
from scipy.stats import zscore

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

# ============================================================
# CONFIG
# ============================================================
class Config:
    EXCEL_PATH = Path(os.environ.get(
        "TNO_METADATA_FILE",
        "/kaggle/input/datasets/souaadrahmoun/datasetcac/ClasseurPFE1 (1).xlsx",
    ))
    IMAGE_DIR = Path(os.environ.get(
        "TNO_IMAGE_DIR",
        "/kaggle/input/datasets/souaadrahmoun/datacacfinal/images_v2_final/data p_s",
    ))
    OUTPUT_DIR = Path(os.environ.get("TNO_OUTPUT_DIR", "/kaggle/working"))
    IMG_SIZE = 224
    MAX_IMGS_PER_PATIENT = 30

cfg = Config()
for p in [cfg.EXCEL_PATH, cfg.IMAGE_DIR]:
    print(p, "->", p.exists())
if not cfg.EXCEL_PATH.exists() or not cfg.IMAGE_DIR.exists():
    raise SystemExit(
        "Required dataset files were not found. Pass --metadata-file and "
        "--image-dir to the wrapper script, or set TNO_METADATA_FILE and "
        "TNO_IMAGE_DIR. See DATA.md."
    )
cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# DONNEES + LABELS
# ============================================================
df = pd.read_excel(cfg.EXCEL_PATH, sheet_name="Feuil1")
for col in ["Origine_Tumeur", "ANNOTATION", "Rapport"]:
    df[col] = df[col].astype(str).fillna("").replace("nan", "")

y = (df["Origine_Tumeur"].str.strip() == "Secondaire").astype(int).values
patient_ids = df["ID_Patient"].values

CLASS_REVEALING_TERMS = [r"\bprimaire\b", r"\bsecondaire\b", r"m[ée]tastase\w*", r"m[ée]tastatique\w*"]
def mask_class_revealing_terms(text_series):
    out = text_series.astype(str)
    for pat in CLASS_REVEALING_TERMS:
        out = out.str.replace(pat, " CLASSE ", regex=True, flags=re.IGNORECASE)
    return out

text_raw = mask_class_revealing_terms(df["ANNOTATION"] + " " + df["Rapport"])
print(f"Patients : {len(y)} (CP={sum(y==0)}, CS={sum(y==1)})")

def summarize(y, proba, thr=0.5):
    pred = (proba >= thr).astype(int)
    return {"AUC": roc_auc_score(y, proba), "Accuracy": accuracy_score(y, pred),
            "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred, zero_division=0),
            "F1": f1_score(y, pred, zero_division=0, average="macro")}

# ============================================================
# CAMEMBERT (TEXTE)
# ============================================================
try:
    from transformers import AutoTokenizer, AutoModel
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'transformers'. Install project dependencies with "
        "`pip install -r requirements.txt` before running fusion."
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

camembert_embeddings = extract_camembert_embeddings(text_raw, tokenizer, camembert, device)
print("CamemBERT embeddings:", camembert_embeddings.shape)

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

proba_camembert = run_loocv_on_text_embeddings(camembert_embeddings, y, seed=SEED)
print("Texte (CamemBERT):", summarize(y, proba_camembert))

# ============================================================
# BIOMEDCLIP (IMAGE) + CLASSIFICATION SLICE-LEVEL
# ============================================================
try:
    import open_clip
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'open_clip_torch'. Install project dependencies with "
        "`pip install -r requirements.txt` before running fusion."
    ) from exc
from PIL import Image

device_img = "cpu"

try:
    model, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    model_name_used = "BiomedCLIP"
except Exception as e:
    print(f"BiomedCLIP indisponible ({e}), repli CLIP general")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model_name_used = "CLIP general"

model.eval().to(device_img)
print("Modele image:", model_name_used)

def list_patient_images(patient_id, image_dir):
    pid_str = f"P{patient_id:02d}"
    files = []
    for cls in ["CP", "CS"]:
        cls_dir = image_dir / cls
        if cls_dir.exists():
            files += sorted(glob.glob(str(cls_dir / f"{pid_str} (*).jpg")))
            files += sorted(glob.glob(str(cls_dir / f"{pid_str} (*).png")))
    return files

def extract_per_image_embeddings_foundation(patient_ids, image_dir, model, preprocess, device,
                                             max_imgs_per_patient=30, seed=SEED):
    rng = random.Random(seed)
    embeddings_per_patient = {}
    for pid in patient_ids:
        files = list_patient_images(int(pid), image_dir)
        if len(files) > max_imgs_per_patient:
            files = rng.sample(files, max_imgs_per_patient)
        imgs = torch.stack([preprocess(Image.open(f).convert("RGB")) for f in files]).to(device)
        with torch.no_grad():
            feats = model.encode_image(imgs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        embeddings_per_patient[int(pid)] = feats.cpu().numpy()
        print(f"Patient {pid}: {len(files)} embeddings extraits")
    return embeddings_per_patient

def run_loocv_slicelevel(patient_ids, y, embeds_per_patient, seed=SEED, agg="mean"):
    n = len(y)
    loo = LeaveOneOut()
    oof_proba = np.zeros(n)
    for train_idx, test_idx in loo.split(patient_ids):
        train_pids = patient_ids[train_idx]
        test_pid = patient_ids[test_idx][0]
        X_train_slices, y_train_slices = [], []
        for pid, label in zip(train_pids, y[train_idx]):
            embeds = embeds_per_patient[int(pid)]
            X_train_slices.append(embeds)
            y_train_slices.append(np.full(len(embeds), label))
        X_train_slices = np.vstack(X_train_slices)
        y_train_slices = np.concatenate(y_train_slices)
        scaler = StandardScaler()
        X_train_slices = scaler.fit_transform(X_train_slices)
        clf = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0, random_state=seed)
        clf.fit(X_train_slices, y_train_slices)
        X_test_slices = scaler.transform(embeds_per_patient[int(test_pid)])
        proba_slices = clf.predict_proba(X_test_slices)[:, 1]
        oof_proba[test_idx] = np.mean(proba_slices) if agg == "mean" else np.median(proba_slices)
    return oof_proba

embeds_foundation = extract_per_image_embeddings_foundation(
    patient_ids, cfg.IMAGE_DIR, model, preprocess, device_img,
    max_imgs_per_patient=cfg.MAX_IMGS_PER_PATIENT, seed=SEED
)
proba_foundation = run_loocv_slicelevel(patient_ids, y, embeds_foundation, seed=SEED)
print("Image (BiomedCLIP):", summarize(y, proba_foundation))

# ============================================================
# FUSION GUIDEE + STATISTIQUE (DELONG)
# ============================================================
def fusion_moyenne_simple(branches):
    return np.stack(list(branches.values()), axis=1).mean(axis=1)

def run_guided_fusion(proba_text, proba_image, y):
    text_confidence = 1 / (1 + np.exp(-zscore(proba_text)))
    agreement = 1 - np.abs(text_confidence - proba_image)
    w_text = agreement / (agreement + (1 - agreement) + 1e-9)
    return w_text * proba_text + (1 - w_text) * proba_image

proba_guided = run_guided_fusion(proba_camembert, proba_foundation, y)
print("Fusion guidee:", summarize(y, proba_guided))

def _compute_midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x)
    T = np.zeros(N, dtype=float); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1; i = j
    T2 = np.empty(N, dtype=float); T2[J] = T
    return T2

def _fast_delong(preds, m):
    n = preds.shape[1] - m
    positive, negative = preds[:, :m], preds[:, m:]
    k = preds.shape[0]
    tx, ty, tz = np.empty([k,m]), np.empty([k,n]), np.empty([k,m+n])
    for r in range(k):
        tx[r,:] = _compute_midrank(positive[r,:])
        ty[r,:] = _compute_midrank(negative[r,:])
        tz[r,:] = _compute_midrank(preds[r,:])
    aucs = tz[:,:m].sum(axis=1)/m/n - float(m+1)/(2*n)
    v01 = (tz[:,:m]-tx)/n; v10 = 1.0-(tz[:,m:]-ty)/m
    delongcov = np.cov(v01)/m + np.cov(v10)/n
    return aucs, delongcov

def delong_roc_test(y_true, proba_a, proba_b):
    order = np.argsort(-y_true)
    m = int(y_true[order].sum())
    preds = np.vstack([proba_a[order], proba_b[order]])
    aucs, cov = _fast_delong(preds, m)
    diff = aucs[0]-aucs[1]
    var = cov[0,0]+cov[1,1]-2*cov[0,1]
    if var <= 1e-10: return diff, 1.0
    z = diff/np.sqrt(var)
    return diff, 2*(1-stats.norm.cdf(abs(z)))

diff_auc, p_value = delong_roc_test(y, proba_guided, proba_foundation)
print(f"\nFusion guidee vs Image seule : diff={diff_auc:+.4f}, p={p_value:.4f}")
print("Significatif (p<0.05)" if p_value < 0.05 else "NON significatif")

# ============================================================
# EXPORT FINAL
# ============================================================
final_export = pd.DataFrame({
    "ID_Patient": patient_ids, "y_true": y,
    "proba_text_camembert": proba_camembert,
    "proba_image_biomedclip": proba_foundation,
    "proba_fusion_guided": proba_guided,
    "proba_fusion_moyenne_simple": fusion_moyenne_simple({"t": proba_camembert, "i": proba_foundation}),
})
final_export.to_csv(cfg.OUTPUT_DIR / "final_probabilities_for_plots.csv", index=False)
print("\n" + final_export.to_string(index=False))
print(f"\nExporte -> {cfg.OUTPUT_DIR / 'final_probabilities_for_plots.csv'}")
