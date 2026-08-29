# =============================================================================
# SOURCE : cellule 47 du notebook de reference (notebooks/full_pipeline_reference.ipynb)
# Script REEL, exécuté sur Kaggle, ayant produit les figures d'occlusion du
# manuscrit (word-occlusion importance, patch-occlusion saliency map) pour
# le patient CS illustratif (Section 4.8, Fig. 6-7).
#
# Entraine un classifieur final sur les 45 patients A BUT INTERPRETATIF
# UNIQUEMENT (pas pour rapporter une performance -- la performance
# officielle vient du LOO-CV, voir models/text/ et fusion/).
# =============================================================================

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import re, glob, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

SEED = 42
np.random.seed(SEED)

# ============================================================
# CONFIG + DONNEES
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

cfg = Config()
if not cfg.EXCEL_PATH.exists() or not cfg.IMAGE_DIR.exists():
    raise SystemExit(
        "Required dataset files were not found. Pass --metadata-file and "
        "--image-dir to the wrapper script, or set TNO_METADATA_FILE and "
        "TNO_IMAGE_DIR. See DATA.md."
    )
cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
print(f"Patients : {len(y)}")

# ============================================================
# CAMEMBERT — modele final (sur les 45 patients, but interpretatif uniquement)
# ============================================================
try:
    from transformers import AutoTokenizer, AutoModel
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'transformers'. Install project dependencies with "
        "`pip install -r requirements.txt` before running XAI."
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

scaler_text = StandardScaler()
X_text_final = scaler_text.fit_transform(camembert_embeddings)
clf_text_final = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED)
clf_text_final.fit(X_text_final, y)
print("Classifieur texte final entraine (interpretation uniquement)")

# ============================================================
# BIOMEDCLIP — modele final image
# ============================================================
try:
    import open_clip
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'open_clip_torch'. Install project dependencies with "
        "`pip install -r requirements.txt` before running XAI."
    ) from exc
from PIL import Image

device_img = "cpu"
try:
    model_img, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
except Exception as e:
    print(f"BiomedCLIP indisponible ({e}), repli CLIP general")
    model_img, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
model_img.eval().to(device_img)

def list_patient_images(patient_id, image_dir):
    pid_str = f"P{patient_id:02d}"
    files = []
    for cls in ["CP", "CS"]:
        cls_dir = image_dir / cls
        if cls_dir.exists():
            files += sorted(glob.glob(str(cls_dir / f"{pid_str} (*).jpg")))
            files += sorted(glob.glob(str(cls_dir / f"{pid_str} (*).png")))
    return files

def embed_image(pil_img):
    with torch.no_grad():
        t = preprocess(pil_img).unsqueeze(0).to(device_img)
        feats = model_img.encode_image(t)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().squeeze(0)

# Embeddings slice-level pour TOUS les patients (modele final, interpretation)
rng = random.Random(SEED)
X_img_slices, y_img_slices, patient_of_slice = [], [], []
for pid, label in zip(patient_ids, y):
    files = list_patient_images(int(pid), cfg.IMAGE_DIR)
    if len(files) > 30:
        files = rng.sample(files, 30)
    for f in files:
        X_img_slices.append(embed_image(Image.open(f).convert("RGB")))
        y_img_slices.append(label)
        patient_of_slice.append(pid)
X_img_slices = np.array(X_img_slices)
y_img_slices = np.array(y_img_slices)

scaler_img = StandardScaler()
X_img_scaled = scaler_img.fit_transform(X_img_slices)
clf_img_final = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED)
clf_img_final.fit(X_img_scaled, y_img_slices)
print("Classifieur image final entraine (interpretation uniquement)")

# ============================================================
# XAI TEXTE — occlusion de mots (1 exemple CS bien predit)
# ============================================================
def word_occlusion_importance(text, clf, scaler, tokenizer, model, device, baseline_proba):
    words = text.split()
    results = []
    for i in range(len(words)):
        text_occluded = " ".join(words[:i] + words[i+1:])
        emb = extract_camembert_embeddings(pd.Series([text_occluded]), tokenizer, model, device)
        emb_scaled = scaler.transform(emb)
        proba_occluded = clf.predict_proba(emb_scaled)[0, 1]
        results.append({"mot": words[i], "delta_proba": baseline_proba - proba_occluded})
    return pd.DataFrame(results).sort_values("delta_proba", key=abs, ascending=False)

example_idx = np.where(y == 1)[0][0]  # premier patient CS
baseline_text = clf_text_final.predict_proba(X_text_final[example_idx:example_idx+1])[0, 1]
print(f"\nPatient {patient_ids[example_idx]} (CS) — proba de base: {baseline_text:.3f}")

importance_words = word_occlusion_importance(
    text_raw.iloc[example_idx], clf_text_final, scaler_text, tokenizer, camembert, device, baseline_text
)
print("Top 15 mots les plus influents :")
print(importance_words.head(15).to_string(index=False))
importance_words.to_csv(cfg.OUTPUT_DIR / "word_occlusion_example.csv", index=False)

# ============================================================
# XAI IMAGE — occlusion de patches (1 image du meme patient)
# ============================================================
def image_occlusion_saliency(image_pil, clf, scaler, baseline_proba, patch_size=32, stride=32):
    w, h = image_pil.size
    arr = np.array(image_pil)
    saliency = []
    for y0 in range(0, h, stride):
        for x0 in range(0, w, stride):
            occluded = arr.copy()
            occluded[y0:y0+patch_size, x0:x0+patch_size] = 127
            occ_img = Image.fromarray(occluded)
            emb = embed_image(occ_img).reshape(1, -1)
            emb_scaled = scaler.transform(emb)
            proba_occluded = clf.predict_proba(emb_scaled)[0, 1]
            saliency.append({"y": y0, "x": x0, "delta_proba": baseline_proba - proba_occluded})
    return pd.DataFrame(saliency)

files_example = list_patient_images(int(patient_ids[example_idx]), cfg.IMAGE_DIR)
if files_example:
    img_example = Image.open(files_example[0]).convert("RGB")
    baseline_img = clf_img_final.predict_proba(scaler_img.transform(embed_image(img_example).reshape(1,-1)))[0,1]
    print(f"\nImage occlusion sur {files_example[0]} — proba de base: {baseline_img:.3f}")
    saliency_df = image_occlusion_saliency(img_example, clf_img_final, scaler_img, baseline_img)
    saliency_df.to_csv(cfg.OUTPUT_DIR / "image_occlusion_example.csv", index=False)
    print("Zones les plus influentes (delta_proba le plus fort) :")
    print(saliency_df.sort_values("delta_proba", key=abs, ascending=False).head(5).to_string(index=False))

print(f"\nExporte -> {cfg.OUTPUT_DIR / 'word_occlusion_example.csv'}, {cfg.OUTPUT_DIR / 'image_occlusion_example.csv'}")
