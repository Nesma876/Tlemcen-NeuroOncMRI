# =============================================================================
# XAI (occlusion, model-agnostic) + TESTS STATISTIQUES FORMELS
#
# Choix assume : pas de SHAP/Grad-CAM classique sur les embeddings CamemBERT/
# BiomedCLIP (denses, non spatiaux pour le texte, attention interne peu
# accessible pour un ViT geleu via open_clip) -> occlusion, model-agnostic,
# fonctionne avec n'importe quel pipeline extracteur+classifieur.
# =============================================================================

import numpy as np
import pandas as pd
from scipy import stats


# -----------------------------------------------------------------------
# 1. XAI TEXTE — importance par occlusion de mots
# -----------------------------------------------------------------------
def word_occlusion_importance(text, clf, scaler, embed_fn, baseline_proba):
    """
    text : chaine de caracteres (rapport d'un patient)
    clf, scaler : le classifieur + scaler DEJA entraines (sur le pipeline complet)
    embed_fn : fonction qui prend une liste de textes et retourne des embeddings
    baseline_proba : proba predite sur le texte complet (reference)

    Retourne un DataFrame mot -> delta de probabilite quand il est retire.
    """
    words = text.split()
    results = []
    for i in range(len(words)):
        text_occluded = " ".join(words[:i] + words[i+1:])
        emb = embed_fn([text_occluded])
        emb_scaled = scaler.transform(emb)
        proba_occluded = clf.predict_proba(emb_scaled)[0, 1]
        delta = baseline_proba - proba_occluded  # positif = le mot poussait vers CS
        results.append({"mot": words[i], "delta_proba": delta})
    return pd.DataFrame(results).sort_values("delta_proba", key=abs, ascending=False)


# -----------------------------------------------------------------------
# 2. XAI IMAGE — saillance par occlusion de patches
# -----------------------------------------------------------------------
def image_occlusion_saliency(image_pil, clf, scaler, encode_fn, baseline_proba,
                              patch_size=32, stride=32):
    """
    image_pil : image PIL RGB
    encode_fn : fonction qui prend une image PIL preprocessee et retourne l'embedding
    Retourne une carte de saillance (meme taille que l'image, submantillonnee par patch)
    """
    import numpy as np
    w, h = image_pil.size
    saliency = np.zeros((h // stride + 1, w // stride + 1))

    arr = np.array(image_pil)
    for yi, y in enumerate(range(0, h, stride)):
        for xi, x in enumerate(range(0, w, stride)):
            occluded = arr.copy()
            occluded[y:y+patch_size, x:x+patch_size] = 127  # gris neutre
            from PIL import Image as PILImage
            occluded_img = PILImage.fromarray(occluded)
            emb = encode_fn(occluded_img)
            emb_scaled = scaler.transform(emb.reshape(1, -1))
            proba_occluded = clf.predict_proba(emb_scaled)[0, 1]
            saliency[yi, xi] = baseline_proba - proba_occluded

    return saliency


# -----------------------------------------------------------------------
# 3. Contribution de modalite (fusion guidee) — deja calculable, gratuit
# -----------------------------------------------------------------------
def analyze_guided_fusion_weights(proba_text, proba_image, y, patient_ids):
    from scipy.stats import zscore
    text_confidence = 1 / (1 + np.exp(-zscore(proba_text)))
    agreement = 1 - np.abs(text_confidence - proba_image)
    w_text = agreement / (agreement + (1 - agreement) + 1e-9)

    df = pd.DataFrame({
        "ID_Patient": patient_ids,
        "y_true": y,
        "poids_texte": w_text,
        "poids_image": 1 - w_text,
        "proba_texte": proba_text,
        "proba_image": proba_image,
    })
    print(f"Poids moyen accorde au texte : {w_text.mean():.3f}")
    print(f"Poids moyen accorde a l'image : {(1-w_text).mean():.3f}")
    print(f"\nPatients ou le texte domine largement (poids>0.7) : {(w_text>0.7).sum()}")
    print(f"Patients ou l'image domine largement (poids<0.3) : {(w_text<0.3).sum()}")
    return df


# -----------------------------------------------------------------------
# 4. TEST DE DELONG — comparaison statistique formelle de deux AUC appariees
# -----------------------------------------------------------------------
def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(preds_sorted_transposed, label_1_count):
    m = label_1_count
    n = preds_sorted_transposed.shape[1] - m
    positive = preds_sorted_transposed[:, :m]
    negative = preds_sorted_transposed[:, m:]
    k = preds_sorted_transposed.shape[0]

    tx = np.empty([k, m])
    ty = np.empty([k, n])
    tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = _compute_midrank(positive[r, :])
        ty[r, :] = _compute_midrank(negative[r, :])
        tz[r, :] = _compute_midrank(preds_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_roc_test(y_true, proba_a, proba_b):
    """Test de DeLong pour comparer deux AUC APPARIEES (memes patients).
    Retourne (diff_auc, p_value)."""
    order = np.argsort(-y_true)
    y_sorted = y_true[order]
    m = int(y_sorted.sum())  # nb positifs

    preds = np.vstack([proba_a[order], proba_b[order]])
    aucs, delongcov = _fast_delong(preds, m)

    diff = aucs[0] - aucs[1]
    var = delongcov[0, 0] + delongcov[1, 1] - 2 * delongcov[0, 1]
    if var <= 1e-10:
        # Modeles quasi-identiques (variance nulle) -> pas de difference detectable
        return diff, 1.0
    z = diff / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return diff, p
