# =============================================================================
# SOURCE : cellule 37 du notebook original (notebooks/full_pipeline_original.ipynb)
# Script REEL, exécuté sur Kaggle, comparant les 5 strategies de fusion
# (Table 5 du manuscrit) sur les MEMES branches CamemBERT + BiomedCLIP.
# Contient run_loocv_early_fusion() et run_guided_fusion() -- les 2
# strategies qui n'existent QUE dans ce notebook (jamais sauvegardees
# comme fichiers autonomes avant cette extraction).
#
# Depend de fonctions definies dans une cellule precedente du notebook
# (fusion_moyenne_simple, fusion_pondere_loocv, fusion_stacking_loocv,
# summarize, bootstrap_ci, proba_camembert, proba_foundation, y, SEED) --
# voir notebooks/full_pipeline_original.ipynb pour le contexte complet.
#
# STATUT RELEASE PUBLIQUE : historique / PARTIAL. Ce fichier preserve la logique
# exacte de la cellule 37. Il n'est pas converti en CLI autonome car les
# embeddings et probabilites intermediaires requis ne sont pas serialises dans
# ce repository public.
# =============================================================================

from sklearn.decomposition import PCA
from scipy.stats import zscore

# ============================================================
# Rappel des branches individuelles
# ============================================================
print("Texte (CamemBERT)  :", summarize(y, proba_camembert))
print("Image (BiomedCLIP) :", summarize(y, proba_foundation))
print()

branches_v3 = {
    "texte_camembert": proba_camembert,
    "image_biomedclip": proba_foundation,
}

all_fusion_results = {}

# ============================================================
# 1-3. Fusion tardive (moyenne simple / ponderee / stacking)
# ============================================================
for nom, fn in [
    ("1. Moyenne simple", lambda: fusion_moyenne_simple(branches_v3)),
    ("2. Moyenne ponderee (LOO-CV)", lambda: fusion_pondere_loocv(branches_v3, y)),
    ("3. Stacking logistique (LOO-CV)", lambda: fusion_stacking_loocv(branches_v3, y)),
]:
    proba = fn()
    res = summarize(y, proba)
    res["AUC_CI95"] = bootstrap_ci(y, proba, seed=SEED)
    all_fusion_results[nom] = res

# ============================================================
# 4. Fusion precoce (concatenation + PCA + un seul classifieur)
# ============================================================
def run_loocv_early_fusion(text_embeddings, image_embeddings_per_patient, y,
                            n_components_text=10, n_components_image=10, seed=42):
    n = len(y)
    loo = LeaveOneOut()
    oof_proba = np.zeros(n)
    image_patient_level = np.stack([
        image_embeddings_per_patient[int(pid)].mean(axis=0) for pid in patient_ids
    ])
    for train_idx, test_idx in loo.split(text_embeddings):
        pca_text = PCA(n_components=n_components_text, random_state=seed)
        text_train = pca_text.fit_transform(text_embeddings[train_idx])
        text_test = pca_text.transform(text_embeddings[test_idx])
        pca_img = PCA(n_components=n_components_image, random_state=seed)
        img_train = pca_img.fit_transform(image_patient_level[train_idx])
        img_test = pca_img.transform(image_patient_level[test_idx])
        X_train = np.hstack([text_train, img_train])
        X_test = np.hstack([text_test, img_test])
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        clf = LogisticRegression(class_weight="balanced", max_iter=2000, C=0.5, random_state=seed)
        clf.fit(X_train, y[train_idx])
        oof_proba[test_idx] = clf.predict_proba(X_test)[:, 1]
    return oof_proba

proba_early = run_loocv_early_fusion(camembert_embeddings, embeds_foundation, y, seed=SEED)
res_early = summarize(y, proba_early)
res_early["AUC_CI95"] = bootstrap_ci(y, proba_early, seed=SEED)
all_fusion_results["4. Fusion precoce (concat+PCA)"] = res_early

# ============================================================
# 5. Fusion guidee (le texte module la confiance accordee a l'image)
# ============================================================
def run_guided_fusion(proba_text, proba_image, y):
    text_confidence = 1 / (1 + np.exp(-zscore(proba_text)))
    agreement = 1 - np.abs(text_confidence - proba_image)
    w_text = agreement / (agreement + (1 - agreement) + 1e-9)
    return w_text * proba_text + (1 - w_text) * proba_image

proba_guided = run_guided_fusion(proba_camembert, proba_foundation, y)
res_guided = summarize(y, proba_guided)
res_guided["AUC_CI95"] = bootstrap_ci(y, proba_guided, seed=SEED)
all_fusion_results["5. Fusion guidee (texte module image)"] = res_guided

# ============================================================
# TABLEAU FINAL
# ============================================================
print("=" * 90)
print("TABLEAU COMPLET DES 5 STRATEGIES DE FUSION (branches : CamemBERT + BiomedCLIP)")
print("=" * 90)
fusion_df = pd.DataFrame(all_fusion_results).T
print(fusion_df.to_string())

best = fusion_df["AUC"].idxmax()
print(f"\nMeilleure strategie : {best} (AUC={fusion_df.loc[best,'AUC']:.4f})")
print(f"Reference — Texte seul : {summarize(y, proba_camembert)['AUC']:.4f}")
print(f"Reference — Image seule : {summarize(y, proba_foundation)['AUC']:.4f}")

fusion_df.to_csv("/kaggle/working/fusion_5strategies_complet.csv")
print("\nSauvegarde -> fusion_5strategies_complet.csv")
