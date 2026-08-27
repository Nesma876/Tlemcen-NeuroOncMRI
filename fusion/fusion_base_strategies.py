# =============================================================================
# FUSION TARDIVE — 3 VARIANTES, DE LA PLUS SIMPLE À LA PLUS SOPHISTIQUÉE
#
# Corrige le défaut de la "Super Fusion" originale : là où le notebook de
# départ optimisait les poids et le seuil de décision directement sur
# l'ensemble des 45 patients (biais optimiste, cf. audit initial), toute
# pondération apprise ici est apprise EN LOO-CV, donc sans jamais voir le
# label du patient testé au moment de fixer les poids.
#
# Entrée attendue : un dict {nom_branche: proba_oof (array de taille 45)},
# chaque proba_oof étant déjà elle-même un résultat out-of-fold LOO-CV
# (texte, image, GAT...). Fonctionne avec 2 branches ou plus — pas besoin
# d'attendre que toutes les modalités soient prêtes.
# =============================================================================

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score


def summarize(y, proba, thr=0.5):
    pred = (proba >= thr).astype(int)
    return {
        "AUC": roc_auc_score(y, proba),
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0, average="macro"),
    }


def bootstrap_ci(y, proba, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y)
    aucs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], proba[idx]))
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


# -----------------------------------------------------------------------
# Variante 1 — Moyenne simple (aucun paramètre appris)
# -----------------------------------------------------------------------
def fusion_moyenne_simple(branch_probas: dict) -> np.ndarray:
    stacked = np.stack(list(branch_probas.values()), axis=1)
    return stacked.mean(axis=1)


# -----------------------------------------------------------------------
# Variante 2 — Moyenne pondérée, poids optimisés en LOO-CV (pas sur tout le set)
# -----------------------------------------------------------------------
def fusion_pondere_loocv(branch_probas: dict, y: np.ndarray) -> np.ndarray:
    """
    À chaque fold, on cherche les poids qui maximisent l'AUC sur les 44
    autres patients (grille simple sur le simplexe), puis on les applique
    au patient testé. Le patient testé n'intervient JAMAIS dans la
    recherche de poids de son propre fold.
    """
    names = list(branch_probas.keys())
    n_branches = len(names)
    stacked = np.stack([branch_probas[k] for k in names], axis=1)  # (n, n_branches)
    n = len(y)
    oof = np.zeros(n)

    # Grille de poids sur le simplexe (pas fin, cohérent avec la taille de cohorte)
    if n_branches == 2:
        grid = [(w, 1 - w) for w in np.linspace(0, 1, 21)]
    else:
        # Grille aléatoire sur le simplexe pour >2 branches (reproductible)
        rng = np.random.RandomState(42)
        raw = rng.dirichlet(np.ones(n_branches), size=200)
        grid = [tuple(r) for r in raw]

    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(stacked):
        best_w, best_auc = None, -1
        for w in grid:
            w = np.array(w)
            combo_train = stacked[train_idx] @ w
            if len(np.unique(y[train_idx])) < 2:
                continue
            auc = roc_auc_score(y[train_idx], combo_train)
            if auc > best_auc:
                best_auc, best_w = auc, w
        oof[test_idx] = stacked[test_idx] @ best_w

    return oof


# -----------------------------------------------------------------------
# Variante 3 — Stacking (méta-régression logistique, appris en LOO-CV)
# -----------------------------------------------------------------------
def fusion_stacking_loocv(branch_probas: dict, y: np.ndarray) -> np.ndarray:
    names = list(branch_probas.keys())
    stacked = np.stack([branch_probas[k] for k in names], axis=1)
    n = len(y)
    oof = np.zeros(n)

    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(stacked):
        meta = LogisticRegression(class_weight="balanced", max_iter=1000)
        meta.fit(stacked[train_idx], y[train_idx])
        oof[test_idx] = meta.predict_proba(stacked[test_idx])[:, 1]

    return oof


# =============================================================================
# MAIN — combine texte + GAT + image dès que les CSV OOF image sont reçus
# =============================================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    import pandas as pd
    import pbt_cm_clean_pipeline as m
    import gnn_patient_similarity as g

    df = pd.read_excel("classeur/ClasseurPFE1 (1).xlsx", sheet_name="Feuil1")
    for col in ["Origine_Tumeur", "Type_Général", "Sexe", "Localisation", "Métastase_Origine",
                "ANNOTATION", "Rapport"]:
        df[col] = df[col].astype(str).fillna("").replace("nan", "")

    y = (df["Origine_Tumeur"].str.strip() == "Secondaire").astype(int).values
    safe_feat = m.build_safe_manual_features(df)
    text_raw = m.mask_class_revealing_terms(df["ANNOTATION"] + " " + df["Rapport"])

    print("Calcul des probas OOF par branche (texte, GAT)...")
    proba_text = m.run_loocv_text_model(safe_feat, text_raw, y)
    proba_gat = g.run_loocv_gat(safe_feat.values.astype(float), y, k=5)

    branches = {"texte": proba_text, "gat": proba_gat}

    # --- Ajouter l'image dès que le CSV OOF est disponible ---
    import glob
    image_oof_candidates = (
        glob.glob("oof_image_*seed*.csv") +
        glob.glob("oof_probabilities_seed*.csv")
    )
    if image_oof_candidates:
        img_df = pd.read_csv(image_oof_candidates[0])
        proba_col = [c for c in img_df.columns if c.startswith("proba_image")][0]
        img_map = img_df.set_index("ID_Patient")[proba_col]
        branches["image"] = df["ID_Patient"].map(img_map).values
        print(f"Branche image ajoutée depuis {image_oof_candidates[0]} (colonne {proba_col})")
    else:
        print("Aucun CSV OOF image trouvé -> fusion texte+GAT uniquement pour l'instant.")

    print()
    for nom_fusion, fn in [
        ("Moyenne simple", lambda: fusion_moyenne_simple(branches)),
        ("Moyenne ponderee (LOO-CV)", lambda: fusion_pondere_loocv(branches, y)),
        ("Stacking logistique (LOO-CV)", lambda: fusion_stacking_loocv(branches, y)),
    ]:
        proba = fn()
        res = summarize(y, proba)
        ci = bootstrap_ci(y, proba)
        print(f"--- {nom_fusion} ---")
        print(res, "IC95% AUC:", ci)
        print()

    print("Pour reference individuelle :")
    for nom, p in branches.items():
        print(f"{nom} seul :", summarize(y, p))
