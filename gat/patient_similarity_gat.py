# =============================================================================
# Q3 — GNN / GAT DE SIMILARITÉ INTER-PATIENTS (ANALYSE EXPLORATOIRE)
#
# Dans les notes exploratoires du notebook, "GAT" pouvait designer une
# aggregation textuelle sans implementation de graphe. Ce module implemente
# un vrai Graph Attention Network, avec :
#   - un vrai graphe de similarité k-NN (cosinus) entre patients
#   - une vraie couche d'attention (coefficients softmax sur les voisins,
#     comme dans les équations de l'article de reference)
#   - un entraînement réel avec autograd (PyTorch), pas une boucle vide
#
# CADRAGE MÉTHODOLOGIQUE (à respecter dans l'article) :
#   - Le graphe est construit de façon TRANSDUCTIVE : au moment de calculer
#     les similarités, les features de TOUS les patients sont visibles
#     (y compris le patient testé). C'est une limite documentée du protocole.
#   - En revanche, le LABEL du patient testé n'est JAMAIS utilisé pendant
#     l'entraînement (masquage explicite à chaque fold de la LOO-CV).
#   - Ce module est un COMPLÉMENT exploratoire, pas le modèle principal de
#     l'article (qui reste la fusion tardive image+texte, Q2).
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# =============================================================================
# 1. Construction du graphe patient (k-NN cosinus)
# =============================================================================
def build_patient_graph(features: np.ndarray, k: int = 5):
    """
    features : matrice (n_patients, n_features) — DÉJÀ standardisée en amont
    (fit sur l'ensemble des patients : voir le cadrage transductif ci-dessus).

    Retourne : adjacency (n,n) binaire symétrique (kNN mutuel ou simple),
    utilisée ensuite comme masque d'attention (un patient n'attend que ses
    voisins, pas tout le monde).
    """
    n = features.shape[0]
    sim = cosine_similarity(features)
    np.fill_diagonal(sim, -np.inf)  # un patient n'est pas son propre voisin

    adjacency = np.zeros((n, n), dtype=bool)
    for i in range(n):
        neighbors = np.argsort(-sim[i])[:k]
        adjacency[i, neighbors] = True
    # Symétrisation : si i voit j comme voisin OU j voit i, l'arête existe
    adjacency = adjacency | adjacency.T
    np.fill_diagonal(adjacency, True)  # self-loop, standard en GAT
    return adjacency


# =============================================================================
# 2. Couche GAT réelle (une tête, un layer — suffisant vu la taille du graphe)
# =============================================================================
class SimpleGATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2 * out_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        h = self.W(x)  # (n, out_dim)

        h_i = h.unsqueeze(1).expand(n, n, -1)
        h_j = h.unsqueeze(0).expand(n, n, -1)
        e = self.leaky_relu(self.a(torch.cat([h_i, h_j], dim=-1)).squeeze(-1))  # (n, n)

        # Masquage : un patient n'attend que ses voisins (adjacency), le reste à -inf
        e = e.masked_fill(~adjacency, float("-inf"))
        alpha = F.softmax(e, dim=1)  # coefficients d'attention (Eq. article de reference)

        h_out = torch.matmul(alpha, h)
        return h_out, alpha


class PatientGAT(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 16):
        super().__init__()
        self.gat1 = SimpleGATLayer(in_dim, hidden_dim)
        self.gat2 = SimpleGATLayer(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, x, adjacency):
        h, alpha1 = self.gat1(x, adjacency)
        h = F.elu(h)
        h, alpha2 = self.gat2(h, adjacency)
        h = F.elu(h)
        logits = self.classifier(h)
        return logits, alpha2


# =============================================================================
# 3. Entraînement LOO-CV avec masquage strict du label testé
# =============================================================================
def run_loocv_gat(features: np.ndarray, y: np.ndarray, k: int = 5,
                   hidden_dim: int = 16, epochs: int = 200, lr: float = 0.01,
                   seed: int = SEED):
    """
    Graphe construit UNE FOIS sur l'ensemble transductif (cf. cadrage plus haut).
    À chaque fold LOO :
      - le label du patient testé est masqué (jamais vu dans la loss)
      - le modèle est réentraîné from scratch (poids réinitialisés)
      - seule la prédiction sur le nœud testé est retenue
    """
    n = len(y)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)  # transductif : fit sur tous

    adjacency_np = build_patient_graph(features_scaled, k=k)
    adjacency = torch.tensor(adjacency_np, dtype=torch.bool)
    x = torch.tensor(features_scaled, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)

    oof_proba = np.zeros(n)

    for test_idx in range(n):
        torch.manual_seed(seed)
        model = PatientGAT(in_dim=features.shape[1], hidden_dim=hidden_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

        train_mask = torch.ones(n, dtype=torch.bool)
        train_mask[test_idx] = False  # masquage strict du label testé

        # Pondération de classe (cohorte déséquilibrée 28/17)
        class_counts = torch.bincount(y_t[train_mask])
        class_weights = 1.0 / class_counts.float()
        class_weights = class_weights / class_weights.sum() * 2

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            logits, _ = model(x, adjacency)
            loss = F.cross_entropy(logits[train_mask], y_t[train_mask], weight=class_weights)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            logits, _ = model(x, adjacency)
            proba = F.softmax(logits, dim=1)[:, 1]
            oof_proba[test_idx] = proba[test_idx].item()

    return oof_proba


def summarize(y: np.ndarray, proba: np.ndarray, thr: float = 0.5) -> dict:
    pred = (proba >= thr).astype(int)
    return {
        "AUC": roc_auc_score(y, proba),
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0, average="macro"),
    }


def bootstrap_ci(y, proba, n_boot=2000, seed=SEED):
    rng = np.random.RandomState(seed)
    n = len(y)
    aucs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], proba[idx]))
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


# =============================================================================
# 4. MAIN — combine texte + image dès que les deux sont disponibles
# =============================================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    import pbt_cm_clean_pipeline as m
    import pandas as pd

    df = pd.read_excel("classeur/ClasseurPFE1 (1).xlsx", sheet_name="Feuil1")
    for col in ["Origine_Tumeur", "Type_Général", "Sexe", "Localisation", "Métastase_Origine",
                "ANNOTATION", "Rapport"]:
        df[col] = df[col].astype(str).fillna("")
        df[col] = df[col].replace("nan", "")

    y = (df["Origine_Tumeur"].str.strip() == "Secondaire").astype(int).values
    safe_feat = m.build_safe_manual_features(df)

    # --- Graphe texte seul (déjà fait) ---
    print("=" * 70)
    print("Q3a — GAT, features texte 'safe' uniquement")
    print("=" * 70)
    proba_gat_text = run_loocv_gat(safe_feat.values.astype(float), y, k=5)
    print(summarize(y, proba_gat_text))

    # --- Graphe texte + image, dès que les probas/embeddings image sont là ---
    # Remplacer ce bloc par le chargement réel du CSV OOF image une fois reçu :
    #   img_oof = pd.read_csv("oof_image_effnet_balanced_seed42.csv")
    #   img_feat = img_oof.set_index("ID_Patient").loc[df["ID_Patient"], "proba_image_effnet_balanced"].values.reshape(-1,1)
    #   combined_feat = np.hstack([safe_feat.values.astype(float), img_feat])
    #   proba_gat_combined = run_loocv_gat(combined_feat, y, k=5)
    #   print("Q3b — GAT, texte + image combinés :", summarize(y, proba_gat_combined))
    print()
    print("Q3b (texte+image) : en attente du CSV OOF image pour être lancé.")
