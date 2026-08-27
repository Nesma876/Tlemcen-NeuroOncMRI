# =============================================================================
# CORRECTION D'ANONYMISATION — MASQUAGE DES SURIMPRESSIONS TEXTE
#
# Principe : dans une image de scanner IRM exportée en capture d'écran, le
# cerveau forme TOUJOURS le plus grand composant connexe de pixels non-noirs.
# Le texte incrusté (nom, date, ID, mesures) forme des composants BEAUCOUP
# plus petits, séparés spatialement du bloc cérébral.
#
# Stratégie : pour chaque image,
#   1. Binariser (seuil bas, pour capturer même les zones sombres du cerveau)
#   2. Identifier tous les composants connexes
#   3. Ne conserver QUE le plus grand composant (+ une marge de dilatation
#      pour ne pas ronger les bords du cerveau)
#   4. Mettre à zéro (noir) tout le reste -> supprime tout texte, peu importe
#      sa position (haut, bas, coin), sans dépendre d'une zone fixe.
#
# Une vérification de sécurité est appliquée : si le plus grand composant
# occupe une part anormalement faible de l'image (signe que la détection a
# échoué), l'image est laissée INCHANGÉE et signalée pour revue manuelle
# plutôt que risquée d'effacer le cerveau par erreur.
# =============================================================================

import numpy as np
from PIL import Image
from scipy import ndimage
from pathlib import Path
import glob
import csv

THRESHOLD = 12          # seuil bas : capture le cerveau même dans ses zones sombres
DILATION_ITER = 3       # marge de sécurité autour du composant conservé
MIN_KEEP_RATIO = 0.03   # si le plus grand composant est plus petit que ça, on n'y touche pas (sécurité)
MAX_KEEP_RATIO = 0.65   # si le plus grand composant est plus grand que ça, l'image est probablement
                        # trop bruitée pour isoler le texte du bruit de fond (cerveau + texte fusionnés
                        # en un seul bloc) -> on ne modifie rien, à traiter manuellement


def clean_single_image(path: Path):
    im = Image.open(path).convert("L")
    arr = np.array(im)

    mask = arr > THRESHOLD
    labeled, n_components = ndimage.label(mask)

    if n_components == 0:
        return arr, "no_foreground", 0.0

    sizes = ndimage.sum(mask, labeled, range(1, n_components + 1))
    largest_label = np.argmax(sizes) + 1
    largest_ratio = sizes[largest_label - 1] / arr.size

    if largest_ratio < MIN_KEEP_RATIO:
        # Détection non fiable sur cette image -> on ne modifie rien,
        # à vérifier manuellement.
        return arr, "SKIPPED_low_confidence", largest_ratio

    if largest_ratio > MAX_KEEP_RATIO:
        # Image trop bruitée : le texte est probablement fusionné avec le
        # bruit de fond dans le même composant connexe -> masquage non fiable.
        return arr, "SKIPPED_too_noisy", largest_ratio

    keep_mask = labeled == largest_label
    keep_mask = ndimage.binary_dilation(keep_mask, iterations=DILATION_ITER)

    cleaned = arr.copy()
    cleaned[~keep_mask] = 0
    return cleaned, "cleaned", largest_ratio


def process_dataset(input_root: Path, output_root: Path, log_csv: Path):
    files = sorted(glob.glob(str(input_root / "**" / "*.png"), recursive=True))
    print(f"Images à traiter : {len(files)}")

    log_rows = []
    for i, f in enumerate(files):
        f = Path(f)
        rel = f.relative_to(input_root)
        out_path = output_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        cleaned, status, ratio = clean_single_image(f)
        Image.fromarray(cleaned).save(out_path)

        log_rows.append({"file": str(rel), "status": status, "largest_component_ratio": round(float(ratio), 4)})

        if i % 3000 == 0:
            print(f"  ... {i}/{len(files)}")

    with open(log_csv, "w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=["file", "status", "largest_component_ratio"])
        writer.writeheader()
        writer.writerows(log_rows)

    n_cleaned = sum(1 for r in log_rows if r["status"] == "cleaned")
    n_skipped_low = sum(1 for r in log_rows if r["status"] == "SKIPPED_low_confidence")
    n_skipped_noisy = sum(1 for r in log_rows if r["status"] == "SKIPPED_too_noisy")
    print(f"\nNettoyées avec succès : {n_cleaned}")
    print(f"Ignorées - confiance insuffisante (À VÉRIFIER) : {n_skipped_low}")
    print(f"Ignorées - image trop bruitée (À VÉRIFIER MANUELLEMENT / méthode alternative) : {n_skipped_noisy}")
    print(f"Log complet -> {log_csv}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Mask likely text overlays in MRI screenshots without modifying source images."
    )
    parser.add_argument("--input-root", type=Path, required=True, help="Directory containing PNG images to clean.")
    parser.add_argument("--output-root", type=Path, required=True, help="Directory for cleaned PNG copies.")
    parser.add_argument("--log-csv", type=Path, required=True, help="CSV log path.")
    args = parser.parse_args()

    process_dataset(
        input_root=args.input_root,
        output_root=args.output_root,
        log_csv=args.log_csv,
    )
