"""
plot_confusion_matrix.py
Valuta il modello fine-tuned su GTZAN con Test-Time Averaging e
salva la Confusion Matrix (normalizzata e conteggio grezzo) in formato immagine.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from src import (
    CLMRModel,
    AudioLinearClassifier,
    get_downstream_dataloaders,
    get_device,
    GTZAN_GENRES,
)


def main():
    device = get_device()
    checkpoint_path = "checkpoints/finetuned_model_best.pt"
    data_dir = "data/genres_original"
    output_dir = "reports"
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print(" GENERAZIONE CONFUSION MATRIX: MODELLO FINE-TUNED (GTZAN)")
    print(f" Dispositivo: {device}")
    print(f" Checkpoint:  {checkpoint_path}")
    print("=" * 60 + "\n")

    # 1. Carica il test set con Test-Time Averaging (10 segmenti)
    _, test_loader = get_downstream_dataloaders(
        data_dir=data_dir,
        batch_size=32,
        num_workers=0,
        seed=42,
        num_eval_segments=10,
    )

    # 2. Inizializza i modelli e carica il checkpoint unificato
    model = CLMRModel(pretrained_backbone=False).to(device)
    classifier = AudioLinearClassifier(in_features=512, num_classes=len(GTZAN_GENRES)).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["backbone_state_dict"])
    classifier.load_state_dict(ckpt["classifier_state_dict"])

    model.eval()
    classifier.eval()

    all_targets = []
    all_preds = []

    # 3. Inferenza aggregata sul test set
    print("Calcolo delle predizioni con Test-Time Averaging...")
    with torch.no_grad():
        for multi_specs, targets in tqdm(test_loader, desc="Valutazione"):
            B, S, C, H, W = multi_specs.shape
            specs_flat = multi_specs.view(B * S, C, H, W).to(device)

            feats = model.encode(specs_flat)
            feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
            logits = classifier(feats)

            # Media dei logits sui segmenti del singolo brano
            logits_reshaped = logits.view(B, S, -1)
            mean_logits = logits_reshaped.mean(dim=1)
            preds = torch.argmax(mean_logits, dim=-1)

            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    # 4. Report di classificazione a terminale
    print("\n" + "=" * 60)
    print(" CLASSIFICATION REPORT DETTAGLIATO")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=GTZAN_GENRES, digits=3, zero_division=0))

    # 5. Calcolo matrici di confusione
    cm_counts = confusion_matrix(y_true, y_pred)
    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")

    # 6. Plotting affiancato
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Subplot 1: Valori assoluti
    disp_counts = ConfusionMatrixDisplay(confusion_matrix=cm_counts, display_labels=GTZAN_GENRES)
    disp_counts.plot(ax=axes[0], cmap="Blues", values_format="d", colorbar=False)
    axes[0].set_title("Matrice di Confusione (Conteggi)", fontsize=13, pad=10)
    axes[0].set_xlabel("Genere Predetto", fontsize=11)
    axes[0].set_ylabel("Genere Reale", fontsize=11)
    axes[0].tick_params(axis="x", rotation=45)

    # Subplot 2: Normalizzata per riga (Recall percentuale per genere)
    disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=GTZAN_GENRES)
    disp_norm.plot(ax=axes[1], cmap="Blues", values_format=".2f", colorbar=False)
    axes[1].set_title("Matrice di Confusione Normalizzata (Recall)", fontsize=13, pad=10)
    axes[1].set_xlabel("Genere Predetto", fontsize=11)
    axes[1].set_ylabel("Genere Reale", fontsize=11)
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "confusion_matrix_finetuned.png")
    plt.savefig(output_path, dpi=300)
    print(f"\n[Salvataggio] Grafico salvato in: {output_path}\n")


if __name__ == "__main__":
    main()