"""
train_linear_probe.py

Valutazione a valle (Linear Probing) del modello CLMR pre-addestrato:
1. Carica l'encoder ResNet-18 dal checkpoint salvato.
2. Congela TUTTI i pesi dell'encoder (requires_grad = False).
3. Addestra uno strato lineare (512 -> 10 generi) con Cross-Entropy.
4. Valuta a test-time con aggregazione multi-segmento (media dei logits sui frammenti).
5. Calcola Top-1 Acc, Top-3 Acc, ROC-AUC e Classification Report.
"""

import argparse
import sys
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score

from src import (
    CLMRModel,
    AudioLinearClassifier,
    get_downstream_dataloaders,
    get_device,
    load_checkpoint,
    AverageMeter,
    GTZAN_GENRES,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Linear Probing su GTZAN con Backbone CLMR congelato")
    parser.add_argument("--data_dir", type=str, default="data/genres_original")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/clmr_best_backbone.pt")
    parser.add_argument("--epochs", type=int, default=30, help="Epoche di addestramento della testa lineare")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate per il classificatore")
    parser.add_argument("--num_segments", type=int, default=10, help="Numero di finestre da aggregare per brano nel test")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def train_classifier(encoder, classifier, dataloader, criterion, optimizer, device, epoch):
    encoder.eval()       # L'encoder rimane sempre in eval (pesi e BatchNorm congelati)
    classifier.train()   # Solo il linear probe impara
    loss_meter = AverageMeter("Loss")

    pbar = tqdm(dataloader, desc=f"Linear Probe [Epoca {epoch}]", leave=False)
    for specs, targets in pbar:
        specs = specs.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            features = encoder.encode(specs)  # [B, 512]

        logits = classifier(features)        # [B, 10]
        loss = criterion(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), n=specs.size(0))
        pbar.set_postfix({"Loss": f"{loss_meter.avg:.4f}"})

    return loss_meter.avg


def evaluate_aggregated(encoder, classifier, dataloader, device):
    """
    Esegue la valutazione aggregando le predizioni su segmenti multipli per ogni brano.
    """
    encoder.eval()
    classifier.eval()

    all_targets = []
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for multi_specs, targets in tqdm(dataloader, desc="Valutazione Multi-Segmento", leave=False):
            # multi_specs shape: [Batch, Segments, 1, 64, 130]
            B, S, C, H, W = multi_specs.shape
            targets = targets.to(device)

            # Riorganizza per passare tutti i segmenti in un unico forward pass
            specs_flattened = multi_specs.view(B * S, C, H, W).to(device)
            features = encoder.encode(specs_flattened)           # [B * S, 512]
            logits = classifier(features)                         # [B * S, 10]

            # Riconfigura a [B, S, 10] e calcola la media dei logits tra i segmenti dello stesso brano
            logits_reshaped = logits.view(B, S, -1)
            mean_logits = logits_reshaped.mean(dim=1)             # [B, 10]

            probs = torch.softmax(mean_logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)

    # 1. Top-1 Accuracy
    top1_acc = (all_preds == all_targets).mean() * 100.0

    # 2. Top-3 Accuracy
    top3_preds = np.argsort(all_probs, axis=1)[:, -3:]
    top3_acc = np.mean([t in top3_preds[i] for i, t in enumerate(all_targets)]) * 100.0

    # 3. ROC-AUC One-vs-Rest (Macro)
    try:
        roc_auc = roc_auc_score(all_targets, all_probs, multi_class="ovr", average="macro") * 100.0
    except Exception:
        roc_auc = float("nan")

    # 4. Report per classe
    report = classification_report(
        all_targets,
        all_preds,
        target_names=GTZAN_GENRES,
        digits=3,
        zero_division=0,
    )

    return top1_acc, top3_acc, roc_auc, report


def main():
    args = parse_args()
    device = get_device()

    print("\n" + "=" * 60)
    print(" LINEAR PROBING: VALUTAZIONE RAPPRESENTAZIONI CLMR")
    print(f" Checkpoint:       {args.checkpoint_path}")
    print(f" Epoche Probe:     {args.epochs}")
    print(f" Batch Size:       {args.batch_size}")
    print(f" Finestre/Brano:   {args.num_segments} (Test-Time Aggregation)")
    print("=" * 60 + "\n")

    # 1. Carica DataLoader
    try:
        train_loader, test_loader = get_downstream_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            num_eval_segments=args.num_segments,
        )
    except FileNotFoundError as e:
        print(f"[ERRORE] {e}")
        sys.exit(1)

    # 2. Istanzia Encoder e carica i pesi del checkpoint
    encoder = CLMRModel(pretrained_backbone=False)
    checkpoint = load_checkpoint(args.checkpoint_path, device=device)
    encoder.load_state_dict(checkpoint["model_state_dict"])
    encoder = encoder.to(device)

    # 3. Congelamento TOTALE dei pesi dell'encoder
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()

    # 4. Istanzia la sola testa di classificazione lineare (512 -> 10)
    classifier = AudioLinearClassifier(in_features=512, num_classes=len(GTZAN_GENRES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=args.lr)

    # 5. Training loop del linear classifier
    print("Inizio addestramento della testa lineare (encoder congelato)...")
    for epoch in range(1, args.epochs + 1):
        loss = train_classifier(encoder, classifier, train_loader, criterion, optimizer, device, epoch)
        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"Epoca [{epoch:02d}/{args.epochs:02d}] - Linear Probe Loss: {loss:.4f}")

    # SALVATAGGIO DEI PESI (subito dopo il ciclo for)
    os.makedirs("checkpoints", exist_ok=True)
    probe_save_path = "checkpoints/linear_probe_best.pt"
    torch.save(classifier.state_dict(), probe_save_path)
    print(f"\n[Checkpoint] Pesi del Linear Probe salvati in: {probe_save_path}\n")
    
    # 6. Valutazione finale con aggregazione multi-segmento
    print("\nCalcolo metriche con Test-Time Averaging sul dataset di test...")
    top1, top3, roc_auc, report = evaluate_aggregated(encoder, classifier, test_loader, device)

    print("\n" + "=" * 60)
    print(" RISULTATI FINALI LINEAR PROBE (GTZAN)")
    print("=" * 60)
    print(f" Top-1 Accuracy: {top1:.2f}%")
    print(f" Top-3 Accuracy: {top3:.2f}%")
    if not np.isnan(roc_auc):
        print(f" ROC-AUC (Macro OVR): {roc_auc:.2f}%")
    print("\nReport Dettagliato per Genere Musicale:")
    print(report)
    print("=" * 60)


if __name__ == "__main__":
    main()