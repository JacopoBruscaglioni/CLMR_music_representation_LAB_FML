"""
train_finetune.py
Fine-tuning supervisionato end-to-end del modello CLMR (Backbone + Testa Lineare) su GTZAN.
Salva il miglior modello in un checkpoint unificato compatibile con batch_test_to_csv.py.
"""

import argparse
import os
import sys
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
    parser = argparse.ArgumentParser(description="Fine-Tuning End-to-End su GTZAN")
    parser.add_argument("--data_dir", type=str, default="data/genres_original")
    parser.add_argument("--backbone_ckpt", type=str, default="checkpoints/clmr_best_backbone.pt")
    parser.add_argument("--probe_ckpt", type=str, default="checkpoints/linear_probe_best.pt")
    parser.add_argument("--save_path", type=str, default="checkpoints/finetuned_model_best.pt")
    parser.add_argument("--epochs", type=int, default=30, help="Epoche di fine-tuning")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr_backbone", type=float, default=5e-5, help="Learning rate per la backbone (conservativo)")
    parser.add_argument("--lr_classifier", type=float, default=1e-3, help="Learning rate per la testa lineare")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_segments", type=int, default=5, help="Finestre aggregate per brano nel test")
    parser.add_argument("--num_workers", type=int, default=0, help="0 consigliato su Windows")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def train_one_epoch(model, classifier, train_loader, criterion, optimizer, device, epoch):
    model.train()
    classifier.train()
    loss_meter = AverageMeter("Loss")
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch [{epoch:02d}] Train", leave=False)
    for specs, targets in pbar:
        specs = specs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        
        # Forward pass end-to-end con gradiente sbloccato sulla backbone
        feats = model.encode(specs)
        feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
        logits = classifier(feats)

        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), n=specs.size(0))
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

        pbar.set_postfix({"Loss": f"{loss_meter.avg:.4f}", "Acc": f"{correct / total * 100:.1f}%"})

    train_acc = (correct / total) * 100.0
    return loss_meter.avg, train_acc


def evaluate_aggregated(model, classifier, test_loader, device):
    model.eval()
    classifier.eval()

    all_targets = []
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for multi_specs, targets in tqdm(test_loader, desc="Valutazione Test-Time", leave=False):
            B, S, C, H, W = multi_specs.shape
            targets = targets.to(device)

            specs_flat = multi_specs.view(B * S, C, H, W).to(device)
            feats = model.encode(specs_flat)
            feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
            logits = classifier(feats)

            # Test-Time Averaging: media dei logits sulle 10 finestre dello stesso brano
            logits_reshaped = logits.view(B, S, -1)
            mean_logits = logits_reshaped.mean(dim=1)

            probs = torch.softmax(mean_logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)

    top1 = (all_preds == all_targets).mean() * 100.0
    top3_preds = np.argsort(all_probs, axis=1)[:, -3:]
    top3 = np.mean([t in top3_preds[i] for i, t in enumerate(all_targets)]) * 100.0

    try:
        auc = roc_auc_score(all_targets, all_probs, multi_class="ovr", average="macro") * 100.0
    except Exception:
        auc = float("nan")

    report = classification_report(
        all_targets,
        all_preds,
        target_names=GTZAN_GENRES,
        digits=3,
        zero_division=0,
    )

    return top1, top3, auc, report


def main():
    args = parse_args()
    device = get_device()
    torch.manual_seed(args.seed)

    print("\n" + "=" * 65)
    print(" FINE-TUNING SUPERVISIONATO END-TO-END (GTZAN)")
    print(f" Dispositivo:          {device}")
    print(f" Backbone Iniziale:    {args.backbone_ckpt}")
    print(f" Epoche:               {args.epochs}")
    print(f" Batch Size:           {args.batch_size}")
    print(f" LR Backbone:          {args.lr_backbone}")
    print(f" LR Classifier:        {args.lr_classifier}")
    print(f" Finestre Aggregazione:{args.num_segments}")
    print("=" * 65 + "\n")

    # 1. Caricamento Dataset GTZAN
    train_loader, test_loader = get_downstream_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        num_eval_segments=args.num_segments,
    )

    # 2. Inizializzazione Modelli
    model = CLMRModel(pretrained_backbone=False)
    ckpt = load_checkpoint(args.backbone_ckpt, device=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    # Assicura che tutti i parametri della backbone siano sbloccati per il fine-tuning
    for param in model.parameters():
        param.requires_grad = True

    classifier = AudioLinearClassifier(in_features=512, num_classes=len(GTZAN_GENRES)).to(device)
    if os.path.exists(args.probe_ckpt):
        pr_ckpt = torch.load(args.probe_ckpt, map_location=device, weights_only=False)
        state_dict = pr_ckpt.get("model_state_dict", pr_ckpt)
        classifier.load_state_dict(state_dict)
        print(f"[Init] Pesi del classificatore caricati con successo da: {args.probe_ckpt}")
    else:
        print("[Init] Nessun probe trovato, inizializzazione casuale della testa.")
    # 3. Ottimizzatore a parametri differenziati
    optimizer = optim.AdamW([
        {"params": model.parameters(), "lr": args.lr_backbone},
        {"params": classifier.parameters(), "lr": args.lr_classifier}
    ], weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    best_val_top1 = 0.0
    best_report = ""

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    # 4. Training Loop End-to-End
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, classifier, train_loader, criterion, optimizer, device, epoch
        )
        scheduler.step()

        # Valutazione ad ogni epoca sul test set aggregato
        val_top1, val_top3, val_auc, report = evaluate_aggregated(model, classifier, test_loader, device)

        print(f"Epoca [{epoch:02d}/{args.epochs:02d}] "
              f"| Train Loss: {train_loss:.4f} | Train Acc: {train_acc:5.1f}% "
              f"| Test Top-1: {val_top1:5.2f}% | Test Top-3: {val_top3:5.2f}% | AUC: {val_auc:5.2f}%")

        # Salva il checkpoint congiunto migliore (formato letto nativamente da batch_test_to_csv.py)
        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            best_report = report
            joint_checkpoint = {
                "epoch": epoch,
                "backbone_state_dict": model.state_dict(),
                "classifier_state_dict": classifier.state_dict(),
                "val_acc": val_top1,
                "val_auc": val_auc,
                "genres": GTZAN_GENRES,
            }
            torch.save(joint_checkpoint, args.save_path)
            print(f"  --> [MIGLIOR MODELLO SALVATO] Top-1: {val_top1:.2f}% in {args.save_path}")

    print("\n" + "=" * 65)
    print(" FINE-TUNING COMPLETATO")
    print(f" Miglior Top-1 Accuracy: {best_val_top1:.2f}%")
    print(f" Checkpoint esportato:   {args.save_path}")
    print("=" * 65)
    print("\nReport Dettagliato del Miglior Modello:")
    print(best_report)


if __name__ == "__main__":
    main()