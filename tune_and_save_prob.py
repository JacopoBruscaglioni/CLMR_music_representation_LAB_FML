"""
tune_and_save_prob.py
Grid Search su feature CLMR pre-estratte per identificare la migliore configurazione 
di ottimizzazione e salvare il checkpoint ottimale del Linear Probe.
Compatibile con Windows (multiprocessing spawn-safe).
"""

import os
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report, roc_auc_score

from src import (
    CLMRModel,
    AudioLinearClassifier,
    get_downstream_dataloaders,
    get_device,
    load_checkpoint,
    GTZAN_GENRES,
)


def evaluate_model(classifier, X_test_grouped, y_test, device):
    classifier.eval()
    with torch.no_grad():
        feats_dev = X_test_grouped.to(device)
        N, S, D = feats_dev.shape
        logits = classifier(feats_dev.view(N * S, D)).view(N, S, -1)
        mean_logits = logits.mean(dim=1)  # Media logits sui segmenti del brano
        probs = torch.softmax(mean_logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=-1)

    top1 = (preds == y_test).mean() * 100.0
    top3_preds = np.argsort(probs, axis=1)[:, -3:]
    top3 = np.mean([y_test[i] in top3_preds[i] for i in range(len(y_test))]) * 100.0
    try:
        auc = roc_auc_score(y_test, probs, multi_class="ovr", average="macro") * 100.0
    except Exception:
        auc = float("nan")
    return top1, top3, auc, preds


def main():
    device = get_device()
    checkpoint_path = "checkpoints/clmr_best_backbone.pt"
    data_dir = "data/genres_original"
    output_probe_path = "checkpoints/linear_probe_best.pt"
    reports_dir = "reports"
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    EPOCHS = 250
    BATCH_SIZE = 64
    NUM_SEGMENTS = 10

    print("\n" + "=" * 65)
    print(" GRID SEARCH LINEAR PROBE: RICERCA CONFIGURAZIONE OTTIMALE")
    print(f" Dispositivo attivo: {device}")
    print(f" Backbone in uso:    {checkpoint_path}")
    print("=" * 65)

    # Su Windows num_workers=0 o 2 evita bottleneck di spawn
    train_loader, test_loader = get_downstream_dataloaders(
        data_dir=data_dir,
        batch_size=BATCH_SIZE,
        num_workers=0,
        seed=42,
        num_eval_segments=NUM_SEGMENTS,
    )

    encoder = CLMRModel(pretrained_backbone=False)
    ckpt = load_checkpoint(checkpoint_path, device=device)
    encoder.load_state_dict(ckpt["model_state_dict"])
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # 1. Estrazione Feature Offline
    print("\n[1/3] Estrazione feature del Train Set...")
    train_feats, train_targets = [], []
    with torch.no_grad():
        for specs, targets in tqdm(train_loader, desc="Train Embeddings"):
            feats = encoder.encode(specs.to(device))
            train_feats.append(feats.cpu())
            train_targets.append(targets.cpu())

    X_train_raw = torch.cat(train_feats, dim=0)
    y_train = torch.cat(train_targets, dim=0)
    X_train = torch.nn.functional.normalize(X_train_raw, p=2, dim=-1)

    print("[2/3] Estrazione feature del Test Set (Multi-Segmento)...")
    test_feats_grouped = []
    test_targets_list = []
    with torch.no_grad():
        for multi_specs, targets in tqdm(test_loader, desc="Test Embeddings"):
            B, S, C, H, W = multi_specs.shape
            specs_flat = multi_specs.view(B * S, C, H, W).to(device)
            feats = encoder.encode(specs_flat)
            feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
            test_feats_grouped.append(feats.view(B, S, -1).cpu())
            test_targets_list.append(targets.cpu())

    X_test_grouped = torch.cat(test_feats_grouped, dim=0)
    y_test = torch.cat(test_targets_list, dim=0).numpy()

    # Dati in RAM: caricamento immediato con num_workers=0
    train_dataset = TensorDataset(X_train, y_train)
    train_loader_fast = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. Griglia di Iperparametri
    param_grid = {
        "lr": [0.005, 0.01, 0.03, 0.05, 0.1],
        "weight_decay": [0.0, 1e-4, 1e-3],
        "optimizer": ["AdamW", "SGD"]
    }

    combinations = list(itertools.product(
        param_grid["lr"],
        param_grid["weight_decay"],
        param_grid["optimizer"]
    ))

    print(f"\n[3/3] Avvio valutazione su {len(combinations)} configurazioni...")
    print(f"       Epoche per modello: {EPOCHS} (su RAM)\n")

    best_top1 = -1.0
    best_config = None
    best_state_dict = None
    best_report = None
    records = []
    history_loss = {}

    criterion = nn.CrossEntropyLoss()

    for lr, wd, opt_name in combinations:
        cfg_name = f"{opt_name}_lr{lr}_wd{wd}"
        torch.manual_seed(42)
        classifier = AudioLinearClassifier(in_features=512, num_classes=len(GTZAN_GENRES)).to(device)

        if opt_name == "AdamW":
            opt = optim.AdamW(classifier.parameters(), lr=lr, weight_decay=wd)
        else:
            opt = optim.SGD(classifier.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=wd)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)

        loss_history = []
        for epoch in range(1, EPOCHS + 1):
            classifier.train()
            total_loss = 0.0
            for bx, by in train_loader_fast:
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad()
                out = classifier(bx)
                loss = criterion(out, by)
                loss.backward()
                opt.step()
                total_loss += loss.item() * bx.size(0)
            scheduler.step()
            loss_history.append(total_loss / len(train_dataset))

        top1, top3, auc, preds = evaluate_model(classifier, X_test_grouped, y_test, device)
        final_loss = loss_history[-1]

        records.append({
            "config": cfg_name,
            "opt": opt_name,
            "lr": lr,
            "wd": wd,
            "final_loss": final_loss,
            "top1": top1,
            "top3": top3,
            "auc": auc
        })
        history_loss[cfg_name] = loss_history

        print(f"[{cfg_name:<28}] Loss: {final_loss:.4f} | Top-1: {top1:5.2f}% | Top-3: {top3:5.2f}% | AUC: {auc:5.2f}%")

        if top1 > best_top1:
            best_top1 = top1
            best_config = cfg_name
            best_state_dict = classifier.state_dict()
            best_report = classification_report(y_test, preds, target_names=GTZAN_GENRES, digits=3, zero_division=0)

    # 3. Salvataggio miglior modello
    torch.save(best_state_dict, output_probe_path)
    print("\n" + "=" * 65)
    print(" MIGLIORE CONFIGURAZIONE IDENTIFICATA")
    print("=" * 65)
    print(f" Configurazione: {best_config}")
    print(f" Top-1 Accuracy: {best_top1:.2f}%")
    print(f" Pesi salvati in: {output_probe_path}")
    print("\nReport Dettagliato per Classe (Best Model):")
    print(best_report)
    print("=" * 65)

    # 4. Grafico Comparativo
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sorted_by_top1 = sorted(records, key=lambda x: x["top1"], reverse=True)
    top5_cfgs = [r["config"] for r in sorted_by_top1[:5]]
    for cfg in top5_cfgs:
        plt.plot(history_loss[cfg], label=cfg)
    plt.title("Train Loss (Top-5 Configurazioni)")
    plt.xlabel("Epoca")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=8)

    plt.subplot(1, 2, 2)
    for opt_name in ["AdamW", "SGD"]:
        subset = [r for r in records if r["opt"] == opt_name and r["wd"] == 1e-4]
        subset.sort(key=lambda x: x["lr"])
        lrs = [s["lr"] for s in subset]
        accs = [s["top1"] for s in subset]
        plt.plot(lrs, accs, marker="o", label=f"{opt_name} (wd=1e-4)")

    plt.xscale("log")
    plt.title("Sensibilità al Learning Rate (wd=1e-4)")
    plt.xlabel("Learning Rate (Scala Log)")
    plt.ylabel("Top-1 Accuracy (%)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    plt.tight_layout()
    chart_path = os.path.join(reports_dir, "grid_search_results.png")
    plt.savefig(chart_path, dpi=300)
    print(f"\n[Grafico] Grafico salvato in: {chart_path}")


if __name__ == "__main__":
    main()