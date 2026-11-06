"""
tune_and_save_prob.py
Grid Search su feature CLMR pre-estratte per identificare la migliore configurazione 
di ottimizzazione e il numero ottimale di segmenti TTA per il Linear Probe.
Genera una suite completa di grafici comparativi per l'analisi dei risultati.
"""

import os
import csv
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

from src import (
    CLMRModel,
    AudioLinearClassifier,
    get_downstream_dataloaders,
    get_device,
    load_checkpoint,
    GTZAN_GENRES,
)


def evaluate_model(classifier, X_test_grouped, y_test, device, num_segments=None):
    classifier.eval()
    with torch.no_grad():
        feats_dev = X_test_grouped.to(device)
        
        if num_segments is not None:
            feats_dev = feats_dev[:, :num_segments, :]
            
        N, S, D = feats_dev.shape
        logits = classifier(feats_dev.reshape(N * S, D)).reshape(N, S, -1)
        mean_logits = logits.mean(dim=1)
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


def plot_comparative_suite(records, history_loss, best_config, best_preds, y_test, genres, save_dir):
    """
    Genera una figura a 4 pannelli per documentare in modo chiaro
    l'impatto di ottimizzatori, learning rate, regolarizzazione e TTA.
    """
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    # -------------------------------------------------------------
    # 1. Curve di Training Loss (AdamW vs SGD)
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    for train_cfg, loss_curve in history_loss.items():
        is_best = train_cfg in best_config
        linestyle = "-" if "AdamW" in train_cfg else "--"
        linewidth = 2.5 if is_best else 1.2
        alpha = 1.0 if is_best else 0.65
        label = f"{train_cfg} (MIGLIORE)" if is_best else train_cfg
        ax1.plot(loss_curve, label=label, linestyle=linestyle, linewidth=linewidth, alpha=alpha)

    ax1.set_title("A. Dinamica di Convergenza (Training Loss)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoche", fontsize=11)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=8, loc="upper right", framealpha=0.9)

    # -------------------------------------------------------------
    # 2. Impatto dei Segmenti TTA sulle varie configurazioni
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    unique_train_cfgs = sorted(list(set(r["train_cfg"] for r in records)))
    
    for train_cfg in unique_train_cfgs:
        sub = [r for r in records if r["train_cfg"] == train_cfg]
        sub.sort(key=lambda x: x["segments"])
        segs = [r["segments"] for r in sub]
        top1_vals = [r["top1"] for r in sub]

        is_best = train_cfg in best_config
        linewidth = 2.8 if is_best else 1.3
        alpha = 1.0 if is_best else 0.55
        marker = "o" if "AdamW" in train_cfg else "s"
        ax2.plot(segs, top1_vals, label=train_cfg, marker=marker, linewidth=linewidth, alpha=alpha)

    ax2.set_title("B. Sensibilità TTA: Top-1 vs Numero di Segmenti", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Segmenti Aggregati per Brano", fontsize=11)
    ax2.set_ylabel("Top-1 Accuracy (%)", fontsize=11)
    ax2.set_xticks(sorted(list(set(r["segments"] for r in records))))
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=8, loc="lower right", framealpha=0.9)

    # -------------------------------------------------------------
    # 3. Confronto a Barre Top-1 vs AUC (Top-10 configurazioni)
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    sorted_records = sorted(records, key=lambda x: x["top1"], reverse=True)[:10]
    labels = [f"{r['train_cfg']}\nseg={r['segments']}" for r in sorted_records]
    top1_scores = [r["top1"] for r in sorted_records]
    auc_scores = [r["auc"] for r in sorted_records]

    y_pos = np.arange(len(labels))
    width = 0.38

    ax3.barh(y_pos - width / 2, top1_scores, width, label="Top-1 (%)", color="#1f77b4")
    ax3.barh(y_pos + width / 2, auc_scores, width, label="ROC-AUC Macro (%)", color="#ff7f0e", alpha=0.85)

    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(labels, fontsize=8)
    ax3.invert_yaxis()
    ax3.set_xlabel("Punteggio (%)", fontsize=11)
    ax3.set_title("C. Top-10 Configurazioni (Accuratezza vs Separabilità)", fontsize=13, fontweight="bold")
    ax3.grid(True, axis="x", linestyle=":", alpha=0.6)
    ax3.legend(loc="lower right", fontsize=9)

    # -------------------------------------------------------------
    # 4. Matrice di Confusione del Miglior Modello Assoluto
    # -------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    cm = confusion_matrix(y_test, best_preds, normalize="true")
    im = ax4.imshow(cm, interpolation="nearest", cmap="Blues")
    cbar = fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    cbar.set_label("Accuratezza Normalizzata", fontsize=10)

    tick_marks = np.arange(len(genres))
    ax4.set_xticks(tick_marks)
    ax4.set_xticklabels(genres, rotation=45, ha="right", fontsize=9)
    ax4.set_yticks(tick_marks)
    ax4.set_yticklabels(genres, fontsize=9)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > 0.5 else "black"
            ax4.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

    ax4.set_title(f"D. Confusion Matrix: {best_config}", fontsize=13, fontweight="bold")
    ax4.set_ylabel("Genere Reale", fontsize=11)
    ax4.set_xlabel("Genere Predetto", fontsize=11)

    plt.suptitle("Analisi Comparativa Iperparametri & TTA per CLMR Linear Probe", fontsize=16, fontweight="bold", y=0.98)
    
    out_path = os.path.join(save_dir, "grid_search_complete_dashboard.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n[Dashboard] Dashboard comparativa salvata in: {out_path}")


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
    MAX_SEGMENTS = 30

    print("\n" + "=" * 65)
    print(" GRID SEARCH LINEAR PROBE: RICERCA CONFIGURAZIONE & SEGMENTI")
    print(f" Dispositivo attivo: {device}")
    print(f" Backbone in uso:    {checkpoint_path}")
    print("=" * 65)

    train_loader, test_loader = get_downstream_dataloaders(
        data_dir=data_dir,
        batch_size=BATCH_SIZE,
        num_workers=0,
        seed=42,
        num_eval_segments=MAX_SEGMENTS,
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

    print(f"[2/3] Estrazione feature del Test Set (fino a {MAX_SEGMENTS} segmenti)...")
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

    train_dataset = TensorDataset(X_train, y_train)
    train_loader_fast = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. Griglia di Iperparametri
    param_grid = {
        "lr": [0.03, 0.05, 0.1],
        "weight_decay": [0.0, 1e-4],
        "optimizer": ["AdamW", "SGD"],
        "num_segments": [1, 5, 10, 15, 20, 30]
    }

    train_combinations = list(itertools.product(
        param_grid["lr"],
        param_grid["weight_decay"],
        param_grid["optimizer"]
    ))

    print(f"\n[3/3] Addestramento {len(train_combinations)} modelli e test su {len(param_grid['num_segments'])} opzioni di segmenti...")
    print(f"       Epoche per modello: {EPOCHS}\n")

    best_top1 = -1.0
    best_config = None
    best_state_dict = None
    best_report = None
    best_preds = None

    records = []
    history_loss = {}
    criterion = nn.CrossEntropyLoss()

    for lr, wd, opt_name in train_combinations:
        train_cfg_label = f"{opt_name}_lr{lr}_wd{wd}"
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
            loss_history.append(total_loss / len(train_loader_fast.dataset))

        history_loss[train_cfg_label] = loss_history

        # Valuta il modello allenato su ciascuna granularità di segmenti TTA
        for seg in param_grid["num_segments"]:
            full_cfg_name = f"{train_cfg_label}_seg{seg}"
            top1, top3, auc, preds = evaluate_model(classifier, X_test_grouped, y_test, device, num_segments=seg)

            records.append({
                "config": full_cfg_name,
                "train_cfg": train_cfg_label,
                "opt": opt_name,
                "lr": lr,
                "wd": wd,
                "segments": seg,
                "final_loss": loss_history[-1],
                "top1": top1,
                "top3": top3,
                "auc": auc
            })

            print(f"[{full_cfg_name:<30}] Loss: {loss_history[-1]:.4f} | Top-1: {top1:5.2f}% | Top-3: {top3:5.2f}% | AUC: {auc:5.2f}%")

            if top1 > best_top1:
                best_top1 = top1
                best_config = full_cfg_name
                best_state_dict = classifier.state_dict()
                best_report = classification_report(y_test, preds, target_names=GTZAN_GENRES, digits=3, zero_division=0)
                best_preds = preds

    # 3. Salvataggio miglior modello
    torch.save(best_state_dict, output_probe_path)
    print("\n" + "=" * 65)
    print(" MIGLIORE CONFIGURAZIONE IDENTIFICATA")
    print("=" * 65)
    print(f" Configurazione:  {best_config}")
    print(f" Top-1 Accuracy:  {best_top1:.2f}%")
    print(f" Pesi salvati in: {output_probe_path}")
    print("\nReport Dettagliato per Classe:")
    print(best_report)
    print("=" * 65)

    # 4. Esportazione CSV riassuntivo
    csv_path = os.path.join(reports_dir, "grid_search_summary.csv")
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"[CSV] Tabella comparativa esportata in: {csv_path}")

    # 5. Generazione Dashboard Grafica
    plot_comparative_suite(records, history_loss, best_config, best_preds, y_test, GTZAN_GENRES, reports_dir)


if __name__ == "__main__":
    main()