import os
import copy
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import src

try:
    from src.dataset import get_downstream_dataloaders, GTZAN_GENRES
    from src.models import CLMRModel, AudioLinearClassifier
except ModuleNotFoundError:
    from dataset import get_downstream_dataloaders, GTZAN_GENRES
    from models import CLMRModel, AudioLinearClassifier

def main():
    # hyperparameters
    data_dir = "data/genres_original"
    backbone_path = "checkpoints/clmr_best_backbone.pt"
    save_path = "checkpoints/finetuned_model_best.pt"

    epochs = 25
    batch_size = 32
    num_eval_segments = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device usato per il fine-tuning: {device}")

    # Caricamento DataLoader
    train_loader, test_loader = get_downstream_dataloaders(data_dir, batch_size=batch_size, test_split=0.2 ,num_eval_segments=num_eval_segments)

    # istanziazione e caricamento backbone CLMR
    model = CLMRModel(pretrained_backbone = False).to(device)
    if os.path.exists(backbone_path):
        chkpt = torch.load(backbone_path, map_location=device)
        model.load_state_dict(chkpt.get("model_state_dict", chkpt))
        print(f"Backbone CLMR caricato da {backbone_path}")
    else: 
        print(f"Backbone CLMR non trovato in {backbone_path}. Assicurati di averlo scaricato o addestrato prima di eseguire il fine-tuning.")

    # instaziazione del classificatore a 10 classi
    classifier = AudioLinearClassifier(in_features=512, num_classes=len(GTZAN_GENRES)).to(device)

    # sblocco di parametri per fine-tuning
    for param in model.parameters():
        param.requires_grad = True
    for param in classifier.parameters():
        param.requires_grad = True

    # Config Learning rate
    optimizer = AdamW([
        {"params": model.parameters(), "lr": 1e-5, "weight_decay": 1e-3},
        {"params": classifier.parameters(), "lr": 1e-3, "weight_decay": 1e-4}
    ])

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print("Inizio del fine-tuning congiunto: ")

    for epoch in range(1, epochs +1):
        # Training
        model.train()
        classifier.train()
        total_train_loss = 0.0
        train_correct = 0
        train_total = 0

        for specs, labels in train_loader:
            specs = specs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            h = model.encode(specs)
            logits = classifier(h)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * specs.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()
        train_acc = 100.0 * train_correct / train_total
        avg_train_loss = total_train_loss / train_total

        # valutazione
        model.eval()
        classifier.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for segments, labels in test_loader:
                bsz, n_segs, c, f, t = segments.shape
                segments = segments.view(bsz * n_segs, c, f, t).to(device)
                labels = labels.to(device)

                h = model.encode(segments)
                logits = classifier(h)
                logits = logits.view(bsz, n_segs, -1).mean(dim=1)

                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100.0 * val_correct / val_total
        print(f"Epoca [{epoch:02d}/{epochs:02d}] - Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        # Salvataggio del modello migliore
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "backbone_state_dict": model.state_dict(),
                "classifier_state_dict": classifier.state_dict(),
                "val_acc": val_acc,
            }, save_path)
            print(f"  --> Nuovo record di accuratezza: {val_acc:.2f}% (Checkpoint salvato)")

    print(f"\nFine-Tuning completato. Miglior accuratezza sul validation set: {best_val_acc:.2f}%")

if __name__ == "__main__":
    main()
