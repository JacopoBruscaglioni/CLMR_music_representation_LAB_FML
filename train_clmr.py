"""
train_clmr.py

Script principale per il pre-training Contrastive Self-Supervised (CLMR).
Addestra la ResNet-18 su GTZAN senza usare le etichette di genere musicale.
"""

import argparse
import os
import sys
import time

import torch
import torch.optim as optim
from tqdm import tqdm

from src import (
    CLMRModel,
    NTXentLoss,
    get_clmr_dataloaders,
    get_device,
    save_checkpoint,
    AverageMeter,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pre-training CLMR (Contrastive Learning of Musical Representations)"
    )
    # Percorsi
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/genres_original",
        help="Percorso della cartella contenente i file audio di GTZAN suddivisi per generi",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Cartella dove salvare i pesi del modello",
    )

    # Parametri di addestramento
    parser.add_argument("--epochs", type=int, default=50, help="Numero di epoche di pre-training")
    parser.add_argument("--batch_size", type=int, default=32, help="Dimensione del batch")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate iniziale")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Regolarizzazione L2")
    parser.add_argument("--temperature", type=float, default=0.1, help="Parametro tau per NT-Xent")
    parser.add_argument("--pretrained", action="store_true", help="Inizializza con pesi ResNet-18 ImageNet")

    # Sistema
    parser.add_argument("--num_workers", type=int, default=0, help="Worker DataLoader (0 su Windows)")
    parser.add_argument("--seed", type=int, default=42, help="Seed di riproducibilità")

    return parser.parse_args()


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    loss_meter = AverageMeter("Train Loss")

    pbar = tqdm(dataloader, desc=f"Epoca [{epoch}] Train", leave=False)
    for view_1, view_2 in pbar:
        view_1 = view_1.to(device)
        view_2 = view_2.to(device)

        # Forward pass: estrae le proiezioni z_1 e z_2
        _, z_1 = model(view_1)
        _, z_2 = model(view_2)

        # Calcolo NT-Xent loss
        loss = criterion(z_1, z_2)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), n=view_1.size(0))
        pbar.set_postfix({"Loss": f"{loss_meter.avg:.4f}"})

    return loss_meter.avg


def validate(model, dataloader, criterion, device):
    model.eval()
    loss_meter = AverageMeter("Val Loss")

    with torch.no_grad():
        for view_1, view_2 in dataloader:
            view_1 = view_1.to(device)
            view_2 = view_2.to(device)

            _, z_1 = model(view_1)
            _, z_2 = model(view_2)

            loss = criterion(z_1, z_2)
            loss_meter.update(loss.item(), n=view_1.size(0))

    return loss_meter.avg


def main():
    args = parse_args()
    device = get_device()

    print("\n" + "=" * 60)
    print(" AVVIO PRE-TRAINING CLMR (SELF-SUPERVISED)")
    print(f" Dataset:       {args.data_dir}")
    print(f" Epoche:        {args.epochs}")
    print(f" Batch size:    {args.batch_size}")
    print(f" Temperatura:   {args.temperature}")
    print(f" Learning rate: {args.lr}")
    print(f" Pretrained:    {args.pretrained}")
    print("=" * 60 + "\n")

    # Carica DataLoader
    try:
        train_loader, val_loader = get_clmr_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
        )
        print(f"[Dataset] Batch train/epoca: {len(train_loader)} | Batch val: {len(val_loader)}")
    except FileNotFoundError as e:
        print(f"\n[ERRORE] {e}")
        print("I file audio non sono ancora presenti nella cartella data/.")
        sys.exit(1)

    # Inizializza modello, loss e ottimizzatore
    model = CLMRModel(pretrained_backbone=args.pretrained).to(device)
    criterion = NTXentLoss(temperature=args.temperature).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - epoch_start
        print(
            f"Epoca [{epoch:03d}/{args.epochs:03d}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f} | "
            f"Tempo: {elapsed:.1f}s"
        )

        # Salva se la validation loss migliora
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                state_dict={
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                },
                checkpoint_dir=args.checkpoint_dir,
                filename="clmr_best_backbone.pt",
            )

    tot_min = (time.time() - start_time) / 60
    print("\n" + "=" * 60)
    print(f" Pre-training completato in {tot_min:.2f} minuti!")
    print(f" Miglior Val Loss: {best_val_loss:.4f}")
    print(" Pesi salvati in: checkpoints/clmr_best_backbone.pt")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()