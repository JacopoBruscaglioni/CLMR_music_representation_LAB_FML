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
import torch.nn as nn
import torchaudio
from tqdm import tqdm

from src import (
    CLMRModel,
    NTXentLoss,
    get_clmr_dataloaders,
    get_device,
    save_checkpoint,
    AverageMeter,
)

class GPUAudioPipeline(nn.Module):
    """Esegue estrazione Mel ed aumentazioni batch direttamente sulla GPU."""
    def __init__(self, sample_rate=22050, n_fft=2048, hop_length=512, n_mels=64):
        super().__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def augment_batch(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Random Gain (variazione ampiezza) sul batch in parallelo
        gains = (torch.rand(x.size(0), 1, 1, device=x.device) * 0.4) + 0.8  # [0.8, 1.2]
        x = x * gains

        # 2. Rumore Gaussiano leggero
        if self.training and torch.rand(1).item() > 0.5:
            noise = torch.randn_like(x) * 0.005
            x = x + noise

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [Batch, 1, Samples]
        x = self.augment_batch(x)
        mel = self.mel_transform(x)
        mel_db = self.amplitude_to_db(mel)
        return mel_db

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pre-training CLMR (Contrastive Learning of Musical Representations)"
    )
    # Percorsi

    parser.add_argument(
        "--samples_per_genre",
        type=int,
        default=None,
        help="Numero di tracce per genere per la stratificazione di FMA (es. 300). Se omesso, usa l'intero dataset.",
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default="data/fma_metadata/tracks.csv",
        help="Percorso al file tracks.csv dei metadati di FMA.",
    )

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
    parser.add_argument("--val_split", type=float, default=0.2, help="Percentuale di dati per la validazione")
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


def train_one_epoch(model, dataloader, criterion, optimizer, gpu_pipeline, device, epoch):
    model.train()
    gpu_pipeline.train()
    total_loss = 0.0

    for batch_idx, (view_1, view_2) in enumerate(dataloader):
        # 1. Spostamento asincrono non bloccante in VRAM
        view_1 = view_1.to(device, non_blocking=True)
        view_2 = view_2.to(device, non_blocking=True)

        # 2. Aumentazione ed estrazione Mel su GPU
        with torch.no_grad():
            feat_1 = gpu_pipeline(view_1)
            feat_2 = gpu_pipeline(view_2)

        # 3. Forward pass ResNet e calcolo loss
        optimizer.zero_grad()
        out_1 = model(feat_1)
        out_2 = model(feat_2)

        loss = criterion(out_1, out_2)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, gpu_pipeline, device):
    model.eval()
    gpu_pipeline.eval()
    total_loss = 0.0

    with torch.no_grad():
        for view_1, view_2 in dataloader:
            view_1 = view_1.to(device, non_blocking=True)
            view_2 = view_2.to(device, non_blocking=True)

            feat_1 = gpu_pipeline(view_1)
            feat_2 = gpu_pipeline(view_2)

            out_1 = model(feat_1)
            out_2 = model(feat_2)

            loss = criterion(out_1, out_2)
            total_loss += loss.item()

    return total_loss / len(dataloader)

def main():
    args = parse_args()
    device = get_device()

    print("\n" + "=" * 60)
    print(" AVVIO PRE-TRAINING CLMR (SELF-SUPERVISED)")
    print(f" Dataset:       {args.data_dir}")
    print(f" Epoche:        {args.epochs}")
    print(f" Batch size:    {args.batch_size}")
    print(f" Val split:     {args.val_split}")
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
            val_split=args.val_split,
            samples_per_genre=args.samples_per_genre,
            metadata_path=args.metadata_path,
        )
        print(f"[Dataset] Batch train/epoca: {len(train_loader)} | Batch val: {len(val_loader)}")
    except FileNotFoundError as e:
        print(f"\n[ERRORE] {e}")
        print("I file audio non sono ancora presenti nella cartella data/.")
        sys.exit(1)

    gpu_pipeline = GPUAudioPipeline().to(device)
    # Inizializza modello, loss e ottimizzatore
    model = CLMRModel(pretrained_backbone=args.pretrained).to(device)
    criterion = NTXentLoss(temperature=args.temperature).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, gpu_pipeline, device, epoch)
        val_loss = validate(model, val_loader, criterion, gpu_pipeline, device)
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