"""
src/utils.py

Funzioni di utilità per:
- Rilevamento e selezione del device di calcolo (CUDA / CPU).
- Salvataggio e caricamento dei checkpoint del modello.
- Calcolo metriche di classificazione (Accuracy Top-1 e Top-k).
- Tracciamento della loss media durante i cicli di training.
"""

import os
from pathlib import Path
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn


def get_device() -> torch.device:
    """
    Rileva e restituisce il device disponibile (CUDA se presente, altrimenti CPU).
    Stampa le informazioni sull'hardware rilevato.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[Device] Utilizzo GPU: {gpu_name}")
    else:
        device = torch.device("cpu")
        print("[Device] CUDA non disponibile. Utilizzo CPU.")
    return device


class AverageMeter:
    """
    Tiene traccia del valore corrente, della media, della somma e del conteggio
    di una grandezza scalare (es. loss o accuratezza).
    """

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0


def calculate_accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> Dict[int, float]:
    """
    Calcola l'accuratezza per i valori indicati in topk (es. Top-1, Top-3).
    Argomenti:
        output: Logit o probabilità del modello, shape [B, num_classes]
        target: Etichette reali di classe, shape [B]
        topk: Tupla con i valori di k da calcolare
    Ritorna:
        Dizionario {k: accuratezza_in_percentuale}
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = {}
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res[k] = (correct_k.mul_(100.0 / batch_size)).item()
        return res


def save_checkpoint(
    state_dict: Dict[str, Any],
    checkpoint_dir: str = "checkpoints",
    filename: str = "clmr_checkpoint.pt"
) -> str:
    """
    Salva il dizionario di stato (pesi, epoca, ottimizzatore) su disco.
    Crea la cartella di destinazione se non esiste.
    """
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    save_path = os.path.join(checkpoint_dir, filename)
    torch.save(state_dict, save_path)
    print(f"[Checkpoint] Modello salvato con successo in: {save_path}")
    return save_path


def load_checkpoint(
    checkpoint_path: str,
    device: torch.device
) -> Dict[str, Any]:
    """
    Carica un checkpoint da disco rimappandolo sul device indicato.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"[Checkpoint] File non trovato: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"[Checkpoint] Caricato con successo da: {checkpoint_path}")
    return checkpoint


if __name__ == "__main__":
    print("Verifica modulo src/utils.py...")

    # Test device
    dev = get_device()

    # Test AverageMeter
    meter = AverageMeter("Loss")
    meter.update(2.5, n=2)
    meter.update(1.5, n=2)
    print(f"Media attesa: 2.0 -> Calcolata: {meter.avg}")
    assert abs(meter.avg - 2.0) < 1e-6

    # Test Accuracy
    dummy_logits = torch.tensor([[0.1, 2.5, 0.3], [1.8, 0.2, 0.5]])  # Classi predette: 1 e 0
    dummy_targets = torch.tensor([1, 0])                              # Entrambe corrette
    acc = calculate_accuracy(dummy_logits, dummy_targets, topk=(1,))
    print(f"Top-1 Accuracy attesa: 100.0% -> Calcolata: {acc[1]:.1f}%")

    print("Modulo utils validato con successo!")