"""
src/losses.py

Implementazione vettorizzata della NT-Xent Loss (Normalized Temperature-scaled Cross Entropy Loss)
utilizzata in SimCLR e CLMR.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    Calcola la NT-Xent loss su un batch di proiezioni di dimensione [N, D] per due viste.
    
    Argomenti:
        temperature (float): Iperparametro tau che scala i logit di similarità (default: 0.1).
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Argomenti:
            z_i: Proiezioni della vista 1, shape [batch_size, proj_dim]
            z_j: Proiezioni della vista 2, shape [batch_size, proj_dim]
            
        Ritorna:
            loss: Scalare torch.Tensor
        """
        batch_size = z_i.shape[0]
        device = z_i.device

        # 1. Normalizzazione L2 sui vettori di proiezione
        z_i_norm = F.normalize(z_i, p=2, dim=1)
        z_j_norm = F.normalize(z_j, p=2, dim=1)

        # 2. Concatena le due viste: shape [2 * N, proj_dim]
        representations = torch.cat([z_i_norm, z_j_norm], dim=0)

        # 3. Matrice di similarità coseno: shape [2 * N, 2 * N]
        similarity_matrix = torch.matmul(representations, representations.T) / self.temperature

        # 4. Crea i target delle coppie positive:
        # Per la prima metà (0 .. N-1), la coppia positiva si trova a (N .. 2N-1)
        # Per la seconda metà (N .. 2N-1), la coppia positiva si trova a (0 .. N-1)
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=device),
            torch.arange(0, batch_size, device=device)
        ], dim=0)

        # 5. Maschera per eliminare i confronti con se stessi (diagonale principale)
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        # Sostituisci i valori sulla diagonale con -inf così la softmax assegna peso nullo
        similarity_matrix = similarity_matrix.masked_fill(mask, -float("inf"))

        # 6. Calcolo della Cross-Entropy loss standard
        loss = F.cross_entropy(similarity_matrix, labels)

        return loss


if __name__ == "__main__":
    print("Verifica diagnostica NTXentLoss...")

    batch_size = 4
    proj_dim = 128
    loss_fn = NTXentLoss(temperature=0.1)

    # Caso 1: Viste identiche (la loss deve essere minima)
    z1 = torch.randn(batch_size, proj_dim)
    z2 = z1.clone()
    perfect_loss = loss_fn(z1, z2)
    print(f"Loss con viste identiche (target ideale basso): {perfect_loss.item():.4f}")

    # Caso 2: Viste casuali e indipendenti
    z1_rand = torch.randn(batch_size, proj_dim)
    z2_rand = torch.randn(batch_size, proj_dim)
    rand_loss = loss_fn(z1_rand, z2_rand)
    print(f"Loss con vettori casuali:                         {rand_loss.item():.4f}")

    assert rand_loss > perfect_loss, "Errore: la loss casuale dovrebbe essere maggiore di quella perfetta!"
    print("NTXentLoss validata con successo!")