

from typing import Tuple
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ProjectionHead(nn.Module):
    """
    Projection Head g(·) a 2 livelli con Batch Normalization e ReLU:
    input_dim (512) -> hidden_dim (512) -> output_dim (128)
    """

    def __init__(self, in_dim: int = 512, hidden_dim: int = 512, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CLMRModel(nn.Module):
    """
    Rete neurale completa per il pre-training contrastivo di CLMR.
    Comprende:
      1. Encoder ResNet-18: Spettrogramma (B, 1, 64, 130) -> Rappresentazione h (B, 512)
      2. Projection Head:   Rappresentazione h (B, 512)  -> Vettore di proiezione z (B, 128)
    """

    def __init__(
        self,
        pretrained_backbone: bool = False,
        embedding_dim: int = 512,
        projection_dim: int = 128,
    ):
        super().__init__()

        # Inizializza ResNet-18 standard
        weights = ResNet18_Weights.DEFAULT if pretrained_backbone else None
        base_resnet = resnet18(weights=weights)

        # --------------------------------------------------------------------
        # 1. Adattamento del primo strato conv1:
        # Standard: nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Nostro:   nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # --------------------------------------------------------------------
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        # Se volessimo sfruttare i pesi ImageNet pre-addestrati, mediamo i canali RGB
        if pretrained_backbone:
            with torch.no_grad():
                self.conv1.weight.copy_(
                    base_resnet.conv1.weight.mean(dim=1, keepdim=True)
                )

        # Estraiamo gli strati intermedi della ResNet-18
        self.bn1 = base_resnet.bn1
        self.relu = base_resnet.relu
        self.maxpool = base_resnet.maxpool

        self.layer1 = base_resnet.layer1
        self.layer2 = base_resnet.layer2
        self.layer3 = base_resnet.layer3
        self.layer4 = base_resnet.layer4

        # Adaptive pooling per ridurre la mappa di feature a dimensione (B, 512, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # --------------------------------------------------------------------
        # 2. Projection Head g(·)
        # --------------------------------------------------------------------
        self.projector = ProjectionHead(
            in_dim=embedding_dim,
            hidden_dim=embedding_dim,
            out_dim=projection_dim,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passaggio attraverso l'Encoder f(·) per estrarre il vettore h.
        Input:  [B, 1, n_mels, time_steps]
        Output: [B, 512]
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        h = torch.flatten(x, 1)  # Shape: [B, 512]
        return h

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass contrastivo.
        Ritorna:
            h: rappresentazione latente (B, 512)
            z: proiezione normalizzata per NT-Xent (B, 128)
        """
        h = self.encode(x)
        z = self.projector(h)
        return h, z


class AudioLinearClassifier(nn.Module):
    """
    Classificatore a singolo strato per il Linear Probing a valle.
    Prende le feature congelate h (512) e predice la classe di genere musicale (10).
    """

    def __init__(self, in_features: int = 512, num_classes: int = 10):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h)


if __name__ == "__main__":
    # Test diagnostico delle dimensioni del tensore
    print("Verifica architettura CLMRModel...")
    
    # Simuliamo un batch di 4 spettrogrammi da [1 canale, 64 bande mel, 130 frame]
    batch_size = 4
    dummy_input = torch.randn(batch_size, 1, 64, 130)

    # Inizializza modello
    model = CLMRModel(pretrained_backbone=False)
    h, z = model(dummy_input)

    print(f"Input shape:            {dummy_input.shape}")
    print(f"Encoder output h shape: {h.shape}  (atteso: [{batch_size}, 512])")
    print(f"Projector output z shape: {z.shape}  (atteso: [{batch_size}, 128])")

    # Verifica linear probe
    classifier = AudioLinearClassifier(in_features=512, num_classes=10)
    logits = classifier(h)
    print(f"Classifier logits shape: {logits.shape}  (atteso: [{batch_size}, 10])")
    print("Modello validato correttamente!")