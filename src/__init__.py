"""
Pacchetto CLMR (Contrastive Learning of Musical Representations).
Esporta i moduli principali per augmentations, modelli, loss, dataset e utility.
"""


try:
    from src.augmentations import AudioDataAugmentation
    from src.dataset import (
        GTZAN_GENRES,
        GENRE_TO_IDX,
        GTZANCLMRDataset,
        GTZANDownstreamDataset,
        get_clmr_dataloaders,
        get_downstream_dataloaders,
    )
    from src.losses import NTXentLoss
    from src.models import CLMRModel, AudioLinearClassifier
    from src.utils import get_device, save_checkpoint, load_checkpoint, AverageMeter
except ModuleNotFoundError:
    from augmentations import AudioDataAugmentation
    from dataset import (
            GTZAN_GENRES,
            GENRE_TO_IDX,
            GTZANCLMRDataset,
            GTZANDownstreamDataset,
            get_clmr_dataloaders,
            get_downstream_dataloaders,
        )
    from losses import NTXentLoss
    from models import CLMRModel, AudioLinearClassifier
    from utils import get_device, save_checkpoint, load_checkpoint, AverageMeter

__all__ = [
    "AudioDataAugmentation",
    "GTZAN_GENRES",
    "GENRE_TO_IDX",
    "GTZANCLMRDataset",
    "GTZANDownstreamDataset",
    "get_clmr_dataloaders",
    "get_downstream_dataloaders",
    "NTXentLoss",
    "CLMRModel",
    "AudioLinearClassifier",
    "get_device",
    "save_checkpoint",
    "load_checkpoint",
    "AverageMeter",
]