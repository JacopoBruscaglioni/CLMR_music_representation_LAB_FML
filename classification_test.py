import argparse
import os
import torch
import torch.nn.functional as F
import soundfile as sf
import torchaudio

try:
    from src.dataset import GTZAN_GENRES, load_audio_file
    from src.augmentations import AudioDataAugmentation
    from src.models import CLMRModel, AudioLinearClassifier
except ModuleNotFoundError:
    from dataset import GTZAN_GENRES, load_audio_file
    from augmentations import AudioDataAugmentation
    from models import CLMRModel, AudioLinearClassifier


def spec_from_audio(audio_chunk: torch.Tensor, augmenter: AudioDataAugmentation) -> torch.Tensor:
    """Replica esattamente la trasformazione usata in GTZANDownstreamDataset._spec_from_audio."""
    mel = augmenter.mel_transform(audio_chunk)
    log_mel = augmenter.amplitude_to_db(mel)
    mean = log_mel.mean()
    std = log_mel.std().clamp(min=1e-3)
    return (log_mel - mean) / std


def preprocess_track(audio_path: str, augmenter: AudioDataAugmentation, num_segments: int = 10) -> torch.Tensor:
    waveform = load_audio_file(audio_path, target_sr=augmenter.sample_rate)

    waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8) 

    max_val = torch.max(torch.abs(waveform))
    if max_val > 0:
        waveform = waveform / max_val

    total_samples = waveform.shape[-1]
    win_samples = augmenter.target_samples

    if total_samples <= win_samples:
        pad = win_samples - total_samples
        waveform = F.pad(waveform, (0, pad))
        segments = [spec_from_audio(waveform, augmenter)] * num_segments
    else:
        stride = (total_samples - win_samples) // max(1, (num_segments - 1))
        segments = []
        for s in range(num_segments):
            start = s * stride
            chunk = waveform[..., start : start + win_samples]
            segments.append(spec_from_audio(chunk, augmenter))

    return torch.stack(segments, dim=0)


def main():
    parser = argparse.ArgumentParser(description="Test classificazione traccia singola")
    parser.add_argument("--audio_path", type=str, required=True, help="Percorso del file audio")
    parser.add_argument("--backbone_path", type=str, default="checkpoints/clmr_best_backbone.pt")
    parser.add_argument("--probe_path", type=str, default="checkpoints/linear_probe_best.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pipeline di trasformazione originale
    augmenter = AudioDataAugmentation()

    # Caricamento Backbone
    model = CLMRModel(pretrained_backbone=False).to(device)
    ckpt = torch.load(args.backbone_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # Caricamento Linear Probe
    classifier = AudioLinearClassifier(in_features=512, num_classes=len(GTZAN_GENRES)).to(device)
    probe_ckpt = torch.load(args.probe_path, map_location=device)
    if isinstance(probe_ckpt, dict) and "model_state_dict" in probe_ckpt:
        classifier.load_state_dict(probe_ckpt["model_state_dict"])
    else:
        classifier.load_state_dict(probe_ckpt)
    classifier.eval()

    # Preprocessing identico al downstream test set
    segments = preprocess_track(args.audio_path, augmenter, num_segments=10).to(device)

    with torch.no_grad():
        h = model.encode(segments)
        logits = classifier(h)
        probs = F.softmax(logits, dim=1).mean(dim=0)

    sorted_indices = torch.argsort(probs, descending=True)

    print("\n" + "=" * 55)
    print(f"🎵 BRANO: {os.path.basename(args.audio_path)}")
    print("=" * 55)
    for rank, idx in enumerate(sorted_indices, 1):
        genre = GTZAN_GENRES[idx.item()]
        conf = probs[idx.item()].item() * 100
        bar = "█" * int(conf // 4)
        print(f"{rank:2d}. {genre:<10} {conf:6.2f}% | {bar}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()