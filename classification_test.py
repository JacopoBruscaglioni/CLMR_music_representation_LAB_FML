import argparse
import os
import torch
import torch.nn.functional as F
import soundfile as sf
import torchaudio.transforms as T
from src.model import CLMREncoder, LinearProbeClassifier

GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]

def load_and_preprocess_audio(audio_path, sample_rate=22050, num_segments=10, segment_duration=2.97):
    # Caricamento del file audio tramite soundfile
    data, orig_sr = sf.read(audio_path, dtype='float32')
    tensor = torch.from_numpy(data)

    # Conversione in mono [1, campioni] se stereo o multicanale
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 2:
        tensor = tensor.mean(dim=1, keepdim=True).t()

    # Ricampionamento alla frequenza target della pipeline (22050 Hz)
    if orig_sr != sample_rate:
        resampler = T.Resample(orig_sr, sample_rate)
        tensor = resampler(tensor)

    total_samples = tensor.shape[-1]
    target_samples = int(sample_rate * segment_duration)

    if total_samples < target_samples:
        raise ValueError(
            f"Il brano è troppo breve ({total_samples / sample_rate:.2f}s). "
            f"Durata minima richiesta: {segment_duration:.2f}s"
        )

    # Estrazione di 10 finestre uniformi distribuite lungo l'intera traccia
    step = (total_samples - target_samples) // max(1, (num_segments - 1))
    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    segments = []
    for i in range(num_segments):
        start = i * step
        chunk = tensor[..., start:start + target_samples]
        mel = mel_transform(chunk)
        log_mel = torch.log(torch.clamp(mel, min=1e-5))
        segments.append(log_mel)

    # Forma finale del tensore: [10, 1, 64, 130]
    return torch.stack(segments, dim=0)

def main():
    parser = argparse.ArgumentParser(description="Test di classificazione su brano singolo (Test-Time Aggregation)")
    parser.add_argument("--audio_path", type=str, required=True, help="Percorso del file audio da testare (.wav, .mp3, ecc.)")
    parser.add_argument("--backbone_path", type=str, default="checkpoints/clmr_best_backbone.pt")
    parser.add_argument("--probe_path", type=str, default="checkpoints/linear_probe_best.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Inizializzazione e caricamento dell'encoder ResNet-18
    encoder = CLMREncoder().to(device)
    encoder.load_state_dict(torch.load(args.backbone_path, map_location=device))
    encoder.eval()

    # Inizializzazione e caricamento della testa lineare
    probe = LinearProbeClassifier(in_features=512, num_classes=len(GENRES)).to(device)
    if not os.path.exists(args.probe_path):
        raise FileNotFoundError(f"File pesi non trovato in '{args.probe_path}'. Esegui prima train_linear_probe.py.")
    
    probe.load_state_dict(torch.load(args.probe_path, map_location=device))
    probe.eval()

    # Preprocessing delle 10 finestre
    segments = load_and_preprocess_audio(args.audio_path).to(device)

    # Inferenza con Test-Time Aggregation
    with torch.no_grad():
        features = encoder(segments)
        logits = probe(features)
        probs_per_segment = F.softmax(logits, dim=1)
        mean_probs = probs_per_segment.mean(dim=0)

    # Ordinamento decrescente delle confidenze
    sorted_indices = torch.argsort(mean_probs, descending=True)

    print("\n" + "=" * 55)
    print(f"🎵 RISULTATO CLASSIFICAZIONE: {os.path.basename(args.audio_path)}")
    print("=" * 55)
    for rank, idx in enumerate(sorted_indices, 1):
        genre = GENRES[idx.item()]
        conf = mean_probs[idx.item()].item() * 100
        bar = "█" * int(conf // 4)
        print(f"{rank:2d}. {genre:<10} {conf:6.2f}% | {bar}")
    print("=" * 55 + "\n")

if __name__ == "__main__":
    main()