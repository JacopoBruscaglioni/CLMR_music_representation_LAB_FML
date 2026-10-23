import os
import glob
import csv
import torch
import torch.nn.functional as F

try:
    from dataset import GTZANDownstreamDataset, GTZAN_GENRES
    from augmentations import AudioDataAugmentation
    from models import CLMRModel, AudioLinearClassifier
except ModuleNotFoundError:
    from src.dataset import GTZANDownstreamDataset, GTZAN_GENRES
    from src.augmentations import AudioDataAugmentation
    from src.models import CLMRModel, AudioLinearClassifier

def main():
    music_dir = "MY MUSIC TEST"
    output_csv = "predictions_report.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Carica i modelli (stessa struttura del training)
    model = CLMRModel(pretrained_backbone=False).to(device)
    ckpt = torch.load("checkpoints/clmr_best_backbone.pt", map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()

    classifier = AudioLinearClassifier(in_features=512, num_classes=len(GTZAN_GENRES)).to(device)
    probe_ckpt = torch.load("checkpoints/linear_probe_best.pt", map_location=device)
    classifier.load_state_dict(probe_ckpt.get("model_state_dict", probe_ckpt))
    classifier.eval()

    # 2. Raccoglie i file
    supported_extensions = ("*.flac", "*.wav", "*.mp3", "*.ogg")
    audio_paths = []
    for ext in supported_extensions:
        audio_paths.extend(glob.glob(os.path.join(music_dir, ext)))
    audio_paths.sort()

    if not audio_paths:
        print(f"Nessun file audio trovato in '{music_dir}'.")
        return

    # 3. USA DIRETTAMENTE IL DATASET DEL TRAINING
    # Creiamo la lista nel formato atteso: [(path, dummy_label), ...]
    dummy_file_list = [(path, 0) for path in audio_paths]
    pipeline = AudioDataAugmentation()

    # eval_mode=True estrae direttamente i 10 segmenti già preprocessati e standardizzati
    eval_dataset = GTZANDownstreamDataset(
        file_list=dummy_file_list,
        augmentation_pipeline=pipeline,
        eval_mode=True,
        num_segments=10
    )

    print(f"Avvio classificazione di {len(eval_dataset)} tracce con GTZANDownstreamDataset...\n")
    results = []

    # 4. Inferenza
    for idx in range(len(eval_dataset)):
        # Il dataset gestisce da solo caricamento, mono, resampling, decibel e z-score
        segments, _ = eval_dataset[idx]
        segments = segments.to(device)  # Shape: [10, 1, 64, 130]
        filename = os.path.basename(audio_paths[idx])

        with torch.no_grad():
            h = model.encode(segments)
            logits = classifier(h)
            probs = F.softmax(logits, dim=1).mean(dim=0)

        sorted_indices = torch.argsort(probs, descending=True)
        top1_genre = GTZAN_GENRES[sorted_indices[0].item()]
        top1_conf = probs[sorted_indices[0].item()].item() * 100
        top2_genre = GTZAN_GENRES[sorted_indices[1].item()]
        top2_conf = probs[sorted_indices[1].item()].item() * 100
        top3_genre = GTZAN_GENRES[sorted_indices[2].item()]
        top3_conf = probs[sorted_indices[2].item()].item() * 100

        print(f"[{idx+1:02d}/{len(eval_dataset):02d}] {filename[:30]:<30} -> 1°: {top1_genre:<8} ({top1_conf:5.1f}%) | 2°: {top2_genre:<8} ({top2_conf:5.1f}%)")

        results.append({
            "file_name": filename,
            "top1_genre": top1_genre,
            "top1_confidence_%": round(top1_conf, 2),
            "top2_genre": top2_genre,
            "top2_confidence_%": round(top2_conf, 2),
            "top3_genre": top3_genre,
            "top3_confidence_%": round(top3_conf, 2),
        })

    # 5. Scrittura CSV
    if results:
        with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"\nReport salvato in: {output_csv}")

if __name__ == "__main__":
    main()