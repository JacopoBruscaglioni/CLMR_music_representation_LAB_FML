import argparse
import csv
import glob
import os
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


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Batch audio genre classification via GTZANDownstreamDataset"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["finetune", "probe"],
        default="finetune",
        help="Modalità checkpoint: 'finetune' (dizionario unico) o 'probe' (due checkpoint separati).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/finetuned_model_best.pt",
        help="Percorso del checkpoint per la modalità 'finetune'.",
    )
    parser.add_argument(
        "--backbone_path",
        type=str,
        default="checkpoints/clmr_best_backbone.pt",
        help="Percorso backbone per la modalità 'probe'.",
    )
    parser.add_argument(
        "--probe_path",
        type=str,
        default="checkpoints/linear_probe_best.pt",
        help="Percorso classificatore per la modalità 'probe'.",
    )
    parser.add_argument(
        "--music_dir",
        type=str,
        default="MY MUSIC TEST",
        help="Cartella contenente i file audio.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Nome file CSV di output (se omesso viene impostato in base al mode).",
    )
    return parser.parse_args()


def load_models(args, device):
    model = CLMRModel(pretrained_backbone=False).to(device)
    classifier = AudioLinearClassifier(in_features=512, num_classes=len(GTZAN_GENRES)).to(device)

    if args.mode == "finetune":
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint fine-tuning non trovato: {args.checkpoint}")
        
        ckpt = torch.load(args.checkpoint, map_location=device)
        print(f"Caricamento checkpoint congiunto: {args.checkpoint}")
        
        if "backbone_state_dict" in ckpt:
            model.load_state_dict(ckpt["backbone_state_dict"])
        else:
            model.load_state_dict(ckpt)

        if "classifier_state_dict" in ckpt:
            classifier.load_state_dict(ckpt["classifier_state_dict"])
        
        if "val_acc" in ckpt:
            print(f"Validation accuracy registrata nel checkpoint: {ckpt['val_acc']:.2f}%")

    elif args.mode == "probe":
        if not os.path.exists(args.backbone_path):
            raise FileNotFoundError(f"Backbone non trovata: {args.backbone_path}")
        if not os.path.exists(args.probe_path):
            raise FileNotFoundError(f"Linear probe non trovata: {args.probe_path}")

        print(f"Caricamento backbone CLMR: {args.backbone_path}")
        bb_ckpt = torch.load(args.backbone_path, map_location=device)
        model.load_state_dict(bb_ckpt.get("model_state_dict", bb_ckpt))

        print(f"Caricamento Linear Probe: {args.probe_path}")
        pr_ckpt = torch.load(args.probe_path, map_location=device)
        classifier.load_state_dict(pr_ckpt.get("model_state_dict", pr_ckpt))

    model.eval()
    classifier.eval()
    return model, classifier


def main():
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo in uso: {device}")

    # Output CSV di default coerente con il modello selezionato
    if args.output_csv is None:
        args.output_csv = f"predictions_{args.mode}.csv"

    model, classifier = load_models(args, device)

    # Scansione tracce
    supported_extensions = ("*.flac", "*.wav", "*.mp3", "*.ogg")
    audio_paths = []
    for ext in supported_extensions:
        audio_paths.extend(glob.glob(os.path.join(args.music_dir, ext)))
    audio_paths.sort()

    if not audio_paths:
        print(f"Nessun file audio trovato in '{args.music_dir}'.")
        return

    pipeline = AudioDataAugmentation()
    dummy_file_list = [(p, 0) for p in audio_paths]

    eval_dataset = GTZANDownstreamDataset(
        file_list=dummy_file_list,
        augmentation_pipeline=pipeline,
        eval_mode=True,
        num_segments=30, ### da 10 a 30
    )

    print(f"\nAvvio classificazione su {len(eval_dataset)} tracce ({args.mode})...\n")
    results = []

    for idx in range(len(eval_dataset)):
        segments, _ = eval_dataset[idx]
        segments = segments.to(device)
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

        print(f"[{idx+1:02d}/{len(eval_dataset):02d}] {filename[:32]:<32} -> 1°: {top1_genre:<8} ({top1_conf:5.1f}%) | 2°: {top2_genre:<8} ({top2_conf:5.1f}%)")

        results.append({
            "file_name": filename,
            "top1_genre": top1_genre,
            "top1_confidence_%": round(top1_conf, 2),
            "top2_genre": top2_genre,
            "top2_confidence_%": round(top2_conf, 2),
            "top3_genre": top3_genre,
            "top3_confidence_%": round(top3_conf, 2),
        })

    if results:
        with open(args.output_csv, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"\nReport esportato in: {args.output_csv}")


if __name__ == "__main__":
    main()