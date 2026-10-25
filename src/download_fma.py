"""
download_fma.py

Script per scaricare ed estrarre fma_small.zip (~7.2 GB).
"""
import os
import sys
import zipfile
import urllib.request
from pathlib import Path

FMA_SMALL_URL = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"


def download_progress(count, block_size, total_size):
    if total_size > 0:
        percent = int(count * block_size * 100 / total_size)
        mb = (count * block_size) // (1024 * 1024)
        total_mb = total_size // (1024 * 1024)
        sys.stdout.write(f"\r[FMA Download] {percent:02d}% [{mb:,}/{total_mb:,} MB]")
        sys.stdout.flush()


def main(dest_dir: str = "data"):
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    extracted_dir = dest_path / "fma_small"
    zip_path = dest_path / "fma_small.zip"

    if extracted_dir.exists() and any(extracted_dir.iterdir()):
        print(f"Dataset FMA-small già presente in: {extracted_dir}")
        return

    # 1. Download
    if not zip_path.exists():
        print(f"Scaricamento di fma_small.zip (~7.2 GB) da {FMA_SMALL_URL}...")
        opener = urllib.request.build_opener()
        opener.addheaders = [("User-Agent", "Mozilla/5.0")]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(FMA_SMALL_URL, zip_path, reporthook=download_progress)
        print("\nDownload completato con successo.")

    # 2. Estrazione
    print(f"Estrazione in corso in '{dest_dir}' (può richiedere qualche minuto)...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dest_path)
    print("Estrazione completata.")

    # 3. Pulizia archivio compresso per liberare spazio su disco/Colab
    if zip_path.exists():
        zip_path.unlink()
        print(f"Rimosso file temporaneo: {zip_path}")


if __name__ == "__main__":
    main()