import random
import torch
import torch.nn as nn
import torchaudio.transforms as T


class AudioDataAugmentation(nn.Module):
    """
    Catena di trasformazioni audio per contrastive learning:
    1. Random Crop a finestra temporale fissa
    2. Log-Mel Spectrogram (n_fft=2048, hop_length=512, n_mels=64)
    3. SpecAugment (Frequency & Time Masking stocastici)
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        clip_length_s: float = 3.0,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 64,
        freq_mask_param: int = 15,
        time_mask_param: int = 35,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.target_samples = int(sample_rate * clip_length_s)

        # Conversione in spettrogramma Log-Mel
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
        self.amplitude_to_db = T.AmplitudeToDB()

        # SpecAugment
        self.freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_mask = T.TimeMasking(time_mask_param=time_mask_param)

    def _random_crop(self, audio: torch.Tensor) -> torch.Tensor:
        """Estrae una finestra casuale di lunghezza target_samples."""
        num_samples = audio.shape[-1]
        if num_samples < self.target_samples:
            padding = self.target_samples - num_samples
            return torch.nn.functional.pad(audio, (0, padding))

        max_start = num_samples - self.target_samples
        start_idx = random.randint(0, max_start)
        return audio[..., start_idx : start_idx + self.target_samples]

    def _apply_random_gain(self, audio: torch.Tensor, min_db: float = -6.0, max_db: float = 6.0) -> torch.Tensor:
        """Applica una variazione stocastica di volume."""
        gain_db = random.uniform(min_db, max_db)
        gain_factor = 10.0 ** (gain_db / 20.0)
        return audio * gain_factor

    def forward_single(self, audio: torch.Tensor) -> torch.Tensor:
        """Applica la trasformazione producendo un singolo spettrogramma normalizzato."""
        # 1. Crop e gain
        x = self._random_crop(audio)
        x = self._apply_random_gain(x)

        # 2. Log-Mel Spectrogram
        mel = self.mel_transform(x)
        log_mel = self.amplitude_to_db(mel)

        # 3. SpecAugment (applicato con probabilità dell'80%)
        if random.random() < 0.8:
            log_mel = self.freq_mask(log_mel)
        if random.random() < 0.8:
            log_mel = self.time_mask(log_mel)

        # Normalizzazione z-score locale
        mean = log_mel.mean()
        std = log_mel.std().clamp(min=1e-6)
        log_mel = (log_mel - mean) / std

        return log_mel

    def forward(self, audio: torch.Tensor):
        """
        Ritorna la coppia di viste positive (view_1, view_2).
        Ogni vista ha dimensione: [1, n_mels, time_steps]
        """
        view_1 = self.forward_single(audio)
        view_2 = self.forward_single(audio)
        return view_1, view_2


if __name__ == "__main__":
    # Test diagnostico rapido
    print("Testing AudioDataAugmentation...")
    dummy_waveform = torch.randn(1, 22050 * 5)  # 5 secondi di audio fittizio
    augmenter = AudioDataAugmentation()
    v1, v2 = augmenter(dummy_waveform)
    print("Test completato!")
    print(f"Shape Vista 1: {v1.shape}")
    print(f"Shape Vista 2: {v2.shape}")