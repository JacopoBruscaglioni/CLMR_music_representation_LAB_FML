import random
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T

class RandomDynamicCompression(nn.Module):
    """ loudness war, limiter digitali, compressione multiband moderna per poter confrontare brani con mixagio e mastering attuali"""
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            gamma = random.uniform(0.55, 1.35)
            sign = torch.sign(waveform)
            waveform = sign * (torch.abs(waveform) ** gamma)
            max_val = torch.max(torch.abs(waveform))
            if max_val > 0:
                waveform = waveform / max_val
        return waveform


class RandomSaturation(nn.Module):
    """ simula saturazione analogica, distorsione armonica, clipping soft e hard """
    def __init__(self, p: float = 0.4):
        super().__init__()
        self.p = p

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            gain = random.uniform(1.2, 3.0)
            waveform = torch.tanh(waveform * gain)
            max_val = torch.max(torch.abs(waveform))
            if max_val > 0:
                waveform = waveform / max_val
        return waveform


class RandomSpectralFilter(nn.Module):
    """Altera il bilanciamento tonale con filtri passa-basso o passa-alto casuali."""
    def __init__(self, sample_rate: int = 22050, p: float = 0.5):
        super().__init__()
        self.sample_rate = sample_rate
        self.p = p

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            if random.random() < 0.5:
                # Attenua le alte frequenze (simula master vintage o cupo)
                cutoff = random.uniform(3500, 7500)
                waveform = torchaudio.functional.lowpass_biquad(waveform, self.sample_rate, cutoff_freq=cutoff)
            else:
                # Attenua le basse frequenze (riduce il bias sui bassi/sub)
                cutoff = random.uniform(60, 250)
                waveform = torchaudio.functional.highpass_biquad(waveform, self.sample_rate, cutoff_freq=cutoff)
        return waveform

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
        n_fft: int = 1024,
        hop_length: int = 512,
        n_mels: int = 64,
        freq_mask_param: int = 12,
        time_mask_param: int = 20,
        enable_specaugment: bool = True,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.target_samples = int(sample_rate * clip_length_s)
        self.enable_specaugment = enable_specaugment

        # Conversione in spettrogramma Log-Mel
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
        self.amplitude_to_db = T.AmplitudeToDB(top_db=80.0)

        self.dynamic_compression = RandomDynamicCompression(p=0.5)
        self.saturation = RandomSaturation(p=0.4)
        self.spectral_filter = RandomSpectralFilter(sample_rate=sample_rate, p=0.5)

        # SpecAugment
        self.freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_mask = T.TimeMasking(time_mask_param=time_mask_param)

    def spec_from_audio(self, waveform: torch.Tensor, apply_specaugment: bool = False) -> torch.Tensor:
        """Pipeline standard deterministica per validazione e inferenza."""
        mel = self.mel_transform(waveform)
        log_mel = self.amplitude_to_db(mel)

        # Standardizzazione z-score
        mean = log_mel.mean()
        std = log_mel.std().clamp(min=1e-3)
        norm_mel = (log_mel - mean) / std

        if apply_specaugment and self.enable_specaugment:
            norm_mel = self.freq_mask(norm_mel)
            norm_mel = self.time_mask(norm_mel)

        return norm_mel

    def get_contrastive_pair(self, waveform_chunk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Genera due viste acustiche alterate (x_i, x_j) dello stesso chunk audio."""
        # Vista 1: compressione dinamica + eventuale mascheramento
        v1 = self.dynamic_compression(waveform_chunk.clone())
        v1 = self.spectral_filter(v1)
        spec1 = self.spec_from_audio(v1, apply_specaugment=True)

        # Vista 2: saturazione armonica + alterazione tonale
        v2 = self.saturation(waveform_chunk.clone())
        v2 = self.spectral_filter(v2)
        spec2 = self.spec_from_audio(v2, apply_specaugment=True)

        return spec1, spec2

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

    def forward_single(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Chiamato da GTZANDownstreamDataset[idx] in train mode.
        Estrae un crop casuale e ne fa lo spettrogramma pulito (senza distorsioni).
        """
        chunk = self._random_crop(waveform)
        return self.spec_from_audio(chunk, apply_specaugment=False)

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Chiamato da GTZANCLMRDataset[idx].
        Genera le due viste contrastive (view1, view2) con alterazioni indipendenti.
        """
        # Vista 1: crop + compressione dinamica + filtro + SpecAugment
        chunk1 = self._random_crop(waveform)
        v1 = self.dynamic_compression(chunk1)
        v1 = self.spectral_filter(v1)
        spec1 = self.spec_from_audio(v1, apply_specaugment=True)

        # Vista 2: crop diverso + saturazione + filtro + SpecAugment
        chunk2 = self._random_crop(waveform)
        v2 = self.saturation(chunk2)
        v2 = self.spectral_filter(v2)
        spec2 = self.spec_from_audio(v2, apply_specaugment=True)

        return spec1, spec2


if __name__ == "__main__":
    # Test diagnostico rapido
    print("Testing AudioDataAugmentation...")
    dummy_waveform = torch.randn(1, 22050 * 5)  # 5 secondi di audio fittizio
    augmenter = AudioDataAugmentation()
    v1, v2 = augmenter(dummy_waveform)
    print("Test completato!")
    print(f"Shape Vista 1: {v1.shape}")
    print(f"Shape Vista 2: {v2.shape}")