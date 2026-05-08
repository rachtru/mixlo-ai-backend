"""
Mixlo AI — Stem classification + mixing engine
LibROSA handles analysis; Pedalboard handles processing.
"""

import numpy as np
import librosa
from pedalboard import (
    Pedalboard, Compressor, HighpassFilter, LowpassFilter,
    HighShelfFilter, LowShelfFilter, Gain, PeakFilter,
)


# ── Stem classification ────────────────────────────────────────────────────────

LABEL_KEYWORDS = {
    "kick":    ["kick", "bd", "bassdrum", "bass_drum"],
    "snare":   ["snare", "sd", "clap", "rimshot"],
    "hihat":   ["hihat", "hi_hat", "hh", "cymbal", "overhead", "oh"],
    "drum":    ["drum", "perc", "percussion", "loop", "beat"],
    "bass":    ["bass", "sub", "808"],
    "lead":    ["lead", "melody", "synth", "arp"],
    "pad":     ["pad", "strings", "choir", "atmos", "ambient", "texture"],
    "vocal":   ["vocal", "vox", "voice", "bv", "bg_vocal", "adlib"],
    "guitar":  ["guitar", "gtr", "acoustic"],
    "piano":   ["piano", "keys", "organ", "rhodes", "wurli"],
    "fx":      ["fx", "riser", "impact", "sweep", "noise", "sfx"],
}


def _label_from_filename(filename: str) -> str | None:
    name = filename.lower().replace("-", "_").replace(" ", "_")
    for label, keywords in LABEL_KEYWORDS.items():
        if any(kw in name for kw in keywords):
            return label
    return None


def _spectral_label(audio: np.ndarray, sr: int) -> tuple[str, float]:
    """
    Fallback classifier using spectral features when filename is ambiguous.
    Returns (label, confidence 0-1).
    """
    centroid  = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
    rolloff   = float(np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.85)))
    rms       = float(np.mean(librosa.feature.rms(y=audio)))
    zcr       = float(np.mean(librosa.feature.zero_crossing_rate(audio)))

    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    tempo, _  = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    tempo     = float(tempo) if np.isscalar(tempo) else float(tempo[0])

    # Very low centroid + strong low-end → bass or kick
    if centroid < 300 and rms > 0.05:
        return ("kick", 0.7) if zcr > 0.05 else ("bass", 0.7)

    # Low centroid, sustained → bass
    if centroid < 500:
        return "bass", 0.65

    # High centroid + high ZCR → hihat / percussion
    if centroid > 4000 and zcr > 0.15:
        return "hihat", 0.7

    # Mid centroid + percussive → snare / drum
    if 500 < centroid < 3000 and zcr > 0.08:
        return "snare" if centroid > 1500 else "drum", 0.6

    # Low ZCR + mid centroid + sustained → pad
    if zcr < 0.04 and centroid < 2500:
        return "pad", 0.6

    # High centroid + low ZCR → lead / melodic
    if centroid > 1500 and zcr < 0.06:
        return "lead", 0.55

    return "lead", 0.4  # catch-all


def classify_stem(audio: np.ndarray, sr: int, filename: str) -> tuple[str, float]:
    label = _label_from_filename(filename)
    if label:
        return label, 1.0
    return _spectral_label(audio, sr)


# ── Per-stem signal chains ─────────────────────────────────────────────────────

def _chain_kick() -> Pedalboard:
    return Pedalboard([
        HighpassFilter(cutoff_frequency_hz=30),
        LowpassFilter(cutoff_frequency_hz=12000),
        Compressor(threshold_db=-18, ratio=4, attack_ms=5, release_ms=60),
        Gain(gain_db=0),
    ])

def _chain_snare() -> Pedalboard:
    return Pedalboard([
        HighpassFilter(cutoff_frequency_hz=80),
        PeakFilter(cutoff_frequency_hz=200, gain_db=-3, q=1.0),   # remove mud
        PeakFilter(cutoff_frequency_hz=3000, gain_db=2, q=1.5),   # snap
        Compressor(threshold_db=-16, ratio=4, attack_ms=3, release_ms=80),
    ])

def _chain_hihat() -> Pedalboard:
    return Pedalboard([
        HighpassFilter(cutoff_frequency_hz=6000),
        Compressor(threshold_db=-20, ratio=3, attack_ms=2, release_ms=40),
        Gain(gain_db=-2),
    ])

def _chain_drum() -> Pedalboard:
    return Pedalboard([
        HighpassFilter(cutoff_frequency_hz=50),
        Compressor(threshold_db=-18, ratio=3, attack_ms=8, release_ms=80),
    ])

def _chain_bass() -> Pedalboard:
    return Pedalboard([
        HighpassFilter(cutoff_frequency_hz=30),
        LowpassFilter(cutoff_frequency_hz=8000),
        PeakFilter(cutoff_frequency_hz=80, gain_db=2, q=1.2),     # weight
        PeakFilter(cutoff_frequency_hz=300, gain_db=-3, q=1.0),   # mud cut
        Compressor(threshold_db=-14, ratio=5, attack_ms=10, release_ms=100),
    ])

def _chain_lead() -> Pedalboard:
    return Pedalboard([
        HighpassFilter(cutoff_frequency_hz=120),
        PeakFilter(cutoff_frequency_hz=250, gain_db=-2, q=1.0),
        HighShelfFilter(cutoff_frequency_hz=8000, gain_db=1.5),
        Compressor(threshold_db=-18, ratio=3, attack_ms=10, release_ms=120),
    ])

def _chain_pad() -> Pedalboard:
    return Pedalboard([
        HighpassFilter(cutoff_frequency_hz=200),
        LowpassFilter(cutoff_frequency_hz=14000),
        Compressor(threshold_db=-22, ratio=2, attack_ms=30, release_ms=200),
        Gain(gain_db=-2),
    ])

def _chain_vocal() -> Pedalboard:
    return Pedalboard([
        HighpassFilter(cutoff_frequency_hz=100),
        PeakFilter(cutoff_frequency_hz=300, gain_db=-3, q=1.2),   # chest mud
        PeakFilter(cutoff_frequency_hz=3000, gain_db=2, q=1.5),   # presence
        HighShelfFilter(cutoff_frequency_hz=10000, gain_db=2),
        Compressor(threshold_db=-16, ratio=4, attack_ms=5, release_ms=80),
    ])

def _chain_guitar() -> Pedalboard:
    return Pedalboard([
        HighpassFilter(cutoff_frequency_hz=80),
        PeakFilter(cutoff_frequency_hz=400, gain_db=-2, q=1.0),
        Compressor(threshold_db=-18, ratio=3, attack_ms=10, release_ms=100),
    ])

def _chain_piano() -> Pedalboard:
    return Pedalboard([
        HighpassFilter(cutoff_frequency_hz=60),
        Compressor(threshold_db=-20, ratio=2, attack_ms=15, release_ms=150),
    ])

def _chain_fx() -> Pedalboard:
    return Pedalboard([
        Compressor(threshold_db=-24, ratio=2, attack_ms=20, release_ms=200),
        Gain(gain_db=-4),
    ])

CHAINS = {
    "kick":   _chain_kick,
    "snare":  _chain_snare,
    "hihat":  _chain_hihat,
    "drum":   _chain_drum,
    "bass":   _chain_bass,
    "lead":   _chain_lead,
    "pad":    _chain_pad,
    "vocal":  _chain_vocal,
    "guitar": _chain_guitar,
    "piano":  _chain_piano,
    "fx":     _chain_fx,
}

# Fader levels per stem type (linear gain applied after processing)
FADER_DB = {
    "kick":   0,
    "snare":  -1,
    "hihat":  -6,
    "drum":   -2,
    "bass":   -2,
    "lead":   -3,
    "pad":    -8,
    "vocal":   0,
    "guitar": -4,
    "piano":  -5,
    "fx":     -10,
}


def _db_to_lin(db: float) -> float:
    return 10 ** (db / 20)


# ── Mix engine ─────────────────────────────────────────────────────────────────

def mix_stems(stems: list[dict]) -> tuple[np.ndarray, int]:
    """
    Process each stem through its signal chain and sum to a stereo bus.
    Returns (stereo_array [2, N], sample_rate).
    """
    sr = stems[0]["sr"]

    # Align lengths — pad shorter stems with silence
    max_len = max(s["audio"].shape[-1] for s in stems)

    processed = []
    for stem in stems:
        audio  = stem["audio"]   # shape: [2, N]
        label  = stem["label"]
        chain  = CHAINS.get(label, CHAINS["lead"])()
        fader  = _db_to_lin(FADER_DB.get(label, -3))

        # Process each channel through Pedalboard
        ch_l = chain(audio[0:1], sr)[0]
        ch_r = chain(audio[1:2], sr)[0]

        stereo = np.stack([ch_l, ch_r]) * fader

        # Pad to max length
        pad = max_len - stereo.shape[1]
        if pad > 0:
            stereo = np.pad(stereo, ((0, 0), (0, pad)))

        processed.append(stereo)

    # Sum all stems
    bus = np.sum(processed, axis=0)

    # Soft-limit bus to prevent clipping (simple peak normalize to -1 dBFS)
    peak = np.max(np.abs(bus))
    if peak > 0:
        target = _db_to_lin(-1)
        bus = bus * (target / peak)

    return bus, sr
