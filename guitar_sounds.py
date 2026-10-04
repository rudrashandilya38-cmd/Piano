"""
guitar_sounds.py — Classical (Nylon-String) Guitar Chord Synthesizer
=====================================================================
Produces warm, finger-plucked nylon-string classical guitar tones:
  - Soft fingertip attack envelope (no pick transient)
  - Rounded harmonic series (heavy fundamental, fast rolloff on highs)
  - Per-string Karplus–Strong style exponential decay
    (nylon strings sustain slightly longer in the midrange)
  - Slow strum spread (~60 ms total) simulating an arpeggio strum
  - Subtle body resonance via low-pass softening (tanh warm clip)
  - Gentle chorus detuning for natural string irregularity

5 Standard Open Chords:
  Em  A m  C  D  G
"""

import os
import math
import struct
import wave

SOUNDS_DIR = os.path.join(os.path.dirname(__file__), "sounds")

SAMPLE_RATE = 44100
DURATION    = 5.0   # seconds — full natural ring-out

# ── Open-chord string frequencies (Hz) in standard tuning ────────────────────
# E2=82.41  A2=110.00  D3=146.83  G3=196.00  B3=246.94  E4=329.63
GUITAR_CHORDS = {
    "guitar_em.wav": {
        "freqs": [82.41, 123.47, 164.81, 196.00, 246.94, 329.63],  # E2 B2 E3 G3 B3 E4 (all 6 strings)
        "label": "E Minor (open Em)",
    },
    "guitar_am.wav": {
        "freqs": [110.00, 164.81, 220.00, 261.63, 329.63],          # A2 E3 A3 C4 E4 (5 strings)
        "label": "A Minor (open Am)",
    },
    "guitar_c.wav": {
        "freqs": [130.81, 164.81, 196.00, 261.63, 329.63],          # C3 E3 G3 C4 E4 (5 strings)
        "label": "C Major (open C)",
    },
    "guitar_d.wav": {
        "freqs": [146.83, 220.00, 293.66, 369.99],                   # D3 A3 D4 F#4 (4 strings)
        "label": "D Major (open D)",
    },
    "guitar_g.wav": {
        "freqs": [98.00, 123.47, 196.00, 246.94, 329.63, 392.00],   # G2 B2 G3 B3 E4 G4 (6 strings)
        "label": "G Major (open G)",
    },
}


def _nylon_decay(freq):
    """
    Decay rate (1/s) for a nylon classical guitar string.
    Slowed right down so the chord rings for several seconds,
    mimicking a classical guitar allowed to sustain freely.
      Bass E2  → ~0.25 /s  (≈4 s half-life)
      Treble E4 → ~0.70 /s  (≈1.4 s half-life)
    """
    lo, hi = 80.0, 400.0
    t = max(0.0, min(1.0, (freq - lo) / (hi - lo)))
    return 0.25 + t * 0.45  # 0.25 (bass) → 0.70 (treble)


def _nylon_harmonics(t, freq, decay_rate):
    """
    Single nylon string tone.

    Harmonic series for nylon/gut strings:
      h1: 1.00  (very strong fundamental — warm, full body)
      h2: 0.38  (moderate second harmonic)
      h3: 0.16  (third — classical guitar has less brightness here)
      h4: 0.07
      h5: 0.03
    High harmonics roll off fast → the characteristic mellow warmth.

    Three-voice chorus: -2, 0, +2 cents — subtle intonation variation
    (a real player's fingers never place exactly the same pressure twice).
    """
    env = math.exp(-decay_rate * t)

    # Classical guitar harmonic envelope
    harmonic_amps = [1.00, 0.38, 0.16, 0.07, 0.03]

    detune_cents = [-2.0, 0.0, 2.0]
    detune_gains = [0.25, 1.00, 0.25]
    total_gain   = sum(detune_gains)

    val = 0.0
    for dc, dg in zip(detune_cents, detune_gains):
        f0 = freq * (2.0 ** (dc / 1200.0))
        for h_idx, amp in enumerate(harmonic_amps, start=1):
            val += dg * amp * math.sin(2.0 * math.pi * f0 * h_idx * t)

    return (val / total_gain) * env


def _finger_attack_env(t, freq):
    """
    Soft fingertip attack: no sharp click, gentle rise over ~30 ms.
    Uses a raised-cosine (Hann) ramp for a perfectly smooth onset.
    """
    attack_ms = 0.030   # 30 ms soft rise — slightly longer for extra smoothness
    if t < attack_ms:
        return 0.5 * (1.0 - math.cos(math.pi * t / attack_ms))
    return 1.0


def generate_classical_chord_wav(filename, frequencies,
                                  duration=DURATION, sample_rate=SAMPLE_RATE):
    """
    Synthesize a finger-strummed classical guitar chord WAV.

    Strum model:
      - Strings triggered in order low→high, spread across 55 ms total.
      - Each string uses its own onset time and per-frequency decay.
      - Post-processing: comb-reverb adds body resonance warmth.
    """
    num_samples  = int(duration * sample_rate)
    n_strings    = len(frequencies)

    # Total strum spread: 55 ms from lowest to highest string
    strum_span   = 0.055
    strum_delay  = strum_span / max(n_strings - 1, 1)

    # Build raw float samples
    raw = [0.0] * num_samples
    for i in range(num_samples):
        t_global = i / sample_rate
        val      = 0.0
        n_active = 0

        for s_idx, freq in enumerate(frequencies):
            t_onset = s_idx * strum_delay
            if t_global < t_onset:
                continue
            t = t_global - t_onset
            n_active += 1

            decay  = _nylon_decay(freq)
            tone   = _nylon_harmonics(t, freq, decay)
            attack = _finger_attack_env(t, freq)
            val   += tone * attack

        if n_active > 0:
            val /= n_active

        # Gentle warm clip (stays in linear range for quiet signals)
        raw[i] = math.tanh(val * 0.65)

    # ── Comb-reverb post-processing ───────────────────────────────────
    # Three short delay lines simulate guitar body cavity reflections.
    # Adds warmth and removes the "dry" synthetic feeling without a
    # harsh echo effect (delays are short and heavily attenuated).
    reverb_taps = [
        (int(0.021 * sample_rate), 0.22),   # 21 ms  body top-plate reflection
        (int(0.034 * sample_rate), 0.14),   # 34 ms  back-plate reflection
        (int(0.047 * sample_rate), 0.08),   # 47 ms  side-wall diffusion
    ]
    for delay_samples, gain in reverb_taps:
        for i in range(delay_samples, num_samples):
            raw[i] += raw[i - delay_samples] * gain

    # Find actual peak and normalise to a safe level (no clipping)
    peak = max(abs(v) for v in raw) or 1.0
    # Target peak at 80% of int16 range for headroom
    scale = (32767 * 0.80) / peak

    samples = [max(-32767, min(32767, int(v * scale))) for v in raw]

    os.makedirs(SOUNDS_DIR, exist_ok=True)
    filepath = os.path.join(SOUNDS_DIR, filename)

    with wave.open(filepath, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        packed = bytearray()
        for s in samples:
            packed.extend(struct.pack("<h", s))
        wf.writeframes(packed)

    label = GUITAR_CHORDS[filename]["label"]
    print(f"[Classical Guitar] Generated: {filename:<22}  {label}")


def generate_all_guitar_sounds():
    """Generate all 5 classical guitar chord WAVs into the sounds/ directory."""
    os.makedirs(SOUNDS_DIR, exist_ok=True)
    for filename, info in GUITAR_CHORDS.items():
        generate_classical_chord_wav(filename, info["freqs"])
    print("[Classical Guitar] All 5 chord samples ready.\n")


if __name__ == "__main__":
    generate_all_guitar_sounds()
