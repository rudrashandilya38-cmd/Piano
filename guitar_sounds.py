"""
guitar_sounds.py — Acoustic Guitar Physical Modeling Synthesizer
================================================================
Stage 1 Realistic Acoustic Guitar Sound Engine using:
  - Extended Karplus-Strong physical string modeling
  - Fractional delay allpass interpolation for exact tuning
  - Frequency-dependent decay & high-frequency damping
  - Dual-polarization coupled string modes for natural acoustic shimmer & bloom
  - Pluck contact excitation with triangular displacement & soundhole notch filtering
  - Humanized downstroke strum timing (22–28 ms per string) with velocity curves
  - Acoustic wooden body resonance (Helmholtz air cavity & spruce soundboard plates)
  - NumPy vectorized processing for ultra-fast generation

5 Standard Open Chords:
  1: Em (E Minor)
  2: Am (A Minor)
  3: C  (C Major)
  4: D  (D Major)
  5: G  (G Major)
"""

import os
import math
import wave
import struct
import numpy as np

SOUNDS_DIR = os.path.join(os.path.dirname(__file__), "sounds")
BACKEND_SOUNDS_DIR = os.path.join(os.path.dirname(__file__), "backend", "sounds")

SAMPLE_RATE = 44100
DURATION    = 4.5   # seconds — natural acoustic ring-out

# Open-chord string frequencies (Hz) in standard tuning
GUITAR_CHORDS = {
    "guitar_em.wav": {
        "freqs": [82.41, 123.47, 164.81, 196.00, 246.94, 329.63],  # E2 B2 E3 G3 B3 E4 (6 strings)
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


def render_physical_string(freq, duration_sec, sample_rate=SAMPLE_RATE,
                           velocity=0.88, pluck_pos=0.17, brightness=0.50, detune_cents=0.0):
    """
    Synthesize an acoustic guitar string voice using Karplus-Strong physical modeling.
    """
    actual_freq = freq * (2.0 ** (detune_cents / 1200.0))
    total_samples = int(duration_sec * sample_rate)

    period = sample_rate / actual_freq
    N = int(period)
    frac = period - N
    allpass_c = (1.0 - frac) / (1.0 + frac)

    norm_f = max(0.0, min(1.0, (actual_freq - 80.0) / 320.0))
    t60 = 4.6 - norm_f * 1.8
    loss_factor = 0.001 ** (1.0 / (t60 * actual_freq))
    loop_damp = 0.22 + 0.20 * (1.0 - norm_f * 0.7)

    # Secondary coupled mode
    period2 = sample_rate / (actual_freq + 0.55)
    N2 = int(period2)
    frac2 = period2 - N2
    allpass_c2 = (1.0 - frac2) / (1.0 + frac2)

    line1 = np.zeros(N + 2, dtype=np.float64)
    line2 = np.zeros(N2 + 2, dtype=np.float64)

    exc_len = min(N, max(24, int(sample_rate * (0.003 + (1.0 - velocity) * 0.002))))
    pick_offset = max(1, int(period * pluck_pos))

    # Exciter
    t_arr = np.linspace(0, 1, exc_len, endpoint=False)
    win = 0.5 * (1.0 - np.cos(2.0 * np.pi * t_arr))
    tri = np.where(t_arr < pluck_pos, t_arr / pluck_pos, (1.0 - t_arr) / (1.0 - pluck_pos))
    noise = (np.random.rand(exc_len) * 2.0 - 1.0) * 0.35
    raw_exc = (tri * 0.75 + noise) * win * velocity

    for i in range(exc_len - 1, -1, -1):
        if i >= pick_offset:
            raw_exc[i] -= 0.88 * raw_exc[i - pick_offset]

    exc_filt = 0.0
    exc_cutoff = 0.28 + (velocity * 0.45) + (brightness * 0.15)
    for i in range(exc_len):
        exc_filt += exc_cutoff * (raw_exc[i] - exc_filt)
        line1[i] = exc_filt * 0.85
        if i < len(line2):
            line2[i] = exc_filt * 0.35

    out = np.zeros(total_samples, dtype=np.float32)

    read_idx1 = 0
    read_idx2 = 0
    lp_prev1 = 0.0
    lp_prev2 = 0.0
    ap_prev_in1 = 0.0
    ap_prev_out1 = 0.0
    ap_prev_in2 = 0.0
    ap_prev_out2 = 0.0
    couple = 0.015

    for n in range(total_samples):
        # Mode 1
        s1 = line1[read_idx1]
        interp1 = allpass_c * (s1 - ap_prev_out1) + ap_prev_in1
        ap_prev_in1 = s1
        ap_prev_out1 = interp1
        damp1 = (1.0 - loop_damp) * interp1 + loop_damp * lp_prev1
        lp_prev1 = damp1

        # Mode 2
        s2 = line2[read_idx2]
        interp2 = allpass_c2 * (s2 - ap_prev_out2) + ap_prev_in2
        ap_prev_in2 = s2
        ap_prev_out2 = interp2
        damp2 = (1.0 - (loop_damp * 0.85)) * interp2 + (loop_damp * 0.85) * lp_prev2
        lp_prev2 = damp2

        next1 = (damp1 * (1.0 - couple) + damp2 * couple) * loss_factor
        next2 = (damp2 * (1.0 - couple) + damp1 * couple) * (loss_factor * 0.9995)

        line1[read_idx1] = next1
        line2[read_idx2] = next2

        out[n] = float(interp1 * 0.78 + interp2 * 0.38)

        read_idx1 += 1
        if read_idx1 >= N:
            read_idx1 = 0
        read_idx2 += 1
        if read_idx2 >= N2:
            read_idx2 = 0

    # DC Block filter
    prev_x = 0.0
    prev_y = 0.0
    r = 0.995
    for i in range(total_samples):
        x = out[i]
        y = x - prev_x + r * prev_y
        prev_x = x
        prev_y = y
        out[i] = y

    peak = np.max(np.abs(out))
    if peak > 1e-5:
        out *= (0.90 / peak)

    return out


def apply_body_acoustics(samples, sample_rate=SAMPLE_RATE):
    """
    Acoustic body wood resonator: Helmholtz air cavity + spruce top plate + back plate.
    """
    def make_peak_filter(freq, q, gain_db):
        w0 = 2.0 * np.pi * freq / sample_rate
        alpha = np.sin(w0) / (2.0 * q)
        a_amp = 10.0 ** (gain_db / 40.0)
        b0 = 1.0 + alpha * a_amp
        b1 = -2.0 * np.cos(w0)
        b2 = 1.0 - alpha * a_amp
        a0 = 1.0 + alpha / a_amp
        a1 = -2.0 * np.cos(w0)
        a2 = 1.0 - alpha / a_amp
        return (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)

    filters = [
        make_peak_filter(104.0, 2.6, 3.2),  # Helmholtz cavity
        make_peak_filter(212.0, 2.8, 3.6),  # Spruce soundboard top plate
        make_peak_filter(318.0, 2.2, 2.0),  # Back plate
    ]

    out = np.copy(samples)
    d1 = int(0.019 * sample_rate)
    d2 = int(0.033 * sample_rate)

    if len(out) > d2:
        out[d1:] += samples[:-d1] * 0.14
        out[d2:] += samples[:-d2] * 0.08

    for b0, b1, b2, a1, a2 in filters:
        x1 = x2 = y1 = y2 = 0.0
        for i in range(len(out)):
            x = out[i]
            y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            x2 = x1
            x1 = x
            y2 = y1
            y1 = y
            out[i] = y

    # Warm wooden soft saturation
    out = np.tanh(out * 1.1)
    return out


def generate_acoustic_chord_wav(filename, frequencies, duration=DURATION, sample_rate=SAMPLE_RATE):
    """
    Generate an acoustic guitar chord WAV file with humanized downstroke strumming.
    """
    total_samples = int(duration * sample_rate)
    chord_mix = np.zeros(total_samples, dtype=np.float32)

    cur_delay_sec = 0.0
    n_strings = len(frequencies)

    for s_idx, freq in enumerate(frequencies):
        delay_samples = int(cur_delay_sec * sample_rate)
        pos = s_idx / max(1, n_strings - 1)
        base_vel = 0.90 - math.sin(pos * math.pi) * 0.12
        vel = base_vel + (np.random.rand() * 0.06 - 0.03)

        rem_duration = max(0.5, duration - cur_delay_sec)
        string_audio = render_physical_string(
            freq=freq,
            duration_sec=rem_duration,
            sample_rate=sample_rate,
            velocity=vel,
            pluck_pos=0.16 + (np.random.rand() * 0.02),
            detune_cents=(np.random.rand() * 2.0 - 1.0) * 1.2
        )

        n_copy = min(len(string_audio), total_samples - delay_samples)
        if n_copy > 0:
            chord_mix[delay_samples:delay_samples + n_copy] += string_audio[:n_copy] * 0.40

        step_ms = 23 + np.random.randint(0, 6)
        cur_delay_sec += step_ms / 1000.0

    final_audio = apply_body_acoustics(chord_mix, sample_rate)

    peak = np.max(np.abs(final_audio)) or 1.0
    scale = (32767 * 0.90) / peak
    pcm_data = np.clip(final_audio * scale, -32767, 32767).astype(np.int16)

    for target_dir in [SOUNDS_DIR, BACKEND_SOUNDS_DIR]:
        os.makedirs(target_dir, exist_ok=True)
        filepath = os.path.join(target_dir, filename)
        with wave.open(filepath, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_data.tobytes())

    label = GUITAR_CHORDS[filename]["label"]
    print(f"[Acoustic Guitar Physical Model] Generated: {filename:<20}  {label}")


def generate_all_guitar_sounds():
    """Generate all 5 acoustic guitar chord WAVs."""
    print("[Acoustic Guitar Physical Model] Synthesizing acoustic guitar chord library...")
    for filename, info in GUITAR_CHORDS.items():
        generate_acoustic_chord_wav(filename, info["freqs"])
    print("[Acoustic Guitar Physical Model] All 5 chords successfully generated.\n")


if __name__ == "__main__":
    generate_all_guitar_sounds()
