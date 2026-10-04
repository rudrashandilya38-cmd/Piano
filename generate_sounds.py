import os
import math
import struct
import wave

SOUNDS_DIR = os.path.join(os.path.dirname(__file__), "sounds")

# Chord definitions with note frequencies in Hz for Indian Scale (Sa Re Ga Ma Pa Dha Ni Sa)
CHORDS = {
    "c_major.wav":      [261.63, 329.63, 392.00, 523.25],   # Sa:  C4, E4, G4, C5 (C Major)
    "d_minor.wav":      [293.66, 349.23, 440.00, 587.33],   # Re:  D4, F4, A4, D5 (D Minor)
    "e_minor.wav":      [329.63, 392.00, 493.88, 659.25],   # Ga:  E4, G4, B4, E5 (E Minor)
    "f_major.wav":      [349.23, 440.00, 523.25, 698.46],   # Ma:  F4, A4, C5, F5 (F Major)
    "g_major.wav":      [392.00, 493.88, 587.33, 783.99],   # Pa:  G4, B4, D5, G5 (G Major)
    "a_minor.wav":      [440.00, 523.25, 659.25, 880.00],   # Dha: A4, C5, E5, A5 (A Minor)
    "b_dim.wav":        [493.88, 587.33, 698.46, 987.77],   # Ni:  B4, D5, F5, B5 (B Diminished)
    "c_major_high.wav": [523.25, 659.25, 783.99, 1046.50],  # Sa': C5, E5, G5, C6 (C Major High)
}


def generate_chord_wav(filename, frequencies, duration=2.0, sample_rate=44100):
    """Synthesizes a warm harmonium chord — sustained reed organ tone with slow
    attack, rich odd harmonics, multi-reed chorus detuning, and bellows tremolo."""
    num_samples = int(duration * sample_rate)
    samples = []

    # Harmonium reed: strong odd harmonics (h1..h6), slight detuning across reeds
    reed_offsets  = [-0.8, 0.0, 0.8]   # cents offsets for multi-reed chorus
    harmonic_amps = [1.00, 0.55, 0.30, 0.18, 0.10, 0.06]  # h1..h6

    # Tremolo (bellows vibrato) at ~6 Hz, depth ~1.5%
    tremolo_rate  = 6.0
    tremolo_depth = 0.015

    attack_time  = 0.08   # 80 ms slow reed attack
    release_time = 0.15   # 150 ms soft tail

    for i in range(num_samples):
        t = i / sample_rate

        # Amplitude envelope: slow attack, sustain, gentle release
        if t < attack_time:
            env = t / attack_time
        elif t > (duration - release_time):
            env = (duration - t) / release_time
        else:
            env = 1.0
        env = env ** 1.5  # smooth the curve

        # Tremolo (bellows simulation)
        tremolo = 1.0 + tremolo_depth * math.sin(2 * math.pi * tremolo_rate * t)

        sample_val = 0.0
        for freq in frequencies:
            reed_sum = 0.0
            for offset_cents in reed_offsets:
                f = freq * (2 ** (offset_cents / 1200.0))
                for h, amp in enumerate(harmonic_amps, start=1):
                    reed_sum += amp * math.sin(2 * math.pi * (f * h) * t)
            sample_val += reed_sum / len(reed_offsets)

        # Normalize by note count, apply envelope and tremolo
        sample_val = (sample_val / len(frequencies)) * env * tremolo

        # Convert to 16-bit signed integer PCM
        int_val = int(sample_val * 18000.0)
        int_val = max(-32767, min(32767, int_val))
        samples.append(int_val)

    os.makedirs(SOUNDS_DIR, exist_ok=True)
    filepath = os.path.join(SOUNDS_DIR, filename)

    with wave.open(filepath, 'w') as wav_file:
        wav_file.setnchannels(1)   # Mono
        wav_file.setsampwidth(2)   # 16-bit
        wav_file.setframerate(sample_rate)

        packed_bytes = bytearray()
        for sample in samples:
            packed_bytes.extend(struct.pack('<h', sample))
        wav_file.writeframes(packed_bytes)

    print(f"[Sound Synth] Generated harmonium chord: {filename}")


def generate_all_sounds():
    """Generates all harmonium chord sound files in the sounds/ directory."""
    os.makedirs(SOUNDS_DIR, exist_ok=True)
    for filename, freqs in CHORDS.items():
        # Always regenerate so the new harmonium timbre is applied
        generate_chord_wav(filename, freqs)


if __name__ == "__main__":
    generate_all_sounds()
