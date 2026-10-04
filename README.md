# 🎸 Gesture Guitar

**Gesture Guitar** is a real-time Python application that lets you play authentic open-string guitar chords using only your hand gestures captured by a webcam.  Powered by **MediaPipe**, **OpenCV**, **Pygame**, and optionally **SpeechRecognition**.

---

## 🎯 Chord Mapping (5 Fingers)

| Fingers | Chord | Full Name | Keyboard | Voice Command |
| :--- | :--- | :--- | :--- | :--- |
| ✊ Fist (0) | — | **Mute** | — | `"stop"`, `"mute"`, `"silence"` |
| ☝ 1 Finger | **Em** | E Minor | `1` or `E` | `"e minor"`, `"em"`, `"one"` |
| ✌ 2 Fingers | **Am** | A Minor | `2` or `A` | `"a minor"`, `"am"`, `"two"` |
| 🤟 3 Fingers | **C**  | C Major  | `3` or `C` | `"c major"`, `"c"`, `"three"` |
| 🖖 4 Fingers | **D**  | D Major  | `4` or `D` | `"d major"`, `"d"`, `"four"` |
| 🖐 5 Fingers | **G**  | G Major  | `5` or `G` | `"g major"`, `"g"`, `"five"` |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run

```bash
python main.py
```

> **Note**: Guitar chord sound files are automatically synthesized into `sounds/` on first launch.

---

## 🎮 How to Play

1. **Webcam Gesture**: Hold one hand in front of the camera.
   - Raise **1 to 5 fingers** to trigger the corresponding chord.
   - Make a **fist** to mute.
2. **Keyboard**: Press keys `1`–`5` or `E`, `A`, `C`, `D`, `G`.
3. **Voice**: Speak chord names into your microphone (`"E minor"`, `"G"`, etc.).
   - Press **`V`** to toggle voice listening on/off.
4. **Quit**: Press **`Q`**.

---

## 📁 Project Structure

```
Guitar/
├── main.py              # Webcam gesture loop, HUD, voice handler
├── guitar_sounds.py     # Plucked-string synthesizer for 5 guitar chords
├── generate_sounds.py   # (Legacy harmonium synthesizer — kept for reference)
├── requirements.txt     # Dependencies
├── README.md            # This file
└── sounds/
    ├── guitar_em.wav    (E Minor)
    ├── guitar_am.wav    (A Minor)
    ├── guitar_c.wav     (C Major)
    ├── guitar_d.wav     (D Major)
    └── guitar_g.wav     (G Major)
```

---

## 🛠️ Features

- **Smart 3D Orientation-Invariant Gesture Engine** — Transforms 3D hand landmarks into a local hand-centric orthonormal coordinate system. Works flawlessly regardless of hand tilt, angle, distance, roll/pitch/yaw, or left vs. right hand usage.
- **Multi-Criterion Finger Scoring & Hysteresis** — Combines local Y-extension ratios, 3D Euclidean distances, and PIP joint flex angles with per-finger temporal debouncing for flicker-free, instant chord triggers.
- **Real-Time HUD Finger Status Indicators** — Header display features `[T] [I] [M] [R] [P]` status pills showing live feedback for Thumb, Index, Middle, Ring, and Pinky detection alongside fingertip glow nodes.
- **Karplus-Strong Inspired Synthesis** — Guitar chords synthesized with per-string exponential decay, pick transient, strum delay, and chorus detuning for a realistic plucked-string sound.
- **Animated Guitar HUD** — Includes a fretboard sidebar with real-time chord dot highlighting, an animated strum wave, chord reference pills, and voice/status badges.
- **Voice Command Support** — Background SpeechRecognition thread hears chord names without impacting frame rate.
- **Keyboard Fallback** — Full keyboard control for accessibility or demos without a camera.

