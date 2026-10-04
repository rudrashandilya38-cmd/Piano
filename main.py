"""
main.py — Gesture-Responsive Guitar
====================================
Play 5 open guitar chords in real-time using hand gestures detected via
your webcam.  Powered by MediaPipe Hands, OpenCV, and Pygame.

Gesture Map (single hand, 0-5 fingers):
  Fist  (0) → Mute
  1 Finger  → Em  (E Minor)
  2 Fingers → Am  (A Minor)
  3 Fingers → C   (C Major)
  4 Fingers → D   (D Major)
  5 Fingers → G   (G Major)

Keyboard shortcuts:  1-5 / E,A,C,D,G → trigger chord  |  Q → quit
Voice commands:      "e minor", "a minor", "c", "d", "g", "stop"
"""

import sys
import os
import math
import time
import urllib.request
import threading
from collections import deque

# ── Dependency guard ─────────────────────────────────────────────────────────
try:
    import cv2
    import numpy as np
    import pygame
    import mediapipe as mp
except ImportError as e:
    print("\n" + "=" * 60)
    print(" [ERROR] Missing required Python package!")
    print(f" Details: {e}")
    print(" Please install dependencies using:")
    print("    pip install -r requirements.txt")
    print("=" * 60 + "\n")
    sys.exit(1)

try:
    import speech_recognition as sr
    HAS_SPEECH_REC = True
except ImportError:
    sr = None
    HAS_SPEECH_REC = False

try:
    from guitar_sounds import generate_all_guitar_sounds
except ImportError:
    generate_all_guitar_sounds = None

# ── Chord configuration ───────────────────────────────────────────────────────
# Keys 1-5 = number of raised fingers that trigger each chord
CHORD_CONFIG = {
    1: {"name": "Em",  "full": "E Minor",  "file": os.path.join("sounds", "guitar_em.wav"), "color": (255,  90,  90), "keys": "Em"},
    2: {"name": "Am",  "full": "A Minor",  "file": os.path.join("sounds", "guitar_am.wav"), "color": (255, 165,  60), "keys": "Am"},
    3: {"name": "C",   "full": "C Major",  "file": os.path.join("sounds", "guitar_c.wav"),  "color": ( 80, 220, 120), "keys": "C "},
    4: {"name": "D",   "full": "D Major",  "file": os.path.join("sounds", "guitar_d.wav"),  "color": ( 80, 160, 255), "keys": "D "},
    5: {"name": "G",   "full": "G Major",  "file": os.path.join("sounds", "guitar_g.wav"),  "color": (200,  80, 255), "keys": "G "},
}

# ── Voice command map ─────────────────────────────────────────────────────────
VOICE_CHORD_MAP = {
    # Em
    "e minor": 1, "e": 1, "em": 1, "one": 1, "1": 1,
    # Am
    "a minor": 2, "a": 2, "am": 2, "two": 2, "2": 2,
    # C
    "c major": 3, "c": 3, "three": 3, "3": 3, "see": 3,
    # D
    "d major": 4, "d": 4, "four": 4, "4": 4, "dee": 4,
    # G
    "g major": 5, "g": 5, "five": 5, "5": 5, "gee": 5,
    # Stop
    "stop": 0, "mute": 0, "silence": 0, "fist": 0, "off": 0, "clear": 0,
}

DEBOUNCE_WINDOW_SIZE = 3   # smaller = faster chord response
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "hand_landmarker.task")

# MediaPipe hand skeleton connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

CHORD_CHANNEL_ID = 0   # Pygame mixer channel for chords

# Guitar string colour palette
AMBER   = (255, 180,  50)
ORANGE  = (255, 120,  30)
DARK_BG = ( 18,  20,  28)

# ── Guitar body ASCII fretboard (decorative overlay) ─────────────────────────
FRET_LINES = 4   # number of horizontal fret lines in the sidebar


# ═════════════════════════════════════════════════════════════════════════════
#  Voice command handler
# ═════════════════════════════════════════════════════════════════════════════
class VoiceCommandHandler:
    def __init__(self):
        self.running     = False
        self.enabled     = True
        self.thread      = None
        self.last_text   = ""
        self.last_chord  = None
        self.last_event_time = 0
        self.status      = "Voice: Off"
        self.recognizer  = None
        self.microphone  = None

        if HAS_SPEECH_REC:
            try:
                self.recognizer = sr.Recognizer()
                self.recognizer.energy_threshold      = 300
                self.recognizer.dynamic_energy_threshold = True
                self.recognizer.pause_threshold       = 0.5
                self.microphone = sr.Microphone()
                self.status = "Voice: Ready"
            except Exception as exc:
                self.status = f"Voice: Mic Error ({exc})"
        else:
            self.status = "Voice: SpeechRec Missing"

    def start(self):
        if not HAS_SPEECH_REC or not self.microphone or self.running:
            return
        self.running = True
        self.thread  = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()

    def toggle(self):
        self.enabled = not self.enabled
        if self.enabled:
            if not self.running:
                self.start()
            self.status = "Voice: Enabled"
        else:
            self.status = "Voice: Muted"

    def _listen_loop(self):
        self.status = "Voice: Listening..."
        try:
            with self.microphone as source:
                try:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                except Exception:
                    pass

                while self.running:
                    if not self.enabled:
                        self.status = "Voice: Muted"
                        time.sleep(0.5)
                        continue
                    try:
                        self.status = "Voice: Listening..."
                        audio = self.recognizer.listen(source, timeout=1.2, phrase_time_limit=2.5)
                        self.status = "Voice: Processing..."
                        try:
                            raw_text = self.recognizer.recognize_google(audio).lower().strip()
                            chord_id = self.parse_text(raw_text)
                            if chord_id is not None:
                                self.last_text        = raw_text
                                self.last_chord       = chord_id
                                self.last_event_time  = time.time()
                                label = CHORD_CONFIG[chord_id]["full"] if chord_id in CHORD_CONFIG else "Stop"
                                self.status = f"Voice: '{raw_text}' ({label})"
                                print(f"[Voice] Heard: '{raw_text}' → {label}")
                            else:
                                self.status = f"Voice: '{raw_text}'"
                        except sr.UnknownValueError:
                            self.status = "Voice: Listening..."
                        except sr.RequestError:
                            self.status = "Voice: Offline/Net Err"
                            time.sleep(2)
                    except sr.WaitTimeoutError:
                        self.status = "Voice: Listening..."
                    except Exception:
                        time.sleep(0.3)
        except Exception:
            self.status = "Voice: Mic Unavailable"

    def parse_text(self, text):
        clean = text.lower().strip()
        # Try multi-word phrases first
        for phrase in ["e minor", "a minor", "c major", "d major", "g major"]:
            if phrase in clean:
                return VOICE_CHORD_MAP[phrase]
        # Try individual words
        for word in clean.split():
            w = "".join(c for c in word if c.isalnum())
            if w in VOICE_CHORD_MAP:
                return VOICE_CHORD_MAP[w]
        return None

    def get_chord(self, hold_duration=2.5):
        if self.last_chord is not None and (time.time() - self.last_event_time) < hold_duration:
            return self.last_chord
        return None

    def clear_chord(self):
        self.last_chord = None

    def stop(self):
        self.running = False


# ═════════════════════════════════════════════════════════════════════════════
#  Audio initialisation
# ═════════════════════════════════════════════════════════════════════════════
def init_audio():
    try:
        # Larger buffer (1024) gives smoother sustained playback with less dropouts
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
        pygame.mixer.set_num_channels(4)
    except Exception as exc:
        print(f"[Warning] Audio mixer init failed: {exc}")
        return {}

    # Generate guitar WAVs if needed
    if generate_all_guitar_sounds:
        generate_all_guitar_sounds()

    sounds = {}
    for chord_id, info in CHORD_CONFIG.items():
        fp = info["file"]
        if os.path.exists(fp):
            try:
                sounds[chord_id] = pygame.mixer.Sound(fp)
            except Exception as ex:
                print(f"[Error] Could not load {fp}: {ex}")
        else:
            print(f"[Warning] Sound file not found: {fp}")
    return sounds


# ═════════════════════════════════════════════════════════════════════════════
#  MediaPipe model helpers
# ═════════════════════════════════════════════════════════════════════════════
def ensure_model_file():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        print("[Setup] Downloading MediaPipe hand model…")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("[Setup] Model downloaded.")
        except Exception as exc:
            print(f"[Error] Download failed: {exc}")
            sys.exit(1)
    return MODEL_PATH


def create_hand_detector():
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
        print("[Info] Using MediaPipe Solutions API")
        mp_hands = mp.solutions.hands
        return mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            # Slightly relaxed thresholds → faster inference on CPU
            min_detection_confidence=0.6,
            min_tracking_confidence=0.55,
        ), "solutions"
    else:
        print("[Info] Using MediaPipe Tasks Vision API")
        model_file = ensure_model_file()
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        base_options = python.BaseOptions(model_asset_path=model_file)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.55,
            min_tracking_confidence=0.50,
        )
        return vision.HandLandmarker.create_from_options(options), "tasks"


# ═════════════════════════════════════════════════════════════════════════════
#  Smart 3D Gesture Classifier — Orientation-Invariant & Scale-Normalized
# ═════════════════════════════════════════════════════════════════════════════
def analyze_hand_landmarks(landmarks):
    """
    Smart 3D Finger Classifier:
    1. Transforms 3D landmarks into a local hand-centric orthonormal basis.
    2. 3D Orientation-invariant (works regardless of hand rotation/tilt/angle).
    3. Scale-normalized by wrist-to-middle-MCP distance.
    4. Multi-criterion scoring (local extension, Euclidean extension ratio, PIP flex angle).
    5. Left/Right hand agnostic thumb detection.

    Returns:
      (finger_states, count)
      - finger_states: [Thumb, Index, Middle, Ring, Pinky] (list of 5 booleans)
      - count: total extended fingers (0-5)
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

    w     = pts[0]       # Wrist
    m_mid = pts[9]       # Middle MCP
    m_idx = pts[5]       # Index MCP
    m_pky = pts[17]      # Pinky MCP

    # Y-axis (Palm upward vector from Wrist to Middle MCP)
    v_y = m_mid - w
    norm_y = np.linalg.norm(v_y)
    if norm_y < 1e-6:
        return [False] * 5, 0
    v_y = v_y / norm_y
    hand_scale = norm_y

    # Palm reference vectors
    v_idx = m_idx - w
    v_pky = m_pky - w

    # Z-axis (Palm normal plane)
    v_z = np.cross(v_idx, v_pky)
    norm_z = np.linalg.norm(v_z)
    if norm_z < 1e-6:
        v_z = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        v_z = v_z / norm_z

    # X-axis (Across palm)
    v_x = np.cross(v_y, v_z)
    norm_x = np.linalg.norm(v_x)
    if norm_x < 1e-6:
        v_x = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        v_x = v_x / norm_x

    # Orthonormal transformation basis R (3x3)
    R = np.vstack([v_x, v_y, v_z])
    pts_local = (pts - w) @ R.T    # (21, 3)

    extended = []

    # 1. Thumb analysis (1: CMC, 2: MCP, 3: IP, 4: TIP)
    t_tip = pts[4]
    t_ip  = pts[3]

    d_tip_idx = np.linalg.norm(t_tip - pts[5]) / hand_scale
    d_tip_pky = np.linalg.norm(t_tip - pts[17]) / hand_scale
    d_ip_idx  = np.linalg.norm(t_ip - pts[5]) / hand_scale

    thumb_spread = (d_tip_idx > 0.52) and (d_tip_pky > 0.75) and (d_tip_idx > d_ip_idx * 1.05)
    thumb_local_ext = np.linalg.norm(pts_local[4] - pts_local[2]) / hand_scale
    thumb_up = thumb_spread or (thumb_local_ext > 0.55 and d_tip_idx > 0.48)
    extended.append(bool(thumb_up))

    # 2. Main 4 Fingers (Index, Middle, Ring, Pinky)
    finger_indices = [
        (5, 6, 7, 8),     # Index: MCP, PIP, DIP, TIP
        (9, 10, 11, 12),  # Middle: MCP, PIP, DIP, TIP
        (13, 14, 15, 16), # Ring: MCP, PIP, DIP, TIP
        (17, 18, 19, 20)  # Pinky: MCP, PIP, DIP, TIP
    ]

    for mcp, pip, dip, tip in finger_indices:
        loc_ext = (pts_local[tip, 1] - pts_local[mcp, 1]) / max(hand_scale, 1e-5)
        pip_loc_ext = (pts_local[pip, 1] - pts_local[mcp, 1]) / max(hand_scale, 1e-5)

        d_tip_w = np.linalg.norm(pts[tip] - w)
        d_pip_w = np.linalg.norm(pts[pip] - w)
        euclid_ratio = d_tip_w / max(d_pip_w, 1e-5)

        u = pts[pip] - pts[mcp]
        v = pts[tip] - pts[pip]
        u_norm = np.linalg.norm(u)
        v_norm = np.linalg.norm(v)
        if u_norm > 1e-6 and v_norm > 1e-6:
            cos_flex = np.dot(u, v) / (u_norm * v_norm)
        else:
            cos_flex = 1.0

        is_extended = (loc_ext > pip_loc_ext + 0.12) and (euclid_ratio > 1.05) and (cos_flex > 0.40)
        extended.append(bool(is_extended))

    return extended, min(sum(extended), 5)


def count_raised_fingers(landmarks):
    _, count = analyze_hand_landmarks(landmarks)
    return count


def get_stable_count(history):
    if not history:
        return 0
    counts = {}
    for v in history:
        counts[v] = counts.get(v, 0) + 1
    most_common = max(counts, key=counts.get)
    if counts[most_common] >= (len(history) // 2 + 1):
        return most_common
    return history[-1]


# ═════════════════════════════════════════════════════════════════════════════
#  OpenCV drawing helpers
# ═════════════════════════════════════════════════════════════════════════════
def draw_hand_landmarks(frame, landmarks, finger_states=None):
    h, w, _ = frame.shape
    coords = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    # Skeleton lines — warm amber
    for p1, p2 in HAND_CONNECTIONS:
        cv2.line(frame, coords[p1], coords[p2], (0, 165, 255), 2)

    # Knuckle/tip dots
    tip_ids = [4, 8, 12, 16, 20]
    for idx, (x, y) in enumerate(coords):
        if idx in tip_ids:
            finger_idx = tip_ids.index(idx)
            is_active = finger_states[finger_idx] if (finger_states and finger_idx < len(finger_states)) else True
            color  = (0, 240, 120) if is_active else (100, 100, 140)
            radius = 8 if is_active else 5
            cv2.circle(frame, (x, y), radius, color, -1)
            cv2.circle(frame, (x, y), radius + 3, (255, 255, 255), 1)
        else:
            cv2.circle(frame, (x, y), 4, (255, 140, 30), -1)
            cv2.circle(frame, (x, y), 6, (255, 255, 255), 1)



def alpha_rect(frame, x1, y1, x2, y2, color, alpha):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_pill(frame, text, cx, cy, fill, text_color=(255, 255, 255), scale=0.50):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    px, py = 10, 6
    x1, y1 = cx - tw // 2 - px, cy - th // 2 - py
    x2, y2 = cx + tw // 2 + px, cy + th // 2 + py
    cv2.rectangle(frame, (x1, y1), (x2, y2), fill, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
    cv2.putText(frame, text, (x1 + px, y2 - py),
                cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, 1, cv2.LINE_AA)


def draw_fretboard_sidebar(frame, active_fingers):
    """Draw a simple fretboard graphic on the right side of the frame."""
    h, w, _ = frame.shape
    sb_x = w - 68
    # Sidebar background
    alpha_rect(frame, sb_x, 120, w - 4, h - 44, (25, 28, 38), 0.88)

    # Guitar neck strings (vertical lines)
    string_cols = [sb_x + 10 + i * 9 for i in range(6)]
    for sx in string_cols:
        cv2.line(frame, (sx, 130), (sx, h - 50), (180, 140, 60), 1)

    # Fret lines (horizontal)
    fret_rows = [130 + i * ((h - 180) // (FRET_LINES + 1)) for i in range(1, FRET_LINES + 1)]
    for fy in fret_rows:
        cv2.line(frame, (sb_x + 8, fy), (sb_x + 58, fy), (200, 160, 80), 2)

    # Nut (top thick line)
    cv2.line(frame, (sb_x + 8, 130), (sb_x + 58, 130), (220, 190, 100), 3)

    # Fret dots — highlight the active chord fret pattern
    chord_dot_patterns = {
        1: [2],       # Em — dots at fret 2
        2: [2],       # Am — dots at fret 2
        3: [1, 2],    # C  — dots at fret 1&2
        4: [2, 3],    # D  — dots at fret 2&3
        5: [2, 3],    # G  — dots at fret 2&3
    }
    if active_fingers in chord_dot_patterns and active_fingers > 0:
        dot_color = CHORD_CONFIG[active_fingers]["color"]
        for fret_idx in chord_dot_patterns[active_fingers]:
            dot_y = 130 + fret_idx * ((h - 180) // (FRET_LINES + 1)) - ((h - 180) // (FRET_LINES + 1)) // 2
            dot_x = sb_x + 33
            cv2.circle(frame, (dot_x, dot_y), 8, dot_color, -1)
            cv2.circle(frame, (dot_x, dot_y), 8, (255, 255, 255), 1)


def draw_strumline(frame, active_fingers, frame_count):
    """Animated strum wave that pulses when a chord is active."""
    if active_fingers <= 0:
        return
    h, w, _ = frame.shape
    color = CHORD_CONFIG[active_fingers]["color"]
    amp   = 6
    freq  = 0.04
    phase = frame_count * 0.18
    pts   = []
    for x in range(10, w - 80, 3):
        y = int(h // 2 + amp * math.sin(freq * x + phase))
        pts.append((x, y))
    if len(pts) > 1:
        overlay = frame.copy()
        for i in range(len(pts) - 1):
            cv2.line(overlay, pts[i], pts[i + 1], color, 2)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)


# ═════════════════════════════════════════════════════════════════════════════
#  Main HUD draw
# ═════════════════════════════════════════════════════════════════════════════
def draw_ui(frame, instant_count, stable_count, chord_info,
            play_status, has_hand, voice_status, finger_states=None):
    h, w, _ = frame.shape

    # ── Header bar ────────────────────────────────────────────────────────────
    alpha_rect(frame, 0, 0, w, 110, (14, 16, 24), 0.88)
    cv2.line(frame, (0, 110), (w, 110), AMBER[::-1], 2)

    # Title
    cv2.putText(frame, "GESTURE GUITAR", (18, 36),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, AMBER, 2, cv2.LINE_AA)
    cv2.putText(frame, "5-Finger Chord Controller", (20, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 150, 80), 1, cv2.LINE_AA)

    # Voice status badge
    v_col = (0, 210, 255) if "Listening" in voice_status or "'" in voice_status else (130, 130, 130)
    cv2.putText(frame, f"[ {voice_status} ]", (w - 260, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, v_col, 1, cv2.LINE_AA)

    # Play status badge
    s_colors = {"Playing": (0, 210, 90), "Ready": (0, 130, 255)}
    s_col = s_colors.get(play_status, (120, 120, 160))
    cv2.putText(frame, f"[ {play_status} ]", (w - 260, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, s_col, 1, cv2.LINE_AA)

    # Finger / chord readout
    if has_hand or stable_count > 0:
        cv2.putText(frame, f"Fingers: {instant_count}", (18, 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2, cv2.LINE_AA)
        if chord_info:
            ch_col = chord_info["color"]
            cv2.putText(frame, f"Chord: {chord_info['full']}", (210, 88),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, ch_col, 2, cv2.LINE_AA)
        elif stable_count == 0:
            cv2.putText(frame, "Chord: [Mute]", (210, 88),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (140, 140, 140), 2, cv2.LINE_AA)

        # Real-time finger indicators HUD badge: [T] [I] [M] [R] [P]
        if finger_states and len(finger_states) == 5:
            finger_names = ["T", "I", "M", "R", "P"]
            start_x = w - 260
            badge_y = 88
            for idx, (fname, active) in enumerate(zip(finger_names, finger_states)):
                bx = start_x + idx * 30
                fill_color = (0, 220, 110) if active else (40, 44, 58)
                txt_color  = (10, 10, 10)  if active else (130, 135, 150)
                draw_pill(frame, fname, bx, badge_y, fill_color, text_color=txt_color, scale=0.40)
    else:
        cv2.putText(frame, "Show your hand to the camera", (18, 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (120, 120, 120), 1, cv2.LINE_AA)

    # ── Chord reference pills (centre row) ───────────────────────────────────
    pill_y   = 135
    pill_gap = (w - 80) // 6
    for i, (fid, info) in enumerate(CHORD_CONFIG.items()):
        cx   = pill_gap * (i + 1)
        active = (fid == stable_count)
        fill = info["color"] if active else (40, 44, 58)
        label = f"{fid}:{info['name']}"
        draw_pill(frame, label, cx, pill_y, fill,
                  text_color=(10, 10, 10) if active else (200, 200, 200))

    # ── Footer ────────────────────────────────────────────────────────────────
    alpha_rect(frame, 0, h - 38, w, h, (10, 12, 18), 0.90)
    cv2.line(frame, (0, h - 38), (w, h - 38), (50, 55, 70), 1)
    cv2.putText(frame,
                "1-5 Fingers / Keys 1-5,E,A,C,D,G → Play Chord  |  V: Voice Toggle  |  Q: Quit",
                (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 155, 170), 1, cv2.LINE_AA)


# ═════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print(" 🎸  Gesture Guitar — Starting up…")
    print(" Initialising audio and synthesising guitar chord samples…")

    sounds = init_audio()
    if not sounds:
        print("[Notice] No audio — running in visual-only mode.")

    voice = VoiceCommandHandler()
    voice.start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("\n" + "=" * 60)
        print(" [ERROR] Cannot open webcam!")
        print("=" * 60 + "\n")
        voice.stop()
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    # *** Key lag fix: keep internal buffer at 1 frame so we always get
    # the LATEST frame from the sensor, not a stale queued frame. ***
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    detector, api_type = create_hand_detector()

    print(" Webcam ready.  Show 1-5 fingers to play a chord.")
    print(" Fist = mute.  Keys 1-5 / E,A,C,D,G / Voice commands also work.")
    print(" Press V to toggle voice listening.  Q to quit.")
    print("=" * 60 + "\n")

    KEY_CHORD_MAP = {
        ord('1'): 1, ord('e'): 1, ord('E'): 1,
        ord('2'): 2, ord('a'): 2, ord('A'): 2,
        ord('3'): 3, ord('c'): 3, ord('C'): 3,
        ord('4'): 4, ord('d'): 4, ord('D'): 4,
        ord('5'): 5, ord('g'): 5, ord('G'): 5,
    }

    finger_history     = [deque(maxlen=DEBOUNCE_WINDOW_SIZE) for _ in range(5)]
    last_played        = -1
    play_status        = "Ready"
    frame_count        = 0
    last_chord_time    = 0.0       # cooldown: avoid re-firing same chord mid-sustain
    cached_landmarks   = []        # reuse last landmarks on skipped inference frames
    CHORD_COOLDOWN_S   = 0.45      # seconds before same chord can retrigger
    INFER_EVERY_N      = 1         # run MediaPipe on every frame for sharp precision

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[Error] Failed to read webcam frame.")
            break

        frame = cv2.flip(frame, 1)
        frame_count += 1

        has_hand       = False
        instant_count  = 0
        landmarks_list = []
        instant_states = [False] * 5
        stable_states  = [False] * 5

        # ── Hand detection (Full resolution for crisp keypoints) ──────────────
        run_inference = (frame_count % INFER_EVERY_N == 0)
        if run_inference:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if api_type == "solutions":
                results = detector.process(rgb_frame)
                if results.multi_hand_landmarks:
                    cached_landmarks = [h_lm.landmark for h_lm in results.multi_hand_landmarks]
                else:
                    cached_landmarks = []
            else:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                det_result = detector.detect(mp_image)
                cached_landmarks = det_result.hand_landmarks if det_result.hand_landmarks else []

        landmarks_list = cached_landmarks

        # Only use the first detected hand
        if landmarks_list:
            has_hand = True
            lm = landmarks_list[0]
            instant_states, instant_count = analyze_hand_landmarks(lm)

            # Per-finger temporal debouncing
            for f_idx in range(5):
                finger_history[f_idx].append(instant_states[f_idx])
                # Majority vote for finger stability
                is_active = sum(finger_history[f_idx]) >= (len(finger_history[f_idx]) // 2 + 1)
                stable_states[f_idx] = is_active

            stable_count = min(sum(stable_states), 5)
            draw_hand_landmarks(frame, lm, stable_states)
        else:
            for f_idx in range(5):
                finger_history[f_idx].append(False)
            stable_states = [False] * 5
            stable_count  = -1

        # ── Voice ─────────────────────────────────────────────────────────────
        voice_chord  = voice.get_chord(hold_duration=2.5)
        voice_active = voice_chord is not None

        # ── Keyboard ──────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            print("Exiting Gesture Guitar…")
            break
        elif key in (ord('v'), ord('V')):
            voice.toggle()
            print(f"[Voice] {voice.status}")
        elif key in KEY_CHORD_MAP:
            stable_count = KEY_CHORD_MAP[key]
            has_hand     = True
        elif voice_active and not has_hand:
            if voice_chord == 0:
                pygame.mixer.Channel(CHORD_CHANNEL_ID).stop()
                voice.clear_chord()
                last_played  = 0
                stable_count = 0
                has_hand     = False
            else:
                stable_count = voice_chord
                has_hand     = True

        # ── No-hand / no-voice: let chord ring out naturally ───────────────────
        # Do NOT stop the channel — the WAV will decay to silence on its own.
        # Only reset last_played so a re-gesture triggers a fresh strum.
        if not has_hand and key not in KEY_CHORD_MAP and not voice_active:
            last_played = -1

        # ── Chord playback ────────────────────────────────────────────────────
        now = time.time()
        if stable_count > 0:
            chord_changed  = (stable_count != last_played)
            cooldown_ok    = (now - last_chord_time) > CHORD_COOLDOWN_S
            if chord_changed or (cooldown_ok and not pygame.mixer.Channel(CHORD_CHANNEL_ID).get_busy()):
                if stable_count in CHORD_CONFIG and stable_count in sounds:
                    ch = pygame.mixer.Channel(CHORD_CHANNEL_ID)
                    # Smoothly fade out the previous chord to avoid abrupt cut
                    if ch.get_busy():
                        ch.fadeout(60)          # 60 ms fade-out of old chord
                        pygame.time.wait(30)    # brief pause so fadeout starts
                    # Fade-in new chord over 80 ms → no harsh burst onset
                    ch.play(sounds[stable_count], fade_ms=80)
                    last_played     = stable_count
                    last_chord_time = now
                    print(f"[Guitar] {CHORD_CONFIG[stable_count]['full']}")
                play_status = "Playing"
        elif has_hand and stable_count == 0:
            ch = pygame.mixer.Channel(CHORD_CHANNEL_ID)
            if ch.get_busy():
                ch.fadeout(80)   # smooth mute on fist gesture
            play_status = "Ready"
            last_played  = 0
        else:
            play_status = "Ready"

        # ── Animated strum line ───────────────────────────────────────────────
        draw_strumline(frame, stable_count if stable_count > 0 else 0, frame_count)

        # ── Fretboard sidebar ─────────────────────────────────────────────────
        draw_fretboard_sidebar(frame, stable_count if stable_count > 0 else 0)

        # ── HUD overlay ───────────────────────────────────────────────────────
        chord_info = CHORD_CONFIG.get(stable_count) if stable_count in CHORD_CONFIG else None
        draw_ui(frame, instant_count, stable_count, chord_info,
                play_status, has_hand, voice.status, finger_states=stable_states if has_hand else None)

        cv2.imshow("Gesture Guitar", frame)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    voice.stop()
    cap.release()
    cv2.destroyAllWindows()
    if pygame.mixer.get_init():
        pygame.mixer.quit()
    print("Cleanup complete. Rock on! 🎸")


if __name__ == "__main__":
    main()
