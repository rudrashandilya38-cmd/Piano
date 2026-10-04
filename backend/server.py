"""
server.py — Gesture Guitar Backend API (Render deployment)
===========================================================
Flask REST API that wraps the guitar sound generation utilities.
Exposes endpoints for:
  - GET  /health           → health check
  - GET  /api/chords       → returns chord configuration metadata
  - POST /api/generate     → generates sound files on the server
  - GET  /api/sounds/<name> → streams a generated sound WAV file

The frontend (Vercel) runs fully client-side with MediaPipe + Web Audio API.
This backend is the server-side complement for sound generation and
can optionally serve chord data to the frontend.

Run locally:
    pip install -r requirements.txt
    python server.py

Deploy on Render:
    - Build command: pip install -r requirements.txt
    - Start command: python server.py
    - Environment: Python 3
"""

import os
import sys

try:
    from flask import Flask, jsonify, send_file, abort
    from flask_cors import CORS
except ImportError:
    print("[Error] Missing Flask. Run: pip install -r requirements.txt")
    sys.exit(1)

try:
    from guitar_sounds import generate_all_guitar_sounds, GUITAR_CHORDS, SOUNDS_DIR
except ImportError:
    print("[Error] guitar_sounds.py not found. Make sure it is in the same directory.")
    sys.exit(1)

# ── App setup ─────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, origins="*")   # Allow Vercel frontend to call this API

# Chord configuration (mirrors frontend CHORD_CONFIG)
CHORD_CONFIG = {
    0: {"name": "Mute", "full": "Silence",  "color": "#8a99ad", "freqs": []},
    1: {"name": "Em",   "full": "E Minor",  "color": "#ff5a5a",
        "freqs": [82.41, 123.47, 164.81, 196.00, 246.94, 329.63],
        "file": "guitar_em.wav"},
    2: {"name": "Am",   "full": "A Minor",  "color": "#ffa53c",
        "freqs": [110.00, 164.81, 220.00, 261.63, 329.63],
        "file": "guitar_am.wav"},
    3: {"name": "C",    "full": "C Major",  "color": "#50dc78",
        "freqs": [130.81, 164.81, 196.00, 261.63, 329.63],
        "file": "guitar_c.wav"},
    4: {"name": "D",    "full": "D Major",  "color": "#50a0ff",
        "freqs": [146.83, 220.00, 293.66, 369.99],
        "file": "guitar_d.wav"},
    5: {"name": "G",    "full": "G Major",  "color": "#c850ff",
        "freqs": [98.00, 123.47, 196.00, 246.94, 329.63, 392.00],
        "file": "guitar_g.wav"},
}

# Generate sounds on startup if missing
def ensure_sounds():
    sounds_dir = os.path.join(os.path.dirname(__file__), "sounds")
    all_exist = all(
        os.path.exists(os.path.join(sounds_dir, info["file"]))
        for cid, info in CHORD_CONFIG.items() if cid != 0
    )
    if not all_exist:
        print("[Setup] Generating guitar chord WAV files…")
        generate_all_guitar_sounds()
        print("[Setup] Sound generation complete.")
    else:
        print("[Setup] All sound files present.")


# ── Routes ────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint for Render uptime monitoring."""
    return jsonify({"status": "ok", "service": "gesture-guitar-api"})


@app.route("/api/chords", methods=["GET"])
def get_chords():
    """Return chord configuration metadata to the frontend."""
    return jsonify({
        "chords": {
            str(k): {
                "name":  v["name"],
                "full":  v["full"],
                "color": v["color"],
                "freqs": v["freqs"],
            }
            for k, v in CHORD_CONFIG.items()
        }
    })


@app.route("/api/generate", methods=["POST"])
def generate_sounds():
    """
    Trigger server-side sound generation.
    Useful if sounds need to be regenerated (e.g. after a deployment).
    """
    try:
        generate_all_guitar_sounds()
        return jsonify({"status": "ok", "message": "All chord WAV files generated."})
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/sounds/<string:name>", methods=["GET"])
def serve_sound(name):
    """
    Stream a guitar chord WAV file.
    Valid names: guitar_em.wav, guitar_am.wav, guitar_c.wav, guitar_d.wav, guitar_g.wav
    """
    allowed = {info["file"] for cid, info in CHORD_CONFIG.items() if cid != 0}
    if name not in allowed:
        abort(404)

    sounds_dir = os.path.join(os.path.dirname(__file__), "sounds")
    filepath   = os.path.join(sounds_dir, name)

    if not os.path.exists(filepath):
        # Try generating on-demand
        try:
            generate_all_guitar_sounds()
        except Exception:
            abort(500)

    if not os.path.exists(filepath):
        abort(404)

    return send_file(filepath, mimetype="audio/wav")


# ── Entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    ensure_sounds()
    port = int(os.environ.get("PORT", 5000))
    print(f"[Gesture Guitar API] Starting on port {port}…")
    app.run(host="0.0.0.0", port=port, debug=False)
