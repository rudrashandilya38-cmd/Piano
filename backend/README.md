# 🎸 Gesture Guitar — Backend API

Python/Flask REST API that serves guitar chord audio files and metadata.
Deploy on **Render** (free tier works fine).

## Deploy to Render

1. Push the `backend/` folder to a GitHub repository.
2. Go to [render.com](https://render.com) → **New** → **Web Service**.
3. Connect the repo and set:
   - **Root Directory**: `backend` (if the repo contains both frontend & backend)
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn server:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`
   - **Environment**: Python 3
4. Click **Create Web Service**.

> **Note:** On free Render tier, the service sleeps after 15 min of inactivity.
> The first request after sleep takes ~30 s to cold-start.

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check — returns `{"status":"ok"}` |
| `GET` | `/api/chords` | Returns chord config JSON |
| `POST` | `/api/generate` | Regenerates all sound WAV files |
| `GET` | `/api/sounds/<name>` | Streams a WAV file (`guitar_em.wav` etc.) |

### Example

```bash
# Health check
curl https://your-render-app.onrender.com/health

# Get chord data
curl https://your-render-app.onrender.com/api/chords

# Download a sound file
curl https://your-render-app.onrender.com/api/sounds/guitar_em.wav -o em.wav
```

## Local development

```bash
pip install -r requirements.txt
python server.py
# Runs on http://localhost:5000
```

## Files

| File | Purpose |
|---|---|
| `server.py` | Flask API server (main entry point) |
| `guitar_sounds.py` | Classical guitar chord synthesizer |
| `generate_sounds.py` | Harmonium sound generator (legacy) |
| `main.py` | Desktop OpenCV app (run locally with Python) |
| `sounds/` | Pre-generated guitar chord WAV files |
| `requirements.txt` | Python dependencies |
| `Procfile` | Render/Heroku process config |

## Running the desktop app (local only)

The desktop app (`main.py`) uses your webcam directly via OpenCV and plays
sounds through Pygame. It does **not** require a server — run it standalone:

```bash
pip install -r requirements.txt
python main.py
```
