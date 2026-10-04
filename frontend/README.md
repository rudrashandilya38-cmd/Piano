# 🎸 Gesture Guitar — Frontend

Pure browser app. No build step required. Deploy directly to **Vercel**.

## Deploy to Vercel

1. Push the `frontend/` folder to a GitHub repository (or upload it directly).
2. Go to [vercel.com](https://vercel.com) → **New Project** → Import the repo.
3. Set **Root Directory** to `frontend/` (or deploy from root if repo is just the frontend).
4. **No build command** needed — it's plain HTML/CSS/JS.
5. Click **Deploy**.

## How it works

| Feature | Technology |
|---|---|
| Hand detection | MediaPipe HandLandmarker (loaded from CDN) |
| Sound synthesis | Web Audio API (oscillator nodes — zero latency) |
| Voice commands | Web Speech API |
| Camera | `getUserMedia` (browser native) |
| Styling | Vanilla CSS + Glassmorphism |

## Local development

Just open `index.html` in a browser — but note that ES modules require a
local HTTP server (not `file://`). Use any of:

```bash
# Python
python -m http.server 3000

# Node
npx serve .

# VS Code Live Server extension
```

Then open `http://localhost:3000`.

## Files

| File | Purpose |
|---|---|
| `index.html` | App shell and HTML structure |
| `app.js` | All JavaScript — MediaPipe, Web Audio, gesture logic |
| `style.css` | Glassmorphism design system |
| `vercel.json` | CORS headers required for MediaPipe WASM |

## Camera Button Fix (v2)

- `camera-prompt` overlay now starts **hidden** and only appears after the
  AI model finishes loading — so it never blocks the header Start Camera button.
- `getUserMedia` is called **before** `AudioContext` creation to satisfy
  mobile browser gesture-event requirements (Safari, Chrome mobile).
- Both buttons use `touchend` with `preventDefault` for instant mobile response
  without the 300 ms tap delay.
