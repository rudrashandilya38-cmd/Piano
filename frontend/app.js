/**
 * app.js — Gesture Guitar Web App
 * ================================
 * Real-time hand gesture guitar using MediaPipe Tasks Vision + Web Audio API.
 *
 * Sound Engine: Pre-rendered AudioBuffers generated in-memory via nylon guitar
 *   Karplus-Strong / Plucked-String Synthesis (zero network latency, 0ms play response).
 * Gesture Engine: MediaPipe HandLandmarker (VIDEO mode) + Smart 3D Orientation-Invariant
 *   and Scale-Normalized Finger Scoring algorithm identical to Python main.py.
 * Voice Engine: Web Speech API (webkitSpeechRecognition fallback).
 */

import { HandLandmarker, FilesetResolver }
  from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.15";

// ═══════════════════════════════════════════════════════════════════════════
// CHORD CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════
const CHORD_CONFIG = {
  0: { name: "Mute", full: "Silence",  color: "#8a99ad", freqs: [], file: null },
  1: { name: "Em",   full: "E Minor",  color: "#ff5a5a",
       freqs: [82.41, 123.47, 164.81, 196.00, 246.94, 329.63], file: "guitar_em.wav" },
  2: { name: "Am",   full: "A Minor",  color: "#ffa53c",
       freqs: [110.00, 164.81, 220.00, 261.63, 329.63], file: "guitar_am.wav" },
  3: { name: "C",    full: "C Major",  color: "#50dc78",
       freqs: [130.81, 164.81, 196.00, 261.63, 329.63], file: "guitar_c.wav" },
  4: { name: "D",    full: "D Major",  color: "#50a0ff",
       freqs: [146.83, 220.00, 293.66, 369.99], file: "guitar_d.wav" },
  5: { name: "G",    full: "G Major",  color: "#c850ff",
       freqs: [98.00, 123.47, 196.00, 246.94, 329.63, 392.00], file: "guitar_g.wav" }
};

const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],
  [13,17],[17,18],[18,19],[19,20],
  [0,17]
];

// ═══════════════════════════════════════════════════════════════════════════
// STATE
// ═══════════════════════════════════════════════════════════════════════════
let audioCtx          = null;
const guitarBuffers   = {};
let activeSourceNode  = null;
let soundsSynthesized = false;

let handLandmarker    = null;
let modelReady        = false;
let videoEl           = null;
let outputCanvas      = null;
let outputCtx         = null;
let stringsCanvas     = null;
let stringsCtx        = null;

let cameraStream      = null;
let isCameraRunning   = false;
let facingMode        = "user";
let rafId             = null;

let currentChord      = 0;
let lastChord         = -1;

const DEBOUNCE_N      = 3; // Smaller window = faster response (matches main.py DEBOUNCE_WINDOW_SIZE)
let gestureHistory    = [];

let speechRec         = null;
let voiceActive       = false;

let vibAmps           = [0,0,0,0,0,0];
let cameraStarting    = false;

// ═══════════════════════════════════════════════════════════════════════════
// BOOT
// ═══════════════════════════════════════════════════════════════════════════
document.addEventListener("DOMContentLoaded", () => {
  videoEl       = document.getElementById("webcam");
  outputCanvas  = document.getElementById("output-canvas");
  outputCtx     = outputCanvas.getContext("2d");
  stringsCanvas = document.getElementById("strings-canvas");
  stringsCtx    = stringsCanvas.getContext("2d");

  syncCanvasSize();
  window.addEventListener("resize", syncCanvasSize);
  requestAnimationFrame(animateStrings);

  bindUI();
  loadMediaPipeModel();
});

// ═══════════════════════════════════════════════════════════════════════════
// AUDIO ENGINE (Plucked Nylon Guitar Synthesis + Instant Buffer Caching)
// ═══════════════════════════════════════════════════════════════════════════
function ensureAudioCtx() {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
  if (audioCtx.state === "suspended") {
    audioCtx.resume();
  }
  buildGuitarChordBuffers();
}

function buildGuitarChordBuffers() {
  if (soundsSynthesized || !audioCtx) return;
  soundsSynthesized = true;

  for (const [idStr, cfg] of Object.entries(CHORD_CONFIG)) {
    const id = parseInt(idStr, 10);
    if (id === 0 || !cfg.freqs || cfg.freqs.length === 0) continue;
    try {
      guitarBuffers[id] = generateGuitarChordBuffer(audioCtx, cfg.freqs);
    } catch (err) {
      console.warn(`[Guitar Synth Error for chord ${cfg.name}]:`, err);
    }
  }

  preloadStaticWavs();
}

async function preloadStaticWavs() {
  for (const [idStr, cfg] of Object.entries(CHORD_CONFIG)) {
    const id = parseInt(idStr, 10);
    if (!cfg.file) continue;
    try {
      const res = await fetch(`sounds/${cfg.file}`);
      if (res.ok) {
        const buf = await res.arrayBuffer();
        const decoded = await audioCtx.decodeAudioData(buf);
        guitarBuffers[id] = decoded;
      }
    } catch (_) {}
  }
}

/**
 * Classical Guitar Nylon-String Synthesis (matches guitar_sounds.py)
 */
function generateGuitarChordBuffer(ctx, freqs, duration = 3.5) {
  const sampleRate = ctx.sampleRate;
  const numSamples = Math.floor(duration * sampleRate);
  const buffer = ctx.createBuffer(1, numSamples, sampleRate);
  const data = buffer.getChannelData(0);

  const strumSpan = 0.055;
  const strumStep = strumSpan / Math.max(freqs.length - 1, 1);

  const harmonicAmps = [1.00, 0.38, 0.16, 0.07, 0.03];
  const detuneCents  = [-2.0, 0.0, 2.0];
  const detuneGains  = [0.25, 1.00, 0.25];
  const totalGain    = 1.5;

  for (let sIdx = 0; sIdx < freqs.length; sIdx++) {
    const freq = freqs[sIdx];
    const onsetSec = sIdx * strumStep;
    const onsetSample = Math.floor(onsetSec * sampleRate);

    const normF = Math.max(0, Math.min(1, (freq - 80) / 320));
    const decayRate = 0.25 + normF * 0.45;

    for (let i = onsetSample; i < numSamples; i++) {
      const t = (i - onsetSample) / sampleRate;

      // Soft fingertip attack (12ms)
      let att = 1.0;
      if (t < 0.012) {
        att = Math.sin((Math.PI / 2) * (t / 0.012));
      }

      const env = Math.exp(-decayRate * t);

      let val = 0.0;
      for (let d = 0; d < 3; d++) {
        const dc = detuneCents[d];
        const dg = detuneGains[d];
        const f0 = freq * Math.pow(2, dc / 1200);

        for (let h = 0; h < harmonicAmps.length; h++) {
          const hNum = h + 1;
          const amp = harmonicAmps[h];
          val += dg * amp * Math.sin(2 * Math.PI * f0 * hNum * t);
        }
      }

      const sig = (val / totalGain) * env;
      data[i] += att * sig;
    }
  }

  // Soft tanh saturation for warm body resonance & peak normalization
  let maxVal = 0;
  for (let i = 0; i < numSamples; i++) {
    const absV = Math.abs(data[i]);
    if (absV > maxVal) maxVal = absV;
  }
  if (maxVal < 1e-6) maxVal = 1;

  for (let i = 0; i < numSamples; i++) {
    const normVal = data[i] / maxVal;
    data[i] = Math.tanh(1.8 * normVal) * 0.85;
  }

  return buffer;
}

function playChord(chordId) {
  ensureAudioCtx();

  if (activeSourceNode) {
    try {
      activeSourceNode.stop();
      activeSourceNode.disconnect();
    } catch (_) {}
    activeSourceNode = null;
  }

  if (chordId === 0 || !CHORD_CONFIG[chordId]) return;

  kickStrings();

  const buf = guitarBuffers[chordId];
  if (buf) {
    const src = audioCtx.createBufferSource();
    src.buffer = buf;
    src.connect(audioCtx.destination);
    src.start(0);
    activeSourceNode = src;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// MEDIAPIPE MODEL LOADING
// ═══════════════════════════════════════════════════════════════════════════
async function loadMediaPipeModel() {
  const overlay  = document.getElementById("loading-overlay");
  const loadText = document.getElementById("loading-text");
  const prompt   = document.getElementById("camera-prompt");

  const tryLoad = async (delegate) => {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.15/wasm"
    );
    return HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        delegate
      },
      runningMode: "VIDEO",
      numHands: 1
    });
  };

  try {
    handLandmarker = await tryLoad("GPU");
  } catch {
    try {
      loadText.textContent = "Loading model (CPU fallback)…";
      handLandmarker = await tryLoad("CPU");
    } catch (err) {
      loadText.textContent = "⚠️ Failed to load AI model. Check internet connection and refresh.";
      console.error(err);
      return;
    }
  }

  modelReady = true;
  overlay.classList.add("hidden");
  prompt.classList.remove("hidden");
}

// ═══════════════════════════════════════════════════════════════════════════
// CAMERA
// ═══════════════════════════════════════════════════════════════════════════
async function startCamera() {
  if (cameraStarting) return;
  cameraStarting = true;

  const btnCam = document.getElementById("btn-start-camera");
  const prompt = document.getElementById("camera-prompt");

  btnCam.disabled = true;
  btnCam.querySelector("span").textContent = "Starting…";

  try {
    if (cameraStream) {
      cameraStream.getTracks().forEach(t => t.stop());
      cameraStream = null;
    }

    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode,
        width:  { ideal: 1280 },
        height: { ideal: 720 }
      },
      audio: false
    });

    ensureAudioCtx();

    videoEl.srcObject = cameraStream;
    await videoEl.play();

    isCameraRunning = true;
    prompt.classList.add("hidden");
    btnCam.disabled = false;
    btnCam.querySelector("span").textContent = "Stop Camera";
    btnCam.classList.add("btn-danger");

    syncCanvasSize();
    if (rafId) cancelAnimationFrame(rafId);
    rafId = requestAnimationFrame(processFrame);

  } catch (err) {
    console.error("Camera error:", err);
    btnCam.disabled = false;
    btnCam.querySelector("span").textContent = "Start Camera";
    btnCam.classList.remove("btn-danger");
    if (!isCameraRunning) prompt.classList.remove("hidden");

    let msg = "Could not access the camera.";
    if (err.name === "NotAllowedError")  msg = "Camera permission denied. Please allow camera access and try again.";
    if (err.name === "NotFoundError")    msg = "No camera found on this device.";
    if (err.name === "NotReadableError") msg = "Camera is already in use by another app.";
    showToast(msg, "error");
  } finally {
    cameraStarting = false;
  }
}

function stopCamera() {
  isCameraRunning = false;
  if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
  if (cameraStream) {
    cameraStream.getTracks().forEach(t => t.stop());
    cameraStream = null;
  }
  videoEl.srcObject = null;

  const btnCam = document.getElementById("btn-start-camera");
  btnCam.querySelector("span").textContent = "Start Camera";
  btnCam.classList.remove("btn-danger");
  document.getElementById("camera-prompt").classList.remove("hidden");

  outputCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
  updateFingerPills([false, false, false, false, false]);
  setChord(0);
}

async function flipCamera() {
  facingMode = facingMode === "user" ? "environment" : "user";
  if (isCameraRunning) {
    stopCamera();
    await startCamera();
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// FRAME PROCESSING & GESTURE DETECTION
// ═══════════════════════════════════════════════════════════════════════════
function processFrame() {
  if (!isCameraRunning) return;

  if (videoEl.readyState >= 2) {
    const now = performance.now();
    outputCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);

    if (modelReady && handLandmarker) {
      const result = handLandmarker.detectForVideo(videoEl, now);

      if (result.landmarks && result.landmarks.length > 0) {
        const lms = result.landmarks[0];
        drawSkeleton(lms);

        const { states, count } = analyzeHandLandmarks(lms);
        updateFingerPills(states);

        gestureHistory.push(count);
        if (gestureHistory.length > DEBOUNCE_N) gestureHistory.shift();

        setChord(majority(gestureHistory));
      } else {
        updateFingerPills([false, false, false, false, false]);
        setChord(0);
      }
    }
  }

  rafId = requestAnimationFrame(processFrame);
}

/**
 * Smart 3D Finger Classifier — 3D Orientation-Invariant & Scale-Normalized
 * Optimized for high responsiveness across all 5 fingers (Thumb, Index, Middle, Ring, Pinky).
 */
function analyzeHandLandmarks(lms) {
  if (!lms || lms.length < 21) {
    return { states: [false, false, false, false, false], count: 0 };
  }

  const pts = lms.map(lm => [lm.x, lm.y, lm.z || 0]);

  const w     = pts[0];   // Wrist
  const m_mid = pts[9];   // Middle MCP
  const m_idx = pts[5];   // Index MCP
  const m_pky = pts[17];  // Pinky MCP

  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const norm = (v) => Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  const dot = (u, v) => u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
  const cross = (u, v) => [
    u[1] * v[2] - u[2] * v[1],
    u[2] * v[0] - u[0] * v[2],
    u[0] * v[1] - u[1] * v[0]
  ];
  const dist = (a, b) => norm(sub(a, b));

  // Y-axis (Palm upward vector from Wrist to Middle MCP)
  let v_y = sub(m_mid, w);
  const hand_scale = norm(v_y);
  if (hand_scale < 1e-6) {
    return { states: [false, false, false, false, false], count: 0 };
  }
  v_y = [v_y[0] / hand_scale, v_y[1] / hand_scale, v_y[2] / hand_scale];

  // Palm reference vectors
  const v_idx = sub(m_idx, w);
  const v_pky = sub(m_pky, w);

  // Z-axis (Palm normal plane)
  let v_z = cross(v_idx, v_pky);
  let norm_z = norm(v_z);
  if (norm_z < 1e-6) {
    v_z = [0, 0, 1];
  } else {
    v_z = [v_z[0] / norm_z, v_z[1] / norm_z, v_z[2] / norm_z];
  }

  // X-axis (Across palm)
  let v_x = cross(v_y, v_z);
  let norm_x = norm(v_x);
  if (norm_x < 1e-6) {
    v_x = [1, 0, 0];
  } else {
    v_x = [v_x[0] / norm_x, v_x[1] / norm_x, v_x[2] / norm_x];
  }

  // Orthonormal transformation basis: pts_local = (pts - w) @ R.T
  const pts_local = pts.map(p => {
    const p_rel = sub(p, w);
    return [dot(p_rel, v_x), dot(p_rel, v_y), dot(p_rel, v_z)];
  });

  const extended = [];

  // 1. Thumb analysis (1: CMC, 2: MCP, 3: IP, 4: TIP)
  const t_tip = pts[4];
  const t_ip  = pts[3];

  const d_tip_idx = dist(t_tip, pts[5]) / hand_scale;
  const d_tip_pky = dist(t_tip, pts[17]) / hand_scale;
  const d_ip_idx  = dist(t_ip, pts[5]) / hand_scale;

  const thumb_spread = (d_tip_idx > 0.42) && (d_tip_pky > 0.55) && (d_tip_idx > d_ip_idx * 1.02);
  const thumb_local_ext = dist(pts_local[4], pts_local[2]) / hand_scale;
  const thumb_side_out = Math.abs(pts_local[4][0] - pts_local[2][0]) / hand_scale;
  const thumb_up = thumb_spread || (thumb_local_ext > 0.42 && d_tip_idx > 0.40) || (thumb_side_out > 0.30);
  extended.push(Boolean(thumb_up));

  // 2. Main 4 Fingers (Index, Middle, Ring, Pinky)
  const finger_indices = [
    [5, 6, 7, 8],     // Index: MCP, PIP, DIP, TIP
    [9, 10, 11, 12],  // Middle: MCP, PIP, DIP, TIP
    [13, 14, 15, 16], // Ring: MCP, PIP, DIP, TIP
    [17, 18, 19, 20]  // Pinky: MCP, PIP, DIP, TIP
  ];

  for (const [mcp, pip, dip, tip] of finger_indices) {
    const loc_ext = (pts_local[tip][1] - pts_local[mcp][1]) / Math.max(hand_scale, 1e-5);
    const pip_loc_ext = (pts_local[pip][1] - pts_local[mcp][1]) / Math.max(hand_scale, 1e-5);

    const d_tip_w = dist(pts[tip], w);
    const d_pip_w = dist(pts[pip], w);
    const euclid_ratio = d_tip_w / Math.max(d_pip_w, 1e-5);

    const u = sub(pts[pip], pts[mcp]);
    const v = sub(pts[tip], pts[pip]);
    const u_norm = norm(u);
    const v_norm = norm(v);
    const cos_flex = (u_norm > 1e-6 && v_norm > 1e-6) ? (dot(u, v) / (u_norm * v_norm)) : 1.0;

    const isPinky = (mcp === 17);
    const min_euclid = isPinky ? 1.00 : 1.02;
    const ext_margin = isPinky ? 0.04 : 0.05;

    const is_extended = (loc_ext > pip_loc_ext + ext_margin) && (euclid_ratio > min_euclid) && (cos_flex > 0.20);
    extended.push(Boolean(is_extended));
  }

  const count = Math.min(extended.filter(Boolean).length, 5);
  return { states: extended, count };
}

function majority(arr) {
  if (!arr.length) return 0;
  const cnt = {};
  let best = arr[0], bestC = 0;
  for (const v of arr) {
    cnt[v] = (cnt[v] || 0) + 1;
    if (cnt[v] > bestC) { bestC = cnt[v]; best = v; }
  }
  return best;
}

function setChord(id) {
  currentChord = id;
  if (currentChord !== lastChord) {
    lastChord = currentChord;
    playChord(currentChord);
    updateChordUI(currentChord);
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// CANVAS RENDERING
// ═══════════════════════════════════════════════════════════════════════════
function syncCanvasSize() {
  const stage = document.querySelector(".stage-container");
  if (!stage) return;
  outputCanvas.width  = stage.clientWidth;
  outputCanvas.height = stage.clientHeight;
  stringsCanvas.width  = stage.clientWidth;
  stringsCanvas.height = stringsCanvas.parentElement ? stringsCanvas.parentElement.clientHeight : 80;
}

function drawSkeleton(lms) {
  const W = outputCanvas.width;
  const H = outputCanvas.height;
  const color = CHORD_CONFIG[currentChord].color;

  const px = (lm) => [(1 - lm.x) * W, lm.y * H];

  outputCtx.lineWidth   = 3.5;
  outputCtx.strokeStyle = color;
  outputCtx.shadowColor = color;
  outputCtx.shadowBlur  = 10;

  for (const [i, j] of HAND_CONNECTIONS) {
    const [x1, y1] = px(lms[i]);
    const [x2, y2] = px(lms[j]);
    outputCtx.beginPath();
    outputCtx.moveTo(x1, y1);
    outputCtx.lineTo(x2, y2);
    outputCtx.stroke();
  }

  for (let k = 0; k < lms.length; k++) {
    const [x, y] = px(lms[k]);
    const isTip = [4,8,12,16,20].includes(k);
    outputCtx.beginPath();
    outputCtx.arc(x, y, isTip ? 7 : 3.5, 0, 2 * Math.PI);
    outputCtx.fillStyle = isTip ? "#ffffff" : color;
    outputCtx.shadowBlur = isTip ? 18 : 6;
    outputCtx.fill();
  }

  outputCtx.shadowBlur = 0;
}

// ── Guitar String Animation ───────────────────────────────────────────────
function kickStrings() {
  vibAmps = vibAmps.map(() => 1.0);
}

function animateStrings(ts) {
  const W = stringsCanvas.width;
  const H = stringsCanvas.height;
  stringsCtx.clearRect(0, 0, W, H);

  const n       = 6;
  const spacing = H / (n + 1);
  const color   = CHORD_CONFIG[currentChord].color;

  for (let i = 0; i < n; i++) {
    const y   = spacing * (i + 1);
    const amp = vibAmps[i];

    stringsCtx.lineWidth   = 1.5 + (n - 1 - i) * 0.35;
    stringsCtx.strokeStyle = amp > 0.04 ? color : "rgba(255,255,255,0.18)";
    stringsCtx.shadowColor = color;
    stringsCtx.shadowBlur  = amp > 0.04 ? 12 * amp : 0;

    stringsCtx.beginPath();
    stringsCtx.moveTo(0, y);

    if (amp > 0.04) {
      const waveAmp = 9 * amp;
      const phase   = ts * 0.012 + i * 0.7;
      for (let x = 0; x <= W; x += 6) {
        const offset = waveAmp * Math.sin((x / W) * Math.PI) * Math.sin(phase + x * 0.04);
        stringsCtx.lineTo(x, y + offset);
      }
      vibAmps[i] *= 0.96;
    } else {
      stringsCtx.lineTo(W, y);
      vibAmps[i] = 0;
    }

    stringsCtx.stroke();
  }

  stringsCtx.shadowBlur = 0;
  requestAnimationFrame(animateStrings);
}

// ═══════════════════════════════════════════════════════════════════════════
// CHORD UI UPDATE
// ═══════════════════════════════════════════════════════════════════════════
function updateChordUI(id) {
  const cfg = CHORD_CONFIG[id];

  const nameEl  = document.getElementById("chord-display-name");
  const countEl = document.getElementById("finger-count-pill");
  const badgeEl = document.getElementById("chord-badge");

  if (nameEl) {
    nameEl.textContent = cfg.name;
    nameEl.style.color = cfg.color;
  }
  if (countEl) {
    countEl.textContent = id === 0 ? "Fist (Mute)" : `${id} Finger${id > 1 ? "s" : ""}`;
  }
  if (badgeEl) {
    badgeEl.style.setProperty("--active-chord-color", cfg.color);
  }

  document.documentElement.style.setProperty("--active-chord-color", cfg.color);

  document.querySelectorAll(".chord-btn").forEach(btn => {
    btn.classList.toggle("active", parseInt(btn.dataset.chord) === id);
  });
}

function updateFingerPills(states) {
  const ids   = ["pill-thumb","pill-index","pill-middle","pill-ring","pill-pinky"];
  const color = CHORD_CONFIG[currentChord].color;
  ids.forEach((id, i) => {
    const el = document.getElementById(id);
    if (!el) return;
    if (states[i]) {
      el.classList.add("active");
      el.style.background = color;
      el.style.boxShadow  = `0 0 10px ${color}`;
    } else {
      el.classList.remove("active");
      el.style.background = "";
      el.style.boxShadow  = "";
    }
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// VOICE COMMANDS
// ═══════════════════════════════════════════════════════════════════════════
function setupVoice() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {
    const btn = document.getElementById("btn-voice");
    if (btn) btn.style.display = "none";
    return false;
  }
  speechRec = new SR();
  speechRec.continuous     = true;
  speechRec.interimResults = false;
  speechRec.lang           = "en-US";

  speechRec.onresult = (e) => {
    const t = e.results[e.results.length - 1][0].transcript.trim().toLowerCase();
    if      (t.includes("e minor") || t.includes("em")     || t === "one") setChord(1);
    else if (t.includes("a minor") || t.includes("am")     || t === "two") setChord(2);
    else if (t.includes("c major") || t === "c" || t === "see"|| t === "three") setChord(3);
    else if (t.includes("d major") || t === "d" || t === "dee"|| t === "four")  setChord(4);
    else if (t.includes("g major") || t === "g" || t === "gee"|| t === "five")  setChord(5);
    else if (t.includes("stop")   || t.includes("mute") || t.includes("silence")) setChord(0);
  };

  speechRec.onerror = () => {};
  speechRec.onend   = () => { if (voiceActive) speechRec.start(); };
  return true;
}

function toggleVoice() {
  if (!speechRec && !setupVoice()) return;
  voiceActive = !voiceActive;
  const btn   = document.getElementById("btn-voice");
  const label = document.getElementById("voice-label");
  if (voiceActive) {
    try { speechRec.start(); } catch (_) {}
    if (btn) btn.classList.add("active");
    if (label) label.textContent = "Voice On";
  } else {
    speechRec.stop();
    if (btn) btn.classList.remove("active");
    if (label) label.textContent = "Voice Off";
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// TOAST NOTIFICATION
// ═══════════════════════════════════════════════════════════════════════════
function showToast(msg, type = "info") {
  const t = document.createElement("div");
  t.className = `toast toast-${type}`;
  t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.classList.add("show"), 10);
  setTimeout(() => { t.classList.remove("show"); setTimeout(() => t.remove(), 400); }, 4000);
}

// ═══════════════════════════════════════════════════════════════════════════
// UI EVENT BINDING
// ═══════════════════════════════════════════════════════════════════════════
function bindUI() {
  const btnCam    = document.getElementById("btn-start-camera");
  const btnPrompt = document.getElementById("btn-start-prompt");

  function handleCameraToggle(e) {
    e.preventDefault();
    e.stopPropagation();
    if (isCameraRunning) {
      stopCamera();
    } else {
      startCamera();
    }
  }

  if (btnCam) {
    btnCam.addEventListener("click",    handleCameraToggle);
    btnCam.addEventListener("touchend", handleCameraToggle, { passive: false });
  }

  function handlePromptStart(e) {
    e.preventDefault();
    e.stopPropagation();
    startCamera();
  }

  if (btnPrompt) {
    btnPrompt.addEventListener("click",    handlePromptStart);
    btnPrompt.addEventListener("touchend", handlePromptStart, { passive: false });
  }

  const flipBtn = document.getElementById("btn-camera-flip");
  if (flipBtn) flipBtn.addEventListener("click", flipCamera);

  const voiceBtn = document.getElementById("btn-voice");
  if (voiceBtn) voiceBtn.addEventListener("click", toggleVoice);

  document.querySelectorAll(".chord-btn").forEach(btn => {
    const trigger = (e) => {
      e.preventDefault();
      ensureAudioCtx();
      setChord(parseInt(btn.dataset.chord));
    };
    btn.addEventListener("click",      trigger);
    btn.addEventListener("touchstart", trigger, { passive: false });
  });

  window.addEventListener("keydown", (e) => {
    if (e.ctrlKey || e.altKey || e.metaKey) return;
    const k = e.key.toLowerCase();
    if      (k === "1" || k === "e") setChord(1);
    else if (k === "2" || k === "a") setChord(2);
    else if (k === "3" || k === "c") setChord(3);
    else if (k === "4" || k === "d") setChord(4);
    else if (k === "5" || k === "g") setChord(5);
    else if (k === "0" || k === "q" || k === "m") setChord(0);
    else if (k === "v") toggleVoice();
  });
}
