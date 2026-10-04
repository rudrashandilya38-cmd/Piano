/**
 * app.js — Gesture Guitar Web App
 * ================================
 * Real-time hand gesture guitar using MediaPipe Tasks Vision + Web Audio API.
 *
 * Sound Engine: Real-time oscillator nodes (OscillatorNode + GainNode).
 *   No pre-computation — zero blocking. Camera starts instantly.
 * Gesture Engine: MediaPipe HandLandmarker (VIDEO mode), 3D finger scoring.
 * Voice Engine: Web Speech API (webkitSpeechRecognition fallback).
 */

import { HandLandmarker, FilesetResolver }
  from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.15";

// ═══════════════════════════════════════════════════════════════════════════
// CHORD DATA
// ═══════════════════════════════════════════════════════════════════════════
// Frequencies in Hz for open-chord strings (standard tuning)
const CHORD_CONFIG = {
  0: { name: "Mute", full: "Silence",  color: "#8a99ad", freqs: [] },
  1: { name: "Em",   full: "E Minor",  color: "#ff5a5a",
       freqs: [82.41, 123.47, 164.81, 196.00, 246.94, 329.63] },
  2: { name: "Am",   full: "A Minor",  color: "#ffa53c",
       freqs: [110.00, 164.81, 220.00, 261.63, 329.63] },
  3: { name: "C",    full: "C Major",  color: "#50dc78",
       freqs: [130.81, 164.81, 196.00, 261.63, 329.63] },
  4: { name: "D",    full: "D Major",  color: "#50a0ff",
       freqs: [146.83, 220.00, 293.66, 369.99] },
  5: { name: "G",    full: "G Major",  color: "#c850ff",
       freqs: [98.00, 123.47, 196.00, 246.94, 329.63, 392.00] }
};

// Hand skeleton connections (MediaPipe landmark indices)
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
let activeNodes       = [];   // currently playing oscillator+gain nodes

let handLandmarker    = null;
let modelReady        = false;
let cameraStarting    = false;  // guard against double-click
let videoEl           = null;
let outputCanvas      = null;
let outputCtx         = null;
let stringsCanvas     = null;
let stringsCtx        = null;

let cameraStream      = null;
let isCameraRunning   = false;
let facingMode        = "user"; // "user" = front, "environment" = back
let rafId             = null;
let lastVideoTime     = -1;

let currentChord      = 0;
let lastChord         = -1;

// Debounce: majority vote over last N frames
const DEBOUNCE_N      = 4;
let gestureHistory    = [];

// Voice
let speechRec         = null;
let voiceActive       = false;

// String vibration animation amplitudes [0..1]
let vibAmps           = [0,0,0,0,0,0];

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
// AUDIO — Real-time oscillator synthesis (zero blocking)
// ═══════════════════════════════════════════════════════════════════════════
function ensureAudioCtx() {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
  if (audioCtx.state === "suspended") audioCtx.resume();
}

/**
 * Play a guitar chord by spawning real-time oscillator nodes.
 * One OscillatorNode per string per harmonic — no pre-computed buffers,
 * so there is ZERO main-thread blocking.
 *
 * Sound model:
 *  - Harmonic series: h1=1.0, h2=0.38, h3=0.16, h4=0.07, h5=0.03 (nylon warmth)
 *  - 3-voice chorus: -2, 0, +2 cents per string (intonation irregularity)
 *  - Per-string onset delay: 0..55 ms (strum simulation)
 *  - Gain envelope: 30 ms Hann attack → exponential decay over 4 s
 *  - Lowpass filter at 4 kHz for warmth
 */
function playChord(chordId) {
  ensureAudioCtx();
  stopAllNodes();

  if (chordId === 0 || !CHORD_CONFIG[chordId]) return;

  kickStrings();

  const cfg       = CHORD_CONFIG[chordId];
  const freqs     = cfg.freqs;
  const now       = audioCtx.currentTime;
  const strumSpan = 0.055;
  const strumStep = strumSpan / Math.max(freqs.length - 1, 1);

  const harmonicAmps  = [1.00, 0.38, 0.16, 0.07, 0.03];
  const detuneCents   = [-2.0, 0.0, 2.0];
  const detuneWeights = [0.25, 1.00, 0.25];
  const totalWeight   = 1.5;

  // Low-pass warmth filter shared for this chord event
  const masterGain = audioCtx.createGain();
  masterGain.gain.setValueAtTime(1.0, now);
  masterGain.connect(audioCtx.destination);

  const lpf = audioCtx.createBiquadFilter();
  lpf.type = "lowpass";
  lpf.frequency.setValueAtTime(4000, now);
  lpf.Q.setValueAtTime(0.6, now);
  lpf.connect(masterGain);

  freqs.forEach((freq, sIdx) => {
    const onset    = now + sIdx * strumStep;
    const duration = 4.0;
    // Decay rate by string frequency (bass slower, treble faster)
    const normF    = Math.max(0, Math.min(1, (freq - 80) / 320));
    const decayEnd = onset + duration;

    harmonicAmps.forEach((hAmp, hIdx) => {
      const hNum = hIdx + 1;

      detuneCents.forEach((dc, dIdx) => {
        const f0     = freq * hNum * Math.pow(2, dc / 1200);
        const weight = detuneWeights[dIdx] / totalWeight;
        const amp    = hAmp * weight * (0.55 / freqs.length);

        const osc   = audioCtx.createOscillator();
        const gNode = audioCtx.createGain();

        osc.type = "sine";
        osc.frequency.setValueAtTime(f0, onset);

        // Hann attack 30 ms, then exponential decay
        const decayRate = 0.25 + normF * 0.45;
        gNode.gain.setValueAtTime(0, onset);
        gNode.gain.linearRampToValueAtTime(amp, onset + 0.030);
        gNode.gain.setTargetAtTime(0.001, onset + 0.030, 1 / decayRate);

        osc.connect(gNode);
        gNode.connect(lpf);

        osc.start(onset);
        osc.stop(decayEnd);

        activeNodes.push({ osc, gNode });
      });
    });
  });

  activeNodes.push({ masterGain, lpf });
}

function stopAllNodes() {
  const now = audioCtx ? audioCtx.currentTime : 0;
  activeNodes.forEach(n => {
    try {
      if (n.gNode) {
        n.gNode.gain.cancelScheduledValues(now);
        n.gNode.gain.linearRampToValueAtTime(0, now + 0.05);
      }
      if (n.osc) {
        n.osc.stop(now + 0.06);
      }
    } catch (_) {}
  });
  activeNodes = [];
}

// ═══════════════════════════════════════════════════════════════════════════
// MEDIAPIPE MODEL LOADING
// ═══════════════════════════════════════════════════════════════════════════
async function loadMediaPipeModel() {
  const overlay  = document.getElementById("loading-overlay");
  const loadText = document.getElementById("loading-text");

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
      loadText.textContent = "⚠️ Failed to load AI model. Check your internet connection and refresh.";
      console.error(err);
      return;
    }
  }

  modelReady = true;
  overlay.classList.add("hidden");
  // Show the camera prompt ONLY after model is ready (not on page load)
  document.getElementById("camera-prompt").classList.remove("hidden");
}

// ═══════════════════════════════════════════════════════════════════════════
// CAMERA
// ═══════════════════════════════════════════════════════════════════════════
async function startCamera() {
  if (cameraStarting) return;  // prevent double-click
  cameraStarting = true;

  const btnCam = document.getElementById("btn-start-camera");
  const prompt = document.getElementById("camera-prompt");

  btnCam.disabled = true;
  btnCam.querySelector("span").textContent = "Starting…";

  try {
    // getUserMedia FIRST (must stay close to gesture event for mobile browsers)
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

    // Unlock AudioContext AFTER getUserMedia (safe on mobile now)
    ensureAudioCtx();

    videoEl.srcObject = cameraStream;
    await videoEl.play();

    isCameraRunning = true;
    prompt.classList.add("hidden");
    btnCam.disabled = false;
    btnCam.querySelector("span").textContent = "Stop Camera";
    btnCam.classList.add("btn-danger");

    syncCanvasSize();
    lastVideoTime = -1;
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

  // Only process when a new frame is available
  if (videoEl.readyState >= 2 && videoEl.currentTime !== lastVideoTime) {
    lastVideoTime = videoEl.currentTime;

    outputCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);

    if (modelReady && handLandmarker) {
      const result = handLandmarker.detectForVideo(videoEl, performance.now());

      if (result.landmarks && result.landmarks.length > 0) {
        const lms = result.landmarks[0];
        drawSkeleton(lms);

        const states = fingerStates(lms);
        updateFingerPills(states);

        const count = states.filter(Boolean).length;
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

// ── 3D Finger Extension Scoring ─────────────────────────────────────────
function fingerStates(lms) {
  const dist = (a, b) => {
    const dx = a.x-b.x, dy = a.y-b.y, dz = (a.z||0)-(b.z||0);
    return Math.sqrt(dx*dx + dy*dy + dz*dz);
  };

  const wrist    = lms[0];
  const indexMcp = lms[5];

  // Thumb: tip(4) further from index-MCP than thumb-MCP(2)
  const thumbOut = dist(lms[4], indexMcp) > dist(lms[2], indexMcp) * 1.35;

  // Fingers: tip further from wrist than PIP joint * threshold
  const indexOut  = dist(lms[8],  wrist) > dist(lms[6],  wrist) * 1.08;
  const middleOut = dist(lms[12], wrist) > dist(lms[10], wrist) * 1.08;
  const ringOut   = dist(lms[16], wrist) > dist(lms[14], wrist) * 1.08;
  const pinkyOut  = dist(lms[20], wrist) > dist(lms[18], wrist) * 1.08;

  return [thumbOut, indexOut, middleOut, ringOut, pinkyOut];
}

function majority(arr) {
  if (!arr.length) return 0;
  const cnt = {};
  let best = arr[0], bestC = 0;
  for (const v of arr) {
    cnt[v] = (cnt[v]||0) + 1;
    if (cnt[v] > bestC) { bestC = cnt[v]; best = v; }
  }
  return best;
}

// ── Chord Trigger ────────────────────────────────────────────────────────
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
  outputCanvas.width  = stage.clientWidth;
  outputCanvas.height = stage.clientHeight;
  stringsCanvas.width  = stage.clientWidth;
  stringsCanvas.height = stringsCanvas.parentElement.clientHeight || 80;
}

function drawSkeleton(lms) {
  const W = outputCanvas.width;
  const H = outputCanvas.height;
  const color = CHORD_CONFIG[currentChord].color;

  const px = (lm) => [(1 - lm.x) * W, lm.y * H];  // mirrored X

  // Connections
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

  // Knuckle joints (all landmarks)
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

// ─── Guitar String Animation ─────────────────────────────────────────────
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
      vibAmps[i] *= 0.96;  // damping
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

  document.getElementById("chord-display-name").textContent = cfg.name;
  document.getElementById("chord-display-name").style.color = cfg.color;
  document.getElementById("finger-count-pill").textContent  =
    id === 0 ? "Fist (Mute)" : `${id} Finger${id > 1 ? "s" : ""}`;

  document.getElementById("chord-badge").style.setProperty("--active-chord-color", cfg.color);
  document.documentElement.style.setProperty("--active-chord-color", cfg.color);

  document.querySelectorAll(".chord-btn").forEach(btn => {
    btn.classList.toggle("active", parseInt(btn.dataset.chord) === id);
  });
}

function updateFingerPills(states) {
  const ids    = ["pill-thumb","pill-index","pill-middle","pill-ring","pill-pinky"];
  const color  = CHORD_CONFIG[currentChord].color;
  ids.forEach((id, i) => {
    const el = document.getElementById(id);
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
// VOICE COMMANDS  (Web Speech API)
// ═══════════════════════════════════════════════════════════════════════════
function setupVoice() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {
    document.getElementById("btn-voice").style.display = "none";
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
    btn.classList.add("active");
    label.textContent = "Voice On";
  } else {
    speechRec.stop();
    btn.classList.remove("active");
    label.textContent = "Voice Off";
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

  // ── Start / Stop Camera — click + touchend for instant mobile response
  function handleCameraToggle(e) {
    e.preventDefault();
    e.stopPropagation();
    isCameraRunning ? stopCamera() : startCamera();
  }
  btnCam.addEventListener("click",    handleCameraToggle);
  btnCam.addEventListener("touchend", handleCameraToggle, { passive: false });

  // ── Camera-prompt "Tap to Enable" button
  function handlePromptStart(e) {
    e.preventDefault();
    e.stopPropagation();
    startCamera();
  }
  btnPrompt.addEventListener("click",    handlePromptStart);
  btnPrompt.addEventListener("touchend", handlePromptStart, { passive: false });

  // Flip Camera
  document.getElementById("btn-camera-flip").addEventListener("click", flipCamera);

  // Voice Toggle
  document.getElementById("btn-voice").addEventListener("click", toggleVoice);

  // Chord Deck Buttons (touch + click)
  document.querySelectorAll(".chord-btn").forEach(btn => {
    const trigger = (e) => {
      e.preventDefault();
      ensureAudioCtx();
      setChord(parseInt(btn.dataset.chord));
    };
    btn.addEventListener("click",      trigger);
    btn.addEventListener("touchstart", trigger, { passive: false });
  });

  // Keyboard shortcuts
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
