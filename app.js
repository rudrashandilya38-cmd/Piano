/**
 * app.js — Gesture Guitar Web App
 * ================================
 * Real-time hand gesture guitar using MediaPipe Tasks Vision + Web Audio API.
 *
 * Sound Engine: Real-time oscillator nodes (OscillatorNode + GainNode).
 *   No pre-computation — zero blocking. Camera starts instantly.
 * Gesture Engine: MediaPipe HandLandmarker (VIDEO mode), 3D finger scoring.
 * Voice Engine: Web Speech API (webkitSpeechRecognition fallback).
 *
 * CAMERA FIX:
 *   - camera-prompt starts hidden; only shown after model load completes.
 *   - ensureAudioCtx() is deferred so getUserMedia stays in the direct
 *     gesture-event call stack (required by mobile browsers).
 *   - startCamera() guard removed so button is never permanently disabled.
 */

import { HandLandmarker, FilesetResolver }
  from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.15";

const BACKEND_URL = "https://piano-1-s83z.onrender.com";

// ═══════════════════════════════════════════════════════════════════════════
// CHORD DATA
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
let activeNodes       = [];

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
let lastVideoTime     = -1;

let currentChord      = 0;
let lastChord         = -1;

const DEBOUNCE_N      = 4;
let gestureHistory    = [];

let speechRec         = null;
let voiceActive       = false;

let vibAmps           = [0,0,0,0,0,0];

// Track if camera start is already in progress to prevent double-clicks
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
// AUDIO
// ═══════════════════════════════════════════════════════════════════════════
const audioBuffers = {};
let soundsPreloaded = false;

async function preloadBackendSounds() {
  if (soundsPreloaded || !audioCtx) return;
  soundsPreloaded = true;
  for (const [id, cfg] of Object.entries(CHORD_CONFIG)) {
    if (!cfg.file) continue;
    try {
      const res = await fetch(`${BACKEND_URL}/api/sounds/${cfg.file}`);
      if (res.ok) {
        const arrayBuf = await res.arrayBuffer();
        audioBuffers[id] = await audioCtx.decodeAudioData(arrayBuf);
        console.log(`[Backend Sound] Preloaded ${cfg.name} from Render API`);
      }
    } catch (err) {
      console.warn(`[Backend Sound] Fallback to synth for ${cfg.name}:`, err);
    }
  }
}

function ensureAudioCtx() {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    preloadBackendSounds();
  }
  if (audioCtx.state === "suspended") audioCtx.resume();
}

function playChord(chordId) {
  ensureAudioCtx();
  stopAllNodes();

  if (chordId === 0 || !CHORD_CONFIG[chordId]) return;

  kickStrings();

  const cfg = CHORD_CONFIG[chordId];
  const now = audioCtx.currentTime;

  // Use preloaded recorded WAV buffer from Render backend if available
  if (audioBuffers[chordId]) {
    const src = audioCtx.createBufferSource();
    src.buffer = audioBuffers[chordId];
    src.connect(audioCtx.destination);
    src.start(now);
    activeNodes.push({ osc: src });
    return;
  }


  const freqs     = cfg.freqs;
  const strumSpan = 0.055;
  const strumStep = strumSpan / Math.max(freqs.length - 1, 1);


  const harmonicAmps  = [1.00, 0.38, 0.16, 0.07, 0.03];
  const detuneCents   = [-2.0, 0.0, 2.0];
  const detuneWeights = [0.25, 1.00, 0.25];
  const totalWeight   = 1.5;

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
      loadText.textContent = "⚠️ Failed to load AI model. Check your internet connection and refresh.";
      console.error(err);
      return;
    }
  }

  modelReady = true;
  // Hide loading overlay and show the camera prompt
  overlay.classList.add("hidden");
  prompt.classList.remove("hidden");  // NOW show camera prompt after model ready
}

// ═══════════════════════════════════════════════════════════════════════════
// CAMERA
// ═══════════════════════════════════════════════════════════════════════════
async function startCamera() {
  // Prevent double-invocation (e.g. if button clicked twice fast)
  if (cameraStarting) return;
  cameraStarting = true;

  const btnCam = document.getElementById("btn-start-camera");
  const prompt = document.getElementById("camera-prompt");

  // Immediately reflect busy state
  btnCam.disabled = true;
  btnCam.querySelector("span").textContent = "Starting…";

  try {
    // NOTE: getUserMedia MUST be called as close to the gesture event as
    // possible. ensureAudioCtx is called AFTER getUserMedia for mobile compat.
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

    // Unlock audio context after camera permission granted (safe now)
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
    // Re-show the prompt so user can try again
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
  // Show the prompt again
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
    const dx = a.x - b.x, dy = a.y - b.y, dz = (a.z || 0) - (b.z || 0);
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  };

  const wrist    = lms[0];
  const indexMcp = lms[5];
  const pinkyMcp = lms[17];

  // 1. Thumb (Tip 4, MCP 2, CMC 1)
  const dThumbTipIndex = dist(lms[4], indexMcp);
  const dThumbMcpIndex = dist(lms[2], indexMcp);
  const dThumbTipWrist = dist(lms[4], wrist);
  const dThumbMcpWrist = dist(lms[2], wrist);
  const thumbOut = (dThumbTipIndex > dThumbMcpIndex * 1.15) || (dThumbTipWrist > dThumbMcpWrist * 1.12);

  // 2. Index (Tip 8, PIP 6, MCP 5)
  const dIndexTipWrist = dist(lms[8], wrist);
  const dIndexPipWrist = dist(lms[6], wrist);
  const dIndexTipMcp   = dist(lms[8], indexMcp);
  const dIndexPipMcp   = dist(lms[6], indexMcp);
  const indexOut = (dIndexTipWrist > dIndexPipWrist * 1.05) || (dIndexTipMcp > dIndexPipMcp * 1.25);

  // 3. Middle (Tip 12, PIP 10, MCP 9)
  const middleMcp       = lms[9];
  const dMiddleTipWrist = dist(lms[12], wrist);
  const dMiddlePipWrist = dist(lms[10], wrist);
  const dMiddleTipMcp   = dist(lms[12], middleMcp);
  const dMiddlePipMcp   = dist(lms[10], middleMcp);
  const middleOut = (dMiddleTipWrist > dMiddlePipWrist * 1.05) || (dMiddleTipMcp > dMiddlePipMcp * 1.25);

  // 4. Ring (Tip 16, PIP 14, MCP 13)
  const ringMcp       = lms[13];
  const dRingTipWrist = dist(lms[16], wrist);
  const dRingPipWrist = dist(lms[14], wrist);
  const dRingTipMcp   = dist(lms[16], ringMcp);
  const dRingPipMcp   = dist(lms[14], ringMcp);
  const ringOut = (dRingTipWrist > dRingPipWrist * 1.04) || (dRingTipMcp > dRingPipMcp * 1.25);

  // 5. Pinky (Tip 20, PIP 18, MCP 17) — Tailored specifically for pinky proportions
  const dPinkyTipWrist = dist(lms[20], wrist);
  const dPinkyPipWrist = dist(lms[18], wrist);
  const dPinkyTipMcp   = dist(lms[20], pinkyMcp);
  const dPinkyPipMcp   = dist(lms[18], pinkyMcp);
  const pinkyOut = (dPinkyTipWrist > dPinkyPipWrist * 1.02) || (dPinkyTipMcp > dPinkyPipMcp * 1.20);

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
// VOICE COMMANDS
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

  // ── Start / Stop Camera ──────────────────────────────────────────────────
  // Uses both click and touchend to ensure immediate response on mobile.
  // touchend handler calls preventDefault to avoid ghost-click delay.
  function handleCameraToggle(e) {
    e.preventDefault();
    e.stopPropagation();
    if (isCameraRunning) {
      stopCamera();
    } else {
      startCamera();
    }
  }

  btnCam.addEventListener("click",     handleCameraToggle);
  btnCam.addEventListener("touchend",  handleCameraToggle, { passive: false });

  // ── Camera-prompt "Tap to Enable" button ─────────────────────────────────
  function handlePromptStart(e) {
    e.preventDefault();
    e.stopPropagation();
    startCamera();
  }

  btnPrompt.addEventListener("click",    handlePromptStart);
  btnPrompt.addEventListener("touchend", handlePromptStart, { passive: false });

  // ── Flip Camera ──────────────────────────────────────────────────────────
  document.getElementById("btn-camera-flip").addEventListener("click", flipCamera);

  // ── Voice Toggle ─────────────────────────────────────────────────────────
  document.getElementById("btn-voice").addEventListener("click", toggleVoice);

  // ── Chord Deck Buttons (touch + click) ───────────────────────────────────
  document.querySelectorAll(".chord-btn").forEach(btn => {
    const trigger = (e) => {
      e.preventDefault();
      ensureAudioCtx();
      setChord(parseInt(btn.dataset.chord));
    };
    btn.addEventListener("click",      trigger);
    btn.addEventListener("touchstart", trigger, { passive: false });
  });

  // ── Keyboard shortcuts ───────────────────────────────────────────────────
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
