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
  0: { name: "Mute", full: "Silence",  color: "#8a99ad", freqs: [], stringIndices: [], file: null },
  1: { name: "Em",   full: "E Minor",  color: "#ff5a5a",
       freqs: [82.41, 123.47, 164.81, 196.00, 246.94, 329.63], stringIndices: [0, 1, 2, 3, 4, 5], file: "guitar_em.wav" },
  2: { name: "Am",   full: "A Minor",  color: "#ffa53c",
       freqs: [110.00, 164.81, 220.00, 261.63, 329.63], stringIndices: [1, 2, 3, 4, 5], file: "guitar_am.wav" },
  3: { name: "C",    full: "C Major",  color: "#50dc78",
       freqs: [130.81, 164.81, 196.00, 261.63, 329.63], stringIndices: [1, 2, 3, 4, 5], file: "guitar_c.wav" },
  4: { name: "D",    full: "D Major",  color: "#50a0ff",
       freqs: [146.83, 220.00, 293.66, 369.99], stringIndices: [2, 3, 4, 5], file: "guitar_d.wav" },
  5: { name: "G",    full: "G Major",  color: "#c850ff",
       freqs: [98.00, 123.47, 196.00, 246.94, 329.63, 392.00], stringIndices: [0, 1, 2, 3, 4, 5], file: "guitar_g.wav" }
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
// STATE & SOUND PRESETS (Stage 2)
// ═══════════════════════════════════════════════════════════════════════════
const GUITAR_PRESETS = {
  nylon: {
    id: "nylon",
    name: "Warm Nylon",
    icon: "🪵",
    model: "nylon",
    desc: "Warm Nylon • Intimate Classical Tone",
    toneBrightness: -2.5, // dB
    bodyLowGain: 3.5,     // dB
    presenceGain: 0.0,    // dB
    reverbWet: 0.14,      // 14%
    reverbDecay: 1.4,     // s
    sustainScale: 1.0,
    ambientMode: false,
    compThreshold: -15,
    compRatio: 2.2
  },
  bright: {
    id: "bright",
    name: "Bright Acoustic",
    icon: "✨",
    model: "steel",
    desc: "Bright Acoustic • Steel-String Chime",
    toneBrightness: 3.5,  // dB
    bodyLowGain: 1.8,     // dB
    presenceGain: 2.8,    // dB
    reverbWet: 0.09,      // 9%
    reverbDecay: 1.1,     // s
    sustainScale: 0.85,
    ambientMode: false,
    compThreshold: -18,
    compRatio: 2.8
  },
  ambient: {
    id: "ambient",
    name: "Dreamy Ambient",
    icon: "🌌",
    model: "ambient",
    desc: "Dreamy Ambient • Ethereal Sustain & Lush Space",
    toneBrightness: 0.5,   // dB
    bodyLowGain: 3.0,     // dB
    presenceGain: 1.0,    // dB
    reverbWet: 0.38,      // 38%
    reverbDecay: 2.8,     // s
    sustainScale: 1.9,
    ambientMode: true,
    compThreshold: -14,
    compRatio: 3.0
  }
};

const SOUND_SETTINGS = {
  preset: "nylon",
  volume: 0.85,
  sustainScale: 1.0,
  reverbWet: 0.14,
  toneOffset: 0.0,
  ambientSustain: false
};

let audioCtx            = null;
let effectsChain        = null;
let roomImpulseBuffer   = null;
let ambientImpulseBuffer = null;
const stringBufferBanks = {
  nylon: {},
  steel: {},
  ambient: {}
};
const activeVoices      = new Set();
let pendingStrumTimers  = [];
let soundsSynthesized   = false;
let roundRobinIndex     = 0;

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
let candidateChord    = 0;
let candidateCount    = 0;
const STABILITY_THRESHOLD = 3; // 3 consecutive frames confirms chord transition (~90-110ms)

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
  syncStudioUI();
  loadMediaPipeModel();
});

// ═══════════════════════════════════════════════════════════════════════════
// AUDIO ENGINE (Acoustic Physical Modeling & Modular Effects Studio)
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

/**
 * Synthesizes a natural stereo acoustic room impulse response buffer.
 */
function generateReverbImpulse(ctx, durationSec, decay) {
  const sampleRate = ctx.sampleRate;
  const length = Math.floor(sampleRate * durationSec);
  const impulse = ctx.createBuffer(2, length, sampleRate);
  const left = impulse.getChannelData(0);
  const right = impulse.getChannelData(1);

  const earlyReflections = [
    { delay: 0.012, gainL: 0.50, gainR: 0.30 },
    { delay: 0.021, gainL: 0.35, gainR: 0.45 },
    { delay: 0.033, gainL: 0.30, gainR: 0.25 },
    { delay: 0.048, gainL: 0.22, gainR: 0.28 }
  ];

  for (let i = 0; i < length; i++) {
    const t = i / sampleRate;
    const env = Math.exp(-t * (6.91 / decay));
    const damping = Math.exp(-t * 2.5);

    left[i] = (Math.random() * 2 - 1) * env * (0.6 + 0.4 * damping);
    right[i] = (Math.random() * 2 - 1) * env * (0.6 + 0.4 * damping);
  }

  for (const ref of earlyReflections) {
    const idx = Math.floor(ref.delay * sampleRate);
    if (idx < length) {
      left[idx] += ref.gainL * 0.4;
      right[idx] += ref.gainR * 0.4;
    }
  }

  let peak = 0;
  for (let i = 0; i < length; i++) {
    const aL = Math.abs(left[i]);
    const aR = Math.abs(right[i]);
    if (aL > peak) peak = aL;
    if (aR > peak) peak = aR;
  }
  if (peak > 1e-5) {
    for (let i = 0; i < length; i++) {
      left[i] /= peak;
      right[i] /= peak;
    }
  }

  return impulse;
}

/**
 * Initializes the modular guitar effects chain:
 * Strings -> Stereo Pan -> EQ (Low/Mid/High) -> Dynamic Compressor
 *         -> Parallel Reverb (Dry/Wet) -> Master Gain -> Peak Limiter -> Destination
 */
function initGuitarAcoustics(ctx) {
  if (effectsChain) return effectsChain;

  // 1. Voice Summing Input Bus
  const voiceBus = ctx.createGain();
  voiceBus.gain.value = 1.0;

  // 2. Equalization Stage:
  // Low Shelf (110 Hz Body Warmth)
  const eqLowShelf = ctx.createBiquadFilter();
  eqLowShelf.type = "lowshelf";
  eqLowShelf.frequency.value = 110;
  eqLowShelf.gain.value = 3.5;

  // De-Mud Notch (400 Hz)
  const eqDeMud = ctx.createBiquadFilter();
  eqDeMud.type = "peaking";
  eqDeMud.frequency.value = 400;
  eqDeMud.Q.value = 1.6;
  eqDeMud.gain.value = -2.0;

  // Soundboard Spruce Top-Plate Resonance (212 Hz)
  const eqTopPlate = ctx.createBiquadFilter();
  eqTopPlate.type = "peaking";
  eqTopPlate.frequency.value = 212;
  eqTopPlate.Q.value = 2.8;
  eqTopPlate.gain.value = 3.2;

  // Presence Bite (3200 Hz) - active on steel acoustic
  const eqPresence = ctx.createBiquadFilter();
  eqPresence.type = "peaking";
  eqPresence.frequency.value = 3200;
  eqPresence.Q.value = 1.8;
  eqPresence.gain.value = 0.0;

  // Tone Brightness High Shelf (4500 Hz)
  const eqToneBrightness = ctx.createBiquadFilter();
  eqToneBrightness.type = "highshelf";
  eqToneBrightness.frequency.value = 4500;
  eqToneBrightness.gain.value = -2.5;

  // 3. Dynamic Compressor Stage (transparent dynamics control)
  const compressor = ctx.createDynamicsCompressor();
  compressor.threshold.value = -15.0; // dB
  compressor.knee.value = 14.0;      // soft knee
  compressor.ratio.value = 2.4;
  compressor.attack.value = 0.015;    // 15 ms
  compressor.release.value = 0.180;   // 180 ms

  // 4. Natural Acoustic Reverb Stage (Parallel Dry/Wet)
  roomImpulseBuffer = generateReverbImpulse(ctx, 1.6, 1.4);
  ambientImpulseBuffer = generateReverbImpulse(ctx, 3.0, 2.8);

  const convolver = ctx.createConvolver();
  convolver.buffer = roomImpulseBuffer;

  const dryGain = ctx.createGain();
  dryGain.gain.value = 1.0 - SOUND_SETTINGS.reverbWet;

  const wetGain = ctx.createGain();
  wetGain.gain.value = SOUND_SETTINGS.reverbWet;

  const postFxSum = ctx.createGain();
  postFxSum.gain.value = 1.0;

  // 5. Master Output Gain
  const masterGain = ctx.createGain();
  masterGain.gain.value = SOUND_SETTINGS.volume;

  // 6. Peak Protection Limiter Stage (Brickwall Limiter + Soft Saturation)
  const limiter = ctx.createDynamicsCompressor();
  limiter.threshold.value = -1.0; // dB
  limiter.knee.value = 0.0;       // hard knee brickwall
  limiter.ratio.value = 20.0;
  limiter.attack.value = 0.002;   // 2 ms fast clamp
  limiter.release.value = 0.050;  // 50 ms fast recovery

  const waveShaper = ctx.createWaveShaper();
  waveShaper.curve = makeWarmSaturationCurve(1024);
  waveShaper.oversample = "2x";

  // Connect Audio Graph:
  voiceBus.connect(eqLowShelf);
  eqLowShelf.connect(eqDeMud);
  eqDeMud.connect(eqTopPlate);
  eqTopPlate.connect(eqPresence);
  eqPresence.connect(eqToneBrightness);
  eqToneBrightness.connect(compressor);

  // Split after compressor into Dry and Wet paths:
  compressor.connect(dryGain);
  dryGain.connect(postFxSum);

  compressor.connect(convolver);
  convolver.connect(wetGain);
  wetGain.connect(postFxSum);

  // Sum -> masterGain -> limiter -> waveShaper -> destination
  postFxSum.connect(masterGain);
  masterGain.connect(limiter);
  limiter.connect(waveShaper);
  waveShaper.connect(ctx.destination);

  effectsChain = {
    voiceBus,
    eqLowShelf,
    eqDeMud,
    eqTopPlate,
    eqPresence,
    eqToneBrightness,
    compressor,
    convolver,
    dryGain,
    wetGain,
    postFxSum,
    masterGain,
    limiter,
    waveShaper
  };

  return effectsChain;
}

function makeWarmSaturationCurve(n_samples = 1024) {
  const curve = new Float32Array(n_samples);
  for (let i = 0; i < n_samples; i++) {
    const x = (i * 2) / (n_samples - 1) - 1;
    curve[i] = Math.tanh(1.25 * x) / Math.tanh(1.25);
  }
  return curve;
}

/**
 * Enhanced Karplus-Strong Physical Modeling Plucked String Synthesizer
 * Supports physical models: 'nylon', 'steel', and 'ambient'
 */
function renderPhysicalString(ctx, freq, durationSec, options = {}) {
  const {
    instrumentType = "nylon",
    velocity = 0.88,
    pluckPos = 0.17,
    brightness = 0.50,
    detuneCents = 0.0
  } = options;

  const isSteel = (instrumentType === "steel");
  const isAmbient = (instrumentType === "ambient");

  const sampleRate = ctx.sampleRate;
  const actualFreq = freq * Math.pow(2, detuneCents / 1200);
  const totalSamples = Math.floor(durationSec * sampleRate);

  const period = sampleRate / actualFreq;
  const N = Math.floor(period);
  const frac = period - N;
  const allpassC = (1 - frac) / (1 + frac);

  const normF = Math.max(0, Math.min(1, (actualFreq - 80) / 320));

  let t60 = 4.6 - normF * 1.8;
  if (isAmbient) {
    t60 = 7.5 - normF * 1.5;
  } else if (isSteel) {
    t60 = 3.6 - normF * 1.2;
  }

  const lossFactor = Math.pow(0.001, 1.0 / (t60 * actualFreq));

  const loopDamp = isSteel
    ? (0.14 + 0.12 * (1 - normF * 0.5))
    : (0.23 + 0.20 * (1 - normF * 0.7));

  const line1 = new Float64Array(N + 2);
  const line2 = new Float64Array(N + 2);

  const period2 = sampleRate / (actualFreq + (isAmbient ? 0.85 : 0.55));
  const N2 = Math.floor(period2);
  const frac2 = period2 - N2;
  const allpassC2 = (1 - frac2) / (1 + frac2);

  const excMs = isSteel ? 0.0022 : (isAmbient ? 0.0045 : 0.0035);
  const excLen = Math.min(N, Math.max(20, Math.floor(sampleRate * (excMs + (1 - velocity) * 0.002))));
  const effPluckPos = isSteel ? 0.14 : pluckPos;
  const pickOffset = Math.max(1, Math.floor(period * effPluckPos));
  const rawExc = new Float64Array(excLen);

  for (let i = 0; i < excLen; i++) {
    const t = i / excLen;
    const win = 0.5 * (1 - Math.cos(2 * Math.PI * t));
    const tri = t < effPluckPos ? (t / effPluckPos) : ((1 - t) / (1 - effPluckPos));
    const noise = (Math.random() * 2 - 1) * (isSteel ? 0.45 : 0.30);
    rawExc[i] = (tri * 0.75 + noise) * win * velocity;
  }

  const combDepth = isSteel ? 0.94 : 0.88;
  for (let i = excLen - 1; i >= 0; i--) {
    if (i >= pickOffset) {
      rawExc[i] -= combDepth * rawExc[i - pickOffset];
    }
  }

  let excFilt = 0;
  const excCutoff = isSteel
    ? (0.42 + velocity * 0.48 + brightness * 0.20)
    : (0.28 + velocity * 0.45 + brightness * 0.15);

  for (let i = 0; i < excLen; i++) {
    excFilt += Math.min(0.95, excCutoff) * (rawExc[i] - excFilt);
    line1[i] = excFilt * 0.85;
    line2[i] = excFilt * (isAmbient ? 0.50 : 0.35);
  }

  const buffer = ctx.createBuffer(1, totalSamples, sampleRate);
  const channelData = buffer.getChannelData(0);

  let readIdx1 = 0, readIdx2 = 0;
  let lpPrev1 = 0, lpPrev2 = 0;
  let apPrevIn1 = 0, apPrevOut1 = 0;
  let apPrevIn2 = 0, apPrevOut2 = 0;
  const couple = isAmbient ? 0.024 : 0.015;

  for (let n = 0; n < totalSamples; n++) {
    const s1 = line1[readIdx1];
    const interp1 = allpassC * (s1 - apPrevOut1) + apPrevIn1;
    apPrevIn1 = s1;
    apPrevOut1 = interp1;
    const damp1 = (1 - loopDamp) * interp1 + loopDamp * lpPrev1;
    lpPrev1 = damp1;

    const s2 = line2[readIdx2];
    const interp2 = allpassC2 * (s2 - apPrevOut2) + apPrevIn2;
    apPrevIn2 = s2;
    apPrevOut2 = interp2;
    const damp2 = (1 - (loopDamp * 0.85)) * interp2 + (loopDamp * 0.85) * lpPrev2;
    lpPrev2 = damp2;

    const next1 = (damp1 * (1 - couple) + damp2 * couple) * lossFactor;
    const next2 = (damp2 * (1 - couple) + damp1 * couple) * (lossFactor * 0.9995);

    line1[readIdx1] = next1;
    line2[readIdx2] = next2;

    channelData[n] = (interp1 * 0.78 + interp2 * 0.38);

    readIdx1++;
    if (readIdx1 >= N) readIdx1 = 0;
    readIdx2++;
    if (readIdx2 >= N2) readIdx2 = 0;
  }

  let prevX = 0, prevY = 0;
  const R = 0.995;
  let peak = 0;
  for (let i = 0; i < totalSamples; i++) {
    const x = channelData[i];
    const y = x - prevX + R * prevY;
    prevX = x;
    prevY = y;
    channelData[i] = y;
    const absY = Math.abs(y);
    if (absY > peak) peak = absY;
  }

  if (peak > 1e-5) {
    const scale = 0.90 / peak;
    for (let i = 0; i < totalSamples; i++) {
      channelData[i] *= scale;
    }
  }

  return buffer;
}

function buildGuitarChordBuffers() {
  if (soundsSynthesized || !audioCtx) return;
  soundsSynthesized = true;

  initGuitarAcoustics(audioCtx);

  const uniqueFreqs = new Set();
  for (const cfg of Object.values(CHORD_CONFIG)) {
    if (cfg.freqs) {
      cfg.freqs.forEach(f => uniqueFreqs.add(f));
    }
  }

  const models = ["nylon", "steel", "ambient"];

  for (const model of models) {
    stringBufferBanks[model] = {};
    for (const freq of uniqueFreqs) {
      const freqKey = freq.toFixed(2);
      stringBufferBanks[model][freqKey] = [];

      let dur = freq < 150 ? 4.5 : (freq < 250 ? 3.8 : 3.2);
      if (model === "ambient") dur = Math.min(7.5, dur * 1.8);
      else if (model === "steel") dur = Math.max(2.5, dur * 0.85);

      // Variation 0: Standard touch
      stringBufferBanks[model][freqKey].push(
        renderPhysicalString(audioCtx, freq, dur, {
          instrumentType: model,
          velocity: 0.88,
          pluckPos: 0.17,
          brightness: 0.50,
          detuneCents: 0.0
        })
      );

      // Variation 1: Firmer attack
      stringBufferBanks[model][freqKey].push(
        renderPhysicalString(audioCtx, freq, dur, {
          instrumentType: model,
          velocity: 0.94,
          pluckPos: 0.15,
          brightness: 0.62,
          detuneCents: 0.8
        })
      );

      // Variation 2: Softer touch
      stringBufferBanks[model][freqKey].push(
        renderPhysicalString(audioCtx, freq, dur, {
          instrumentType: model,
          velocity: 0.80,
          pluckPos: 0.19,
          brightness: 0.40,
          detuneCents: -0.8
        })
      );
    }
  }

  console.log(`[Guitar Engine] Synthesized multi-preset physical string soundbanks in memory.`);
}

function playChord(chordId) {
  ensureAudioCtx();

  pendingStrumTimers.forEach(id => clearTimeout(id));
  pendingStrumTimers = [];

  const now = audioCtx.currentTime;

  // 1. If MUTE (chord 0 / Fist)
  if (chordId === 0 || !CHORD_CONFIG[chordId]) {
    activeVoices.forEach(voice => voice.fadeAndDispose(0.065));
    return;
  }

  // 2. If changing to a new chord:
  const releaseSec = SOUND_SETTINGS.ambientSustain ? 0.15 : 0.075;
  activeVoices.forEach(voice => voice.fadeAndDispose(releaseSec));

  const cfg = CHORD_CONFIG[chordId];
  if (!cfg.freqs || cfg.freqs.length === 0) return;

  const strumStart = now + 0.004;
  let curDelaySec = 0;
  roundRobinIndex = (roundRobinIndex + 1) % 3;

  const nStrings = cfg.freqs.length;
  const activePreset = GUITAR_PRESETS[SOUND_SETTINGS.preset] || GUITAR_PRESETS.nylon;
  const bank = stringBufferBanks[activePreset.model] || stringBufferBanks.nylon;

  const effectiveSustain = SOUND_SETTINGS.ambientSustain
    ? Math.max(SOUND_SETTINGS.sustainScale, 1.8)
    : SOUND_SETTINGS.sustainScale;

  for (let sIdx = 0; sIdx < nStrings; sIdx++) {
    const freq = cfg.freqs[sIdx];
    const freqKey = freq.toFixed(2);
    const variations = bank[freqKey] || (stringBufferBanks.nylon && stringBufferBanks.nylon[freqKey]);
    if (!variations || variations.length === 0) continue;

    const stringIndex = (cfg.stringIndices && cfg.stringIndices[sIdx] !== undefined)
      ? cfg.stringIndices[sIdx]
      : sIdx;

    const stringStartTime = strumStart + curDelaySec;
    const varIdx = (roundRobinIndex + sIdx) % variations.length;
    const buffer = variations[varIdx];

    const src = audioCtx.createBufferSource();
    src.buffer = buffer;

    const microDetuneCents = (Math.random() * 2 - 1) * 1.2;
    src.playbackRate.value = Math.pow(2, microDetuneCents / 1200);

    const posInChord = sIdx / Math.max(1, nStrings - 1);
    const baseVel = 0.90 - Math.sin(posInChord * Math.PI) * 0.12;
    const velJitter = (Math.random() * 0.08 - 0.04);
    const stringGain = Math.max(0.65, Math.min(1.0, baseVel + velJitter)) * 0.42;

    const voiceGain = audioCtx.createGain();
    voiceGain.gain.setValueAtTime(stringGain, stringStartTime);

    // Dynamic sustain decay envelope
    const baseDur = buffer.duration;
    const noteLifetime = baseDur * effectiveSustain;
    voiceGain.gain.setTargetAtTime(0.0001, stringStartTime + 0.08, noteLifetime * 0.32);

    // Subtle Stereo Panning per string (-0.14 on bass to +0.14 on high treble)
    let pannerNode = null;
    if (audioCtx.createStereoPanner) {
      pannerNode = audioCtx.createStereoPanner();
      const panVal = ((sIdx / Math.max(1, nStrings - 1)) * 2 - 1) * 0.14;
      pannerNode.pan.setValueAtTime(panVal, stringStartTime);
    }

    // Connect voice:
    src.connect(voiceGain);
    if (pannerNode) {
      voiceGain.connect(pannerNode);
      pannerNode.connect(effectsChain.voiceBus);
    } else {
      voiceGain.connect(effectsChain.voiceBus);
    }

    // Cap polyphony
    if (activeVoices.size >= 28) {
      const oldest = activeVoices.values().next().value;
      if (oldest) oldest.fadeAndDispose(0.04);
    }

    const voice = {
      source: src,
      gainNode: voiceGain,
      panner: pannerNode,
      fadeAndDispose: (fadeTime = 0.075) => {
        try {
          const t = audioCtx.currentTime;
          voiceGain.gain.cancelScheduledValues(t);
          voiceGain.gain.setValueAtTime(voiceGain.gain.value, t);
          voiceGain.gain.exponentialRampToValueAtTime(0.0001, t + fadeTime);
          setTimeout(() => {
            try {
              src.stop();
              src.disconnect();
              voiceGain.disconnect();
              if (pannerNode) pannerNode.disconnect();
            } catch (_) {}
            activeVoices.delete(voice);
          }, Math.round((fadeTime + 0.015) * 1000));
        } catch (_) {
          activeVoices.delete(voice);
        }
      }
    };

    src.onended = () => voice.fadeAndDispose(0.01);
    activeVoices.add(voice);

    src.start(stringStartTime);

    // Schedule visual string vibration
    const delayMs = Math.round(curDelaySec * 1000);
    const timerId = setTimeout(() => {
      kickStringIndex(stringIndex);
    }, delayMs);
    pendingStrumTimers.push(timerId);

    const stepMs = 23 + Math.floor(Math.random() * 6);
    curDelaySec += stepMs / 1000;
  }
}

// ── Sound Settings Controls ────────────────────────────────────────────────
function applySoundPreset(presetId) {
  const p = GUITAR_PRESETS[presetId];
  if (!p || !effectsChain || !audioCtx) return;

  SOUND_SETTINGS.preset = presetId;
  SOUND_SETTINGS.reverbWet = p.reverbWet;
  SOUND_SETTINGS.sustainScale = p.sustainScale;
  SOUND_SETTINGS.ambientSustain = p.ambientMode;

  const now = audioCtx.currentTime;

  effectsChain.eqLowShelf.gain.setTargetAtTime(p.bodyLowGain, now, 0.04);
  effectsChain.eqPresence.gain.setTargetAtTime(p.presenceGain, now, 0.04);
  effectsChain.eqToneBrightness.gain.setTargetAtTime(p.toneBrightness + SOUND_SETTINGS.toneOffset, now, 0.04);

  effectsChain.compressor.threshold.setTargetAtTime(p.compThreshold, now, 0.04);
  effectsChain.compressor.ratio.setTargetAtTime(p.compRatio, now, 0.04);

  if (p.id === "ambient" && ambientImpulseBuffer) {
    effectsChain.convolver.buffer = ambientImpulseBuffer;
  } else if (roomImpulseBuffer) {
    effectsChain.convolver.buffer = roomImpulseBuffer;
  }

  effectsChain.dryGain.gain.setTargetAtTime(1.0 - p.reverbWet, now, 0.04);
  effectsChain.wetGain.gain.setTargetAtTime(p.reverbWet, now, 0.04);

  syncStudioUI();
}

function updateMasterVolume(val) {
  const v = Math.max(0, Math.min(100, val)) / 100;
  SOUND_SETTINGS.volume = v;
  if (effectsChain && audioCtx) {
    effectsChain.masterGain.gain.setTargetAtTime(v, audioCtx.currentTime, 0.03);
  }
  const el = document.getElementById("val-volume");
  if (el) el.textContent = `${Math.round(v * 100)}%`;
}

function updateSustain(val) {
  const scale = Math.max(50, Math.min(220, val)) / 100;
  SOUND_SETTINGS.sustainScale = scale;
  const el = document.getElementById("val-sustain");
  if (el) el.textContent = `${scale.toFixed(2)}x`;
}

function updateReverbWet(val) {
  const wet = Math.max(0, Math.min(50, val)) / 100;
  SOUND_SETTINGS.reverbWet = wet;
  if (effectsChain && audioCtx) {
    const now = audioCtx.currentTime;
    effectsChain.dryGain.gain.setTargetAtTime(1.0 - wet, now, 0.03);
    effectsChain.wetGain.gain.setTargetAtTime(wet, now, 0.03);
  }
  const el = document.getElementById("val-reverb");
  if (el) el.textContent = `${Math.round(wet * 100)}%`;
}

function updateToneBrightness(val) {
  const dB = Math.max(-60, Math.min(60, val)) / 10;
  SOUND_SETTINGS.toneOffset = dB;
  const p = GUITAR_PRESETS[SOUND_SETTINGS.preset] || GUITAR_PRESETS.nylon;
  const totalDb = p.toneBrightness + dB;
  if (effectsChain && audioCtx) {
    effectsChain.eqToneBrightness.gain.setTargetAtTime(totalDb, audioCtx.currentTime, 0.03);
  }
  const el = document.getElementById("val-tone");
  if (el) el.textContent = `${dB >= 0 ? "+" : ""}${dB.toFixed(1)} dB`;
}

function toggleAmbientSustain(forceState) {
  const newState = (typeof forceState === "boolean") ? forceState : !SOUND_SETTINGS.ambientSustain;
  SOUND_SETTINGS.ambientSustain = newState;

  if (newState) {
    if (effectsChain && audioCtx) {
      if (ambientImpulseBuffer) effectsChain.convolver.buffer = ambientImpulseBuffer;
      const wet = Math.max(SOUND_SETTINGS.reverbWet, 0.32);
      effectsChain.dryGain.gain.setTargetAtTime(1.0 - wet, audioCtx.currentTime, 0.04);
      effectsChain.wetGain.gain.setTargetAtTime(wet, audioCtx.currentTime, 0.04);
    }
  } else {
    const p = GUITAR_PRESETS[SOUND_SETTINGS.preset] || GUITAR_PRESETS.nylon;
    if (effectsChain && audioCtx) {
      if (p.id !== "ambient" && roomImpulseBuffer) effectsChain.convolver.buffer = roomImpulseBuffer;
      effectsChain.dryGain.gain.setTargetAtTime(1.0 - SOUND_SETTINGS.reverbWet, audioCtx.currentTime, 0.04);
      effectsChain.wetGain.gain.setTargetAtTime(SOUND_SETTINGS.reverbWet, audioCtx.currentTime, 0.04);
    }
  }

  syncStudioUI();
}

function syncStudioUI() {
  const p = GUITAR_PRESETS[SOUND_SETTINGS.preset] || GUITAR_PRESETS.nylon;

  const descEl = document.getElementById("active-preset-desc");
  if (descEl) descEl.textContent = p.desc;

  document.querySelectorAll(".preset-pill").forEach(btn => {
    const isAct = (btn.dataset.preset === SOUND_SETTINGS.preset);
    btn.classList.toggle("active", isAct);
    btn.setAttribute("aria-checked", isAct ? "true" : "false");
  });

  const ambBtn = document.getElementById("btn-ambient-quick");
  if (ambBtn) ambBtn.classList.toggle("active", SOUND_SETTINGS.ambientSustain);

  const slVol = document.getElementById("slider-volume");
  if (slVol) {
    slVol.value = Math.round(SOUND_SETTINGS.volume * 100);
    const vEl = document.getElementById("val-volume");
    if (vEl) vEl.textContent = `${slVol.value}%`;
  }

  const slSus = document.getElementById("slider-sustain");
  if (slSus) {
    slSus.value = Math.round(SOUND_SETTINGS.sustainScale * 100);
    const sEl = document.getElementById("val-sustain");
    if (sEl) sEl.textContent = `${SOUND_SETTINGS.sustainScale.toFixed(2)}x`;
  }

  const slRev = document.getElementById("slider-reverb");
  if (slRev) {
    slRev.value = Math.round(SOUND_SETTINGS.reverbWet * 100);
    const rEl = document.getElementById("val-reverb");
    if (rEl) rEl.textContent = `${slRev.value}%`;
  }

  const slTone = document.getElementById("slider-tone");
  if (slTone) {
    slTone.value = Math.round(SOUND_SETTINGS.toneOffset * 10);
    const tEl = document.getElementById("val-tone");
    if (tEl) tEl.textContent = `${SOUND_SETTINGS.toneOffset >= 0 ? "+" : ""}${SOUND_SETTINGS.toneOffset.toFixed(1)} dB`;
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

        handleDetectedCount(count);
      } else {
        updateFingerPills([false, false, false, false, false]);
        handleDetectedCount(0);
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

function handleDetectedCount(count) {
  if (count === currentChord) {
    candidateChord = count;
    candidateCount = 0;
    return;
  }

  if (count === candidateChord) {
    candidateCount++;
    if (candidateCount >= STABILITY_THRESHOLD) {
      setChord(candidateChord);
      candidateCount = 0;
    }
  } else {
    candidateChord = count;
    candidateCount = 1;
  }
}

function setChord(id, immediate = false) {
  if (id === currentChord && !immediate) return;
  currentChord = id;
  lastChord = currentChord;
  candidateChord = currentChord;
  candidateCount = 0;
  playChord(currentChord);
  updateChordUI(currentChord);
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
function kickStringIndex(idx) {
  if (idx >= 0 && idx < vibAmps.length) {
    vibAmps[idx] = 1.0;
  }
}

function kickStrings(stringIndices) {
  if (Array.isArray(stringIndices)) {
    stringIndices.forEach(idx => kickStringIndex(idx));
  } else {
    vibAmps = vibAmps.map(() => 1.0);
  }
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
    if      (t.includes("e minor") || t.includes("em")     || t === "one") setChord(1, true);
    else if (t.includes("a minor") || t.includes("am")     || t === "two") setChord(2, true);
    else if (t.includes("c major") || t === "c" || t === "see"|| t === "three") setChord(3, true);
    else if (t.includes("d major") || t === "d" || t === "dee"|| t === "four")  setChord(4, true);
    else if (t.includes("g major") || t === "g" || t === "gee"|| t === "five")  setChord(5, true);
    else if (t.includes("stop")   || t.includes("mute") || t.includes("silence")) setChord(0, true);
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
      setChord(parseInt(btn.dataset.chord), true);
    };
    btn.addEventListener("click",      trigger);
    btn.addEventListener("touchstart", trigger, { passive: false });
  });

  // Sound Studio Preset Pills
  document.querySelectorAll(".preset-pill").forEach(btn => {
    btn.addEventListener("click", (e) => {
      e.preventDefault();
      ensureAudioCtx();
      applySoundPreset(btn.dataset.preset);
      showToast(`Switched to ${GUITAR_PRESETS[btn.dataset.preset].name}`, "info");
    });
  });

  // Sound Studio Sliders
  const slVol = document.getElementById("slider-volume");
  if (slVol) {
    slVol.addEventListener("input", (e) => {
      ensureAudioCtx();
      updateMasterVolume(parseFloat(e.target.value));
    });
  }

  const slSus = document.getElementById("slider-sustain");
  if (slSus) {
    slSus.addEventListener("input", (e) => {
      ensureAudioCtx();
      updateSustain(parseFloat(e.target.value));
    });
  }

  const slRev = document.getElementById("slider-reverb");
  if (slRev) {
    slRev.addEventListener("input", (e) => {
      ensureAudioCtx();
      updateReverbWet(parseFloat(e.target.value));
    });
  }

  const slTone = document.getElementById("slider-tone");
  if (slTone) {
    slTone.addEventListener("input", (e) => {
      ensureAudioCtx();
      updateToneBrightness(parseFloat(e.target.value));
    });
  }

  // Ambient Quick Button
  const ambBtn = document.getElementById("btn-ambient-quick");
  if (ambBtn) {
    ambBtn.addEventListener("click", (e) => {
      e.preventDefault();
      ensureAudioCtx();
      toggleAmbientSustain();
      showToast(SOUND_SETTINGS.ambientSustain ? "✨ Ambient Sustain Enabled" : "Ambient Sustain Disabled", "info");
    });
  }

  // Collapse / Expand Sound Studio Header
  const collapseBtn = document.getElementById("btn-collapse-studio");
  const headerToggle = document.getElementById("studio-header-toggle");
  const studioPanel = document.getElementById("sound-studio-panel");
  if (headerToggle && studioPanel) {
    headerToggle.addEventListener("click", (e) => {
      if (e.target.closest("#btn-ambient-quick")) return;
      studioPanel.classList.toggle("collapsed");
      const isExpanded = !studioPanel.classList.contains("collapsed");
      if (collapseBtn) collapseBtn.setAttribute("aria-expanded", isExpanded ? "true" : "false");
    });
  }

  window.addEventListener("keydown", (e) => {
    if (e.ctrlKey || e.altKey || e.metaKey) return;
    const k = e.key.toLowerCase();
    if      (k === "1" || k === "e") setChord(1, true);
    else if (k === "2" || k === "a") setChord(2, true);
    else if (k === "3" || k === "c") setChord(3, true);
    else if (k === "4" || k === "d") setChord(4, true);
    else if (k === "5" || k === "g") setChord(5, true);
    else if (k === "0" || k === "q" || k === "m") setChord(0, true);
    else if (k === "v") toggleVoice();
  });
}
