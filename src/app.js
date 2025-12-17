// heartbeat.js

let panicMode = false;
let heartbeatInterval = null;

// CONFIG
const NORMAL_BPM = 60;
const PANIC_BPM = 140;

// AUDIO
const music = new Audio("/music.mp3");
music.loop = true;
music.volume = 0.8;

// INTERNAL CALLBACKS
let heartbeatListener = () => {};
let stateListener = () => {};

// PUBLIC REGISTRATION FUNCTIONS
function setOnHeartbeat(fn) {
  heartbeatListener = fn;
}

function setOnStateChange(fn) {
  stateListener = fn;
}

// Utils
function bpmToMs(bpm) {
  return 60000 / bpm;
}

// Heartbeat loop
function startHeartbeat(bpm) {
  stopHeartbeat();

  heartbeatInterval = setInterval(() => {
    heartbeatListener(bpm);
  }, bpmToMs(bpm));
}

function stopHeartbeat() {
  if (heartbeatInterval) {
    clearInterval(heartbeatInterval);
    heartbeatInterval = null;
  }
}

// Smooth calm-down
function calmDown() {
  let bpm = PANIC_BPM;

  const calmInterval = setInterval(() => {
    bpm -= 10;

    if (bpm <= NORMAL_BPM) {
      bpm = NORMAL_BPM;
      clearInterval(calmInterval);
    }

    startHeartbeat(bpm);
  }, 300);
}

// Toggle panic
function togglePanic() {
  panicMode = !panicMode;

  if (panicMode) {
    startHeartbeat(PANIC_BPM);
    music.play();
    stateListener("PANIC");
  } else {
    calmDown();
    music.pause();
    music.currentTime = 0;
    stateListener("NORMAL");
  }
}

// Kill switch
function killSwitch() {
  panicMode = false;
  stopHeartbeat();
  music.pause();
  music.currentTime = 0;
  startHeartbeat(NORMAL_BPM);
  stateListener("KILLED");
}

// Keyboard trigger
document.addEventListener("keydown", (e) => {
  if (e.key.toLowerCase() === "t") {
    togglePanic();
  }
});

// Initial state
startHeartbeat(NORMAL_BPM);

// PUBLIC API
export {
  togglePanic,
  killSwitch,
  setOnHeartbeat,
  setOnStateChange
};
