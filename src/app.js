// heartbeat.js

let heartbeatInterval = null;
let audioUnlocked = false;

// CONFIG
const NORMAL_BPM = 60;
const MID_BPM = 100;
const HIGH_BPM = 140;


// 🎵 AUDIO (always normal sound)
const normalSound = new Audio("/normal.mp3");
normalSound.loop = true;
normalSound.volume = 0.6;

// CALLBACKS
let heartbeatListener = () => {};
let stateListener = () => {};

// REGISTRATION
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

// 🔀 BPM flicker (organic feel)
function flickerBpm(targetBpm) {
  let range;

  if (targetBpm <= NORMAL_BPM) range = 5;
  else if (targetBpm < HIGH_BPM) range = 10;
  else range = 15;

  const variation =
    Math.floor(Math.random() * (range * 2 + 1)) - range;

  return Math.max(40, targetBpm + variation);
}

// ❤️ Heartbeat engine
function startHeartbeat(targetBpm) {
  stopHeartbeat();

  heartbeatInterval = setInterval(() => {
    const flickered = flickerBpm(targetBpm);
    heartbeatListener(flickered);
  }, bpmToMs(targetBpm));
}

function stopHeartbeat() {
  if (heartbeatInterval) {
    clearInterval(heartbeatInterval);
    heartbeatInterval = null;
  }
}

// 🔓 Unlock audio ONCE (first interaction)
function unlockAudio() {
  if (audioUnlocked) return;
  audioUnlocked = true;

  normalSound.play().catch(() => {});
}

// 🎚 PUBLIC CONTROLS (NO AUDIO TOGGLING)

function setBpm(targetBpm) {
  unlockAudio();
  startHeartbeat(targetBpm);

  if (targetBpm <= NORMAL_BPM) stateListener("NORMAL");
  else if (targetBpm < HIGH_BPM) stateListener("TENSION");
  else stateListener("PANIC");
}

function setNormal() {
  setBpm(NORMAL_BPM);
}


// 🎚 UPDATED PUBLIC CONTROLS

function setPanic() {
  setBpm(HIGH_BPM);
}
// 🌊 Smooth calm-down (BPM only)
function calmDown() {
  let bpm = HIGH_BPM;

  const calmInterval = setInterval(() => {
    bpm -= 10;

    if (bpm <= NORMAL_BPM) {
      clearInterval(calmInterval);
      setNormal();
      return;
    }

    startHeartbeat(bpm);
  }, 400);
}

//fight logic
// const handleAttack = (weaponName) => {
//   // 1. Generate a random number between 0 and 1
//   const chance = Math.random();

//   if (chance <= 0.25) {
//     // 🏆 WIN LOGIC (25% chance)
//     setKills(prev => prev + 1);
//     setIsFighting(false);
//     setBattleMessage(`${weaponName} SUCCESSFUL! DEMOGORGON SLRAIN.`);
//     setNormal(); // Return BPM to normal
    
//     // Clear the radar blip
//     if (radarRef.current) {
//       radarRef.current.classList.remove("active-blip");
//       radarCountRef.current = 0; // Reset radar for next encounter
//     }

//     // Clear message after 3 seconds
//     setTimeout(() => setBattleMessage(""), 3000);
//   } else {
//     // ❌ FAIL LOGIC (75% chance)
//     setBattleMessage(`${weaponName} FAILED! TRY AGAIN!`);
    
//     // Optional: Make the screen shake or flash red on failure
//     setTimeout(() => setBattleMessage(""), 1000);
//   }
// };

// ☠️ KILL SWITCH (heartbeat reset, music continues)
function killSwitch() {
  unlockAudio();
  stopHeartbeat();
  startHeartbeat(NORMAL_BPM);
  stateListener("KILLED");
}

// Any key unlocks audio (browser rule)
document.addEventListener("keydown", unlockAudio);
document.addEventListener("click", unlockAudio);

// Initial silent heartbeat
startHeartbeat(NORMAL_BPM);

// PUBLIC API
export {
  setOnHeartbeat,
  setOnStateChange,
  setBpm,
  setNormal,
  setPanic, 
  calmDown,
  killSwitch
};


