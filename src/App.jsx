import "./App.css";
import { useState, useEffect, useRef } from "react";

import {
  setPanic,
  setNormal,
  playNowSound,
  playFlamethrowerSound,
  playGunSound,
  playHighPitchSound,
  playNormalSound,
  setOnHeartbeat,
  setOnStateChange
} from "./app.js";

function App() {
  const [isTextVisible, setIsTextVisible] = useState(false);
  const [isCutscene, setIsCutscene] = useState(false);
  const [battleMessage, setBattleMessage] = useState("");
  const [bpm, setBpmState] = useState(60);
  const [mode, setMode] = useState("NORMAL");

  const radarRef = useRef(null);
  const battleTimeoutRef = useRef(null);

  /* ===================== CUTSCENE ===================== */
  const triggerCutscene = () => {
    setIsCutscene(true);

    // wait before showing text
    setTimeout(() => {
      setIsTextVisible(true);
    }, 2000);

    // hide text
    setTimeout(() => {
      setIsTextVisible(false);
    }, 8000);

    // exit cutscene
    setTimeout(() => {
      setIsCutscene(false);
      playNormalSound();
      spawnBlip();
    }, 11000);
  };

  /* 🔊 play audio EXACTLY when text appears */
  useEffect(() => {
    if (isTextVisible) {
      playNowSound();
    }
  }, [isTextVisible]);

  /* ===================== RADAR ===================== */
  const spawnBlip = () => {
    const blip = radarRef.current;
    if (!blip || isCutscene) return;

    blip.classList.remove("active-blip");
    blip.style.opacity = "0";

    const angle = Math.random() * Math.PI * 2;
    blip.style.setProperty("--top", `${50 + Math.cos(angle) * 45}%`);
    blip.style.setProperty("--left", `${50 + Math.sin(angle) * 45}%`);

    setTimeout(() => {
      blip.style.opacity = "1";
      blip.classList.add("active-blip");

      clearTimeout(battleTimeoutRef.current);
      battleTimeoutRef.current = setTimeout(() => {
        setPanic();
        playHighPitchSound();
      }, 15000);
    }, 100);
  };

  /* ===================== EFFECTS ===================== */
  useEffect(() => {
    setOnHeartbeat(setBpmState);
    setOnStateChange(setMode);
    spawnBlip();
  }, []);

  /* ===================== UI ===================== */
  return (
    <div className="app-container">
      {/* BLACK SCREEN */}
      <div className={`blackout-overlay ${isCutscene ? "active" : ""}`}>
        <p className={`fade-text ${isTextVisible ? "visible" : ""}`}>
          NOW, I HAVE YOU CHILD, I AM YOUR MASTER HENCEFORTH
        </p>
      </div>

      <h1 className="heading">Upside Down</h1>

      <div ref={radarRef} className="radar" />

      <div className="fight">
        <button className="btn-3d" onClick={() => playFlamethrowerSound()}>
          FIRE
        </button>
        <button className="btn-3d" onClick={() => playHighPitchSound()}>
          SOUND
        </button>
        <button className="btn-3d" onClick={() => playGunSound()}>
          GUN
        </button>
      </div>

      <button className="portal-button" onClick={triggerCutscene}>
        TRIGGER CUTSCENE
      </button>

      <div className="stats">
        💗 {bpm} BPM | 🧠 {mode}
      </div>

      {battleMessage && <div className="battle-message">{battleMessage}</div>}
    </div>
  );
}

export default App;
