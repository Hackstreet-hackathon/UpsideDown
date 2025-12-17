import "./App.css";
import { useState, useEffect, useRef } from "react";

import {
  setPanic,
  setNormal,
  setBpm as setEngineBpm,
  killSwitch,
  setOnHeartbeat,
  setOnStateChange
} from "./app.js";

function App() {
  const [isTextVisible, setIsTextVisible] = useState(false);
  const [isFighting, setIsFighting] = useState(false);
  const [kills, setKills] = useState(0);
  const [battleMessage, setBattleMessage] = useState("");
  const [bpm, setBpmState] = useState(60);
  const [mode, setMode] = useState("NORMAL");
  const [showCamera, setShowCamera] = useState(false);
  const [isCutscene, setIsCutscene] = useState(false);
  const [typewriterText, setTypewriterText] = useState("");

  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const radarRef = useRef(null); // Fix: Use Ref instead of querySelector
  const radarCountRef = useRef(0);
  const isFightingRef = useRef(false);
  
  const handleAttack = (weaponName) => {
    const chance = Math.random();

    if (chance <= 0.25) {
      // 🏆 WIN LOGIC
      setKills((prev) => prev + 1);
      setIsFighting(false);
      isFightingRef.current = false; // Unblocks the radar to spawn new blips
      setNormal(); // Returns heartbeat to 60 BPM
      
      if (radarRef.current) {
        radarRef.current.classList.remove("active-blip");
      }
      setBattleMessage(`${weaponName} SUCCESSFUL! DEMOGORGON SLAIN.`);
      setTimeout(() => setBattleMessage(""), 3000);
    } else {
      // ❌ FAIL LOGIC
      setBattleMessage(`${weaponName} FAILED! TRY AGAIN!`);
      setTimeout(() => setBattleMessage(""), 1000);
    }
  };
const battleTimeoutRef = useRef(null);
  useEffect(() => {
    // Check if functions exist before calling to avoid crash
    if (setOnHeartbeat) setOnHeartbeat((currentBpm) => setBpmState(currentBpm));
    if (setOnStateChange) setOnStateChange((state) => setMode(state));
  }, []);

  /* 📷 Camera logic */
  useEffect(() => {
    if (!showCamera) return;
    async function startCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "environment" },
          audio: false
        });
        streamRef.current = stream;
        if (videoRef.current) videoRef.current.srcObject = stream;
      } catch (err) {
        try {
          const fallback = await navigator.mediaDevices.getUserMedia({ video: true });
          streamRef.current = fallback;
          if (videoRef.current) videoRef.current.srcObject = fallback;
        } catch (e) { console.error("Camera error", e); }
      }
    }
    startCamera();
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, [showCamera]);
useEffect(() => {
const triggerCutscene = () => {
    setIsFighting(false);
    isFightingRef.current = false;
    
    setIsCutscene(true); // Black screen starts fading in (2s)
    setTypewriterText("NOW, I HAVE YOU CHILD, I, AM YOUR MASTER HENCEFORTH"); 

    // 1. WAIT for the screen to be fully black
    setTimeout(() => {
      setIsTextVisible(true); // White text fades IN slowly

      // 2. HOLD the text so it can be read
      setTimeout(() => {
        setIsTextVisible(false); // White text fades OUT
        
        // 3. Start returning to the dashboard
        setTimeout(() => {
          setIsCutscene(false);
          radarCountRef.current = 0;
          // Clear string after it's fully invisible
          setTimeout(() => setTypewriterText(""), 2500);
        }, 2500);
      }, 4000); // How long the text stays visible
    }, 2000); // Must match your CSS blackout transition
  };

  const spawnBlip = () => {
    const blip = radarRef.current;
    // Don't spawn if radar is missing OR if we are currently in a fight
    if (!blip || isFightingRef.current) return; 

    radarCountRef.current += 1;
    if (radarCountRef.current === 3) {
      triggerCutscene();
      return;
    }

    const angle = Math.random() * Math.PI * 2;
    const startX = 50 + Math.cos(angle) * 45;
    const startY = 50 + Math.sin(angle) * 45;

    blip.style.setProperty("--top", `${startY}%`);
    blip.style.setProperty("--left", `${startX}%`);
    blip.classList.remove("active-blip");

    setTimeout(() => {
      blip.classList.add("active-blip");
      blip.style.setProperty("--top", "48%");
      blip.style.setProperty("--left", "48%");

      // This triggers when the blip reaches the center
      battleTimeoutRef.current = setTimeout(() => {
        setIsFighting(true);
        isFightingRef.current = true;
        setPanic(); 
      }, 15000); 
    }, 100);
  };

  const interval = setInterval(spawnBlip, 20000); // Increased interval to allow for fight time
  spawnBlip();
  
  return () => {
    clearInterval(interval);
    clearTimeout(battleTimeoutRef.current);
  };
}, []);
  /* 📡 Radar Logic (Crash-Proof) */
//   useEffect(() => {
//  const triggerCutscene = () => {
//   setIsCutscene(true);
//   // The text is already set, we just control its visibility
//   setTypewriterText("NOW, I HAVE YOU CHILD, I AM YOUR MASTER HENCEFORTH");

//   // 1. Wait for screen to be fully black (2s)
//   setTimeout(() => {
//     setIsTextVisible(true); // Fades the white text IN

//     // 2. Keep the text visible for 4 seconds
//     setTimeout(() => {
//       setIsTextVisible(false); // Fades the white text OUT
      
//       // 3. Start fading out the black screen shortly after text starts fading
//       setTimeout(() => {
//         setIsCutscene(false);
//         radarCountRef.current = 0;
//         // Clear text only after everything is hidden
//         setTimeout(() => setTypewriterText(""), 2000);
//       }, 5500);
//     }, 6000);
//   }, 2000); 
// };
//     const spawnBlip = () => {
//       const blip = radarRef.current;
//       if (!blip) return; // Prevent crash if radar isn't rendered

//       radarCountRef.current += 1;
//       if (radarCountRef.current === 3) {
//         triggerCutscene();
//         return;
//       }

//       const angle = Math.random() * Math.PI * 2;
//       const startX = 50 + Math.cos(angle) * 45;
//       const startY = 50 + Math.sin(angle) * 45;

//       blip.style.setProperty("--top", `${startY}%`);
//       blip.style.setProperty("--left", `${startX}%`);
//       blip.classList.remove("active-blip");

//       setTimeout(() => {
//         blip.classList.add("active-blip");
//         blip.style.setProperty("--top", "48%");
//         blip.style.setProperty("--left", "48%");
//       }, 100);
//     };

//     const interval = setInterval(spawnBlip, 12000);
//     spawnBlip();
//     return () => clearInterval(interval);
//   }, []);

  return (
    <div className="app-container">
   <div className={`blackout-overlay ${isCutscene ? "active" : ""}`}>
  <p className={`fade-text ${isTextVisible ? "visible" : ""}`}>
    {typewriterText}
  </p>
</div>

      <h1 className="heading">Upside Down</h1>

      {/* Added ref={radarRef} here */}
      <div ref={radarRef} className="radar"></div>

      <div className="ui-controls">
        <button className="portal-button" onClick={() => setShowCamera(true)}>
          SEARCH FOR PORTAL
        </button>

        <div className="fight">
          <button className="btn-3d" onClick={() => handleAttack("FLAMETHROWER")}>🔥</button>
          <button className="btn-3d" onClick={() => handleAttack("SOUND")}>sound</button>
          <button className="btn-3d" onClick={() => handleAttack("GUN")}>gun</button>
        </div>

        <div className="controls">
          <button className="dir-btn">LEFT</button>
          <button className="dir-btn" onClick={killSwitch}>KILL</button>
        </div>

        <div className="stats">
          💗 <strong>{bpm} BPM</strong> | 🧠 <strong>{mode}</strong>
        </div>
      </div>

      {showCamera && (
  <div className="camera-overlay">
    <div className="camera-frame">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        className="camera-video"
      />
      <button
        className="close-btn"
        onClick={() => setShowCamera(false)}
      >
        ✖ CLOSE
      </button>
    </div>
  </div>
)}
    </div>
  );
}

export default App;