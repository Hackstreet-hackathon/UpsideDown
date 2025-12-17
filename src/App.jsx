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
  const [bpm, setBpmState] = useState(60);
  const [mode, setMode] = useState("NORMAL");
  const [showCamera, setShowCamera] = useState(false);

  const videoRef = useRef(null);
  const streamRef = useRef(null);

  /* ❤️ Heartbeat engine listeners */
  useEffect(() => {
    setOnHeartbeat((currentBpm) => {
      setBpmState(currentBpm);
    });

    setOnStateChange((state) => {
      setMode(state);
    });
  }, []);

  /* 📷 Camera control (BACK camera + cleanup) */
  useEffect(() => {
    if (!showCamera) return;

    async function startCamera() {
      try {
        // Try back camera first
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: { exact: "environment" } },
          audio: false
        });

        streamRef.current = stream;
        videoRef.current.srcObject = stream;
      } catch (err) {
        // Fallback if environment camera not supported
        const stream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: false
        });

        streamRef.current = stream;
        videoRef.current.srcObject = stream;
      }
    }

    startCamera();

    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
        streamRef.current = null;
      }
    };
  }, [showCamera]);

  /* 🌫️ Blur background when camera opens */
  useEffect(() => {
    if (showCamera) {
      document.body.classList.add("camera-open");
    } else {
      document.body.classList.remove("camera-open");
    }
  }, [showCamera]);

  /* 📡 Radar animation */
  useEffect(() => {
    const blip = document.querySelector(".radar");
    if (!blip) return;

    const spawnBlip = () => {
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
      }, 100);
    };

    const interval = setInterval(spawnBlip, 12000);
    spawnBlip();

    return () => clearInterval(interval);
  }, []);

  return (
    <>
      <h1 className="heading">Upside Down</h1>

      <div className="radar"></div>

      <div>
        <button
          className="portal-button"
          onClick={() => setShowCamera(true)}
        >
          SEARCH FOR PORTAL
        </button>
      </div>

      <div className="fight">
        <button className="btn-3d" onClick={setPanic}>
          🔥 PANIC
        </button>

        <button
          className="btn-3d"
          onClick={() => setEngineBpm(100)}
        >
          BPM 100
        </button>

        <button className="btn-3d" onClick={setNormal}>
          CALM
        </button>
      </div>

      <div className="controls">
        <button className="dir-btn left">LEFT</button>
        <button className="dir-btn right" onClick={killSwitch}>
          KILL
        </button>
      </div>

      <div>
        💗 <strong>{bpm} BPM</strong><br />
        🧠 <strong>{mode}</strong>
      </div>

      {/* 🔮 CAMERA OVERLAY */}
      {showCamera && (
        <div className="camera-overlay">
          <div className="camera-container">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              className="camera-feed"
            />

            <button
              className="close-camera"
              onClick={() => setShowCamera(false)}
            >
              ✖ CLOSE
            </button>
          </div>
        </div>
      )}
    </>
  );
}

export default App;
