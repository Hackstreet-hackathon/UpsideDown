import { useState, useRef } from 'react';
import './App.css';

function App2() {
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const videoRef = useRef(null);

  const startCamera = async () => {
    setIsCameraOpen(true);
    try {
      // Small timeout ensures the DOM element exists before we attach the stream
      setTimeout(async () => {
        const stream = await navigator.mediaDevices.getUserMedia({ 
          video: { facingMode: "environment" } 
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      }, 100);
    } catch (err) {
      console.error("Camera error:", err);
      alert("Please allow camera access.");
      setIsCameraOpen(false);
    }
  };

  const closeCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject;
      const tracks = stream.getTracks();
      tracks.forEach(track => track.stop());
    }
    setIsCameraOpen(false);
  };

  return (
    <div className="app-container">
      {isCameraOpen && (
        <div className="camera-overlay">
          <video ref={videoRef} autoPlay playsInline className="camera-view" />
          <button className="close-portal" onClick={closeCamera}>CLOSE PORTAL</button>
        </div>
      )}

      <h1 className="heading">Upside Down - V2</h1>
      <div className="radar"></div>
      
      <div>
        <button className="portal-button" onClick={startCamera}>
          SEARCH FOR PORTAL
        </button>
      </div>

      <div className="fight">
        <button className="btn-3d">
          <img src='/src/assets/fire.png' className="btn-icon buttonimg" alt="fire" />
        </button>
        <button className="btn-3d">
          <img src='/src/assets/shotgun.png' className="btn-icon buttonimg" alt="shotgun" />
        </button>
        <button className="btn-3d">RIGHT</button>
      </div>

      <div className="controls">
        <button className="dir-btn left">LEFT</button>
        <button className="dir-btn right">RIGHT</button>
      </div>
    </div>
  );
}

export default App2;