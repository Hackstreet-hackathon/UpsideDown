import './App.css'
import { useEffect } from 'react';

function App() {
  useEffect(() => {
  const blip = document.querySelector('.radar');
  
  const spawnBlip = () => {
    // 1. Pick a random angle (0 to 360 degrees)
    const angle = Math.random() * Math.PI * 2;
    // 2. Start at the far edge (approx 40-45% away from center)
    const startX = 50 + Math.cos(angle) * 45;
    const startY = 50 + Math.sin(angle) * 45;

    // Set initial random edge position instantly
    blip.style.setProperty('--top', `${startY}%`);
    blip.style.setProperty('--left', `${startX}%`);
    blip.classList.remove('active-blip');

    // Small timeout to allow the "jump" to the edge to happen without animation
    setTimeout(() => {
      blip.classList.add('active-blip');
      // Move to center
      blip.style.setProperty('--top', `48%`);
      blip.style.setProperty('--left', `48%`);
    }, 100);
  };

  // Run every 12 seconds (10s for move + 2s pause)
  const interval = setInterval(spawnBlip, 12000);
  spawnBlip(); // Run once immediately

  return () => clearInterval(interval);
}, []);
  return (
    <>
      <h1 className = "heading">Upside Down</h1>
      <div className="radar">
        
      </div>
      <div>
        <button className="portal-button">SEARCH FOR PORTAL</button>
      </div>
      <div class="fight">
        <button class="btn-3d">
 <img src='/src/assets/fire.png' class="btn-icon buttonimg" /></button>
        <button class="btn-3d">CENTER</button>
        <button class="btn-3d">RIGHT</button>
      </div>
      <div class="controls">
        <button class="dir-btn left">LEFT</button>
        <button class="dir-btn right">RIGHT</button>
      </div>
      <div>
        heartrate
      </div>
    </>
  )
}

export default App
