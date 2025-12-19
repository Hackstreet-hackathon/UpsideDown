import cv2
import torch
import numpy as np
import time
import math
import random
import base64
import json
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# --- 1. FRONTEND (With Reset Button) ---
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Stranger Things AR - World Lock</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <style>
        body { margin: 0; background: black; overflow: hidden; font-family: 'Courier New', monospace; }
        #container { position: relative; width: 100vw; height: 100vh; }
        
        #localVideo { 
            position: absolute; top: 0; left: 0; width: 100%; height: 100%; 
            object-fit: cover; z-index: 1; 
        }
        
        #portalLayer { 
            position: absolute; top: 0; left: 0; width: 100%; height: 100%; 
            z-index: 2; pointer-events: none; 
        }
        
        /* THE RETICLE UI */
        #uiLayer {
            position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
            z-index: 3; pointer-events: none; display: flex; flex-direction: column; align-items: center;
        }
        
        .reticle-box {
            width: 80px; height: 80px; 
            border: 2px solid rgba(255, 255, 255, 0.3);
            position: relative; transition: all 0.2s;
        }
        
        /* Corner Brackets */
        .reticle-box::before, .reticle-box::after {
            content: ''; position: absolute; width: 20px; height: 20px;
            border: 2px solid white; transition: all 0.2s;
        }
        .reticle-box::before { top: -2px; left: -2px; border-right: 0; border-bottom: 0; }
        .reticle-box::after { bottom: -2px; right: -2px; border-left: 0; border-top: 0; }
        
        #status {
            margin-top: 60px; font-weight: bold; font-size: 14px; 
            color: rgba(255,255,255,0.7); text-shadow: 1px 1px 2px black;
            background: rgba(0,0,0,0.5); padding: 4px 8px; border-radius: 4px;
        }

        /* RESET BUTTON */
        #resetBtn {
            position: absolute; top: 20px; right: 20px; z-index: 10;
            background: rgba(200, 0, 0, 0.6); color: white;
            border: 1px solid rgba(255,255,255,0.5);
            padding: 8px 16px; border-radius: 4px;
            font-weight: bold; cursor: pointer; display: none;
            backdrop-filter: blur(4px);
        }
        #resetBtn:active { background: rgba(255, 0, 0, 0.8); transform: scale(0.95); }

        /* STATES */
        .locking .reticle-box { border-color: #00ff00; transform: scale(0.9); }
        .locking .reticle-box::before, .locking .reticle-box::after { border-color: #00ff00; }
        
        .hidden { opacity: 0; }
    </style>
</head>
<body>
    <div id="container">
        <video id="localVideo" autoplay playsinline muted></video>
        <canvas id="portalLayer"></canvas>
        
        <div id="resetBtn">↻ RESET</div>

        <div id="uiLayer">
            <div id="reticle" class="reticle-box"></div>
            <div id="status">POINT AT WALL</div>
        </div>
    </div>
    <canvas id="sendCanvas" style="display:none;"></canvas>
    
    <script>
        const video = document.getElementById('localVideo');
        const canvas = document.getElementById('portalLayer');
        const ctx = canvas.getContext('2d');
        const sendCanvas = document.getElementById('sendCanvas');
        const sCtx = sendCanvas.getContext('2d');
        
        const uiLayer = document.getElementById('uiLayer');
        const uiReticle = document.getElementById('reticle');
        const uiStatus = document.getElementById('status');
        const resetBtn = document.getElementById('resetBtn');
        
        const protocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
        const ws = new WebSocket(protocol + window.location.host + "/ws");

        let portalImg = new Image();
        let isLocked = false;
        let isVisible = false; 
        
        // Instant Tracking Variables (No Smoothing to prevent lag drag)
        let current = { x: 0.5, y: 0.5, w: 0, h: 0 };
        let target = { x: 0.5, y: 0.5, w: 0, h: 0 };

        // --- RESET LOGIC ---
        resetBtn.onclick = () => {
            // Send special command to server
            ws.send(JSON.stringify({action: "reset"}));
            
            // Force local reset
            isLocked = false;
            isVisible = false;
            uiLayer.classList.remove("hidden");
            uiReticle.classList.remove("locking");
            uiStatus.innerText = "RESETTING...";
            resetBtn.style.display = "none";
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            if (data.status === "tracking") {
                isLocked = true;
                isVisible = data.visible; 
                resetBtn.style.display = "block"; // Show button when locked
                
                if (isVisible) {
                    uiLayer.classList.add("hidden"); 
                    
                    target.x = data.nx;
                    target.y = data.ny;
                    target.w = data.nw;
                    target.h = data.nh;
                    if (data.tex) portalImg.src = "data:image/png;base64," + data.tex;
                } else {
                    uiLayer.classList.add("hidden"); 
                }
                
            } else {
                // Scanning Mode
                isLocked = false;
                isVisible = false;
                uiLayer.classList.remove("hidden");
                resetBtn.style.display = "none";
                
                if (data.status === "locking") {
                    uiLayer.classList.add("locking");
                    uiStatus.innerText = "HOLD STEADY...";
                    uiStatus.style.color = "#00ff00";
                } else {
                    uiLayer.classList.remove("locking");
                    uiStatus.innerText = "SCANNING...";
                    uiStatus.style.color = "white";
                }
            }
        };

        function render() {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            if (isLocked && isVisible && portalImg.complete) {
                // INSTANT ASSIGNMENT (Fixes sliding feeling)
                current.x = target.x;
                current.y = target.y;
                current.w = target.w;
                current.h = target.h;

                // Aspect Ratio Logic (Cover)
                const screenAspect = canvas.width / canvas.height;
                const videoAspect = 4 / 3;
                let renderW, renderH, offsetX, offsetY;

                if (screenAspect > videoAspect) {
                    renderW = canvas.width;
                    renderH = canvas.width / videoAspect;
                    offsetX = 0;
                    offsetY = (canvas.height - renderH) / 2;
                } else {
                    renderW = canvas.height * videoAspect;
                    renderH = canvas.height;
                    offsetX = (canvas.width - renderW) / 2;
                    offsetY = 0;
                }

                const drawX = offsetX + (current.x * renderW);
                const drawY = offsetY + (current.y * renderH);
                const drawW = current.w * renderW;
                const drawH = current.h * renderH;

                ctx.drawImage(portalImg, drawX - drawW/2, drawY - drawH/2, drawW, drawH);
            }
            requestAnimationFrame(render);
        }
        render();

        navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: "environment", width: { ideal: 640 }, height: { ideal: 480 } } 
        }).then(stream => {
            video.srcObject = stream;
            setInterval(() => {
                if (ws.readyState === WebSocket.OPEN) {
                    sendCanvas.width = 480; sendCanvas.height = 360;
                    sCtx.drawImage(video, 0, 0, 480, 360);
                    // Quality 0.7 for better feature tracking
                    ws.send(sendCanvas.toDataURL("image/jpeg", 0.7));
                }
            }, 66);
        });
    </script>
</body>
</html>
"""

# --- 2. CONFIGURATION ---
MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512" 
CLASS_WALL = 0
RESIZE_WIDTH = 480 
GROWTH_DURATION = 15.0 # Slow, dramatic opening
BASE_SIZE = 65 # Smaller, more realistic size

# --- 3. STABILIZER (Server-Side) ---
class CoordinateSmoother:
    def __init__(self, alpha=0.6):
        self.x = None
        self.y = None
        self.s = None
        self.alpha = alpha 

    def update(self, nx, ny, ns):
        if self.x is None:
            self.x, self.y, self.s = nx, ny, ns
        else:
            self.x = self.x * self.alpha + nx * (1 - self.alpha)
            self.y = self.y * self.alpha + ny * (1 - self.alpha)
            self.s = self.s * self.alpha + ns * (1 - self.alpha)
        return self.x, self.y, self.s

# --- 4. RIGID ANCHOR TRACKER (The Fix) ---
class RigidAnchor:
    def __init__(self):
        # 3000 Features = Hyperspecific lock
        self.orb = cv2.ORB_create(nfeatures=3000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.ref_gray = None
        self.ref_kp = None
        self.ref_des = None
        self.ref_center = None 
        self.is_tracking = False
        self.portal_seed = 0
        self.smoother = CoordinateSmoother(alpha=0.5)

    def set_anchor(self, frame, x, y):
        self.ref_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.ref_center = np.array([[[x, y]]], dtype=np.float32)
        
        mask = np.zeros_like(self.ref_gray)
        cv2.circle(mask, (int(x), int(y)), 250, 255, -1)
        
        self.ref_kp, self.ref_des = self.orb.detectAndCompute(self.ref_gray, mask)
        
        if self.ref_kp is None or len(self.ref_kp) < 50:
            return False
            
        self.is_tracking = True
        self.portal_seed = random.randint(0, 10000)
        self.smoother = CoordinateSmoother(alpha=0.5) 
        return True

    def update(self, frame):
        if not self.is_tracking: return None
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        kp, des = self.orb.detectAndCompute(curr_gray, None)
        if des is None: return None
        
        matches = self.bf.knnMatch(self.ref_des, des, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance: good.append(m)
        
        if len(good) > 15:
            src_pts = np.float32([self.ref_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            
            # --- CRITICAL FIX: RIGID TRANSFORM ---
            # Replaced findHomography with estimateAffinePartial2D
            # This restricts movement to Rotation + Translation + Scale.
            # It physically prevents "Sliding" or Skewing.
            M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts)
            
            if M is not None:
                # Transform original center to new position
                new_center = cv2.transform(self.ref_center, M)[0][0]
                
                # Calculate Scale (Zoom)
                scale = math.sqrt(M[0,0]**2 + M[1,0]**2)
                
                # Server-side smoothing for stability
                sx, sy, ss = self.smoother.update(new_center[0], new_center[1], scale)
                
                return float(sx), float(sy), float(ss)

        return None

# --- 5. TEXTURE GENERATOR ---
def generate_portal_texture(target_size, elapsed, seed_val, scale_factor=1.0):
    progress = min(1.0, elapsed / GROWTH_DURATION)
    current_base = target_size * scale_factor
    max_w, max_h = current_base * 1.3, current_base * 3.5
    
    if progress < 0.2: scale = 0.05
    else:
        p = (progress - 0.2) / 0.8
        scale = 0.05 + (0.95 * (1.0 - math.pow(1.0 - p, 4)))

    cur_w, cur_h = int(max_w * scale), int(max_h * scale)
    cur_w, cur_h = max(10, cur_w), max(20, cur_h)
    
    y, x = np.ogrid[-cur_h//2:cur_h//2, -cur_w//2:cur_w//2]
    base_dist = np.sqrt((x/(cur_w/2.0))**2 + (y/(cur_h/2.0))**2)

    np.random.seed(seed_val)
    noise = cv2.GaussianBlur(np.random.rand(cur_h, cur_w), (3,7), 0)
    np.random.seed(None)
    
    fire_noise = cv2.GaussianBlur(np.random.rand(cur_h, cur_w), (3,3), 0)
    fire_noise = np.roll(fire_noise, -int(elapsed * 90), axis=0)
    
    fire = np.clip(1.0 - (base_dist + noise*0.5), 0, 1) + (fire_noise * 0.2)
    colored = cv2.applyColorMap(np.clip(fire*255, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    colored[np.clip(fire*255, 0, 255).astype(np.uint8) > 230] = [255, 255, 255]
    
    b, g, r = cv2.split(colored)
    alpha = (fire > 0.42).astype(np.uint8) * 255
    return cv2.merge([b, g, r, alpha]), cur_w, cur_h

# --- 6. SERVER ---
app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading AI...")
try:
    processor = SegformerImageProcessor.from_pretrained(MODEL_NAME, do_resize=True, size={"height": 512, "width": 512})
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME).to(device).eval()
    print("✅ Model Loaded")
except: pass

@app.get("/")
async def get(): return HTMLResponse(html_content)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    tracker = RigidAnchor()
    start_time = None
    lock_timer = 0
    REQUIRED_FRAMES = 15
    
    try:
        while True:
            data = await websocket.receive_text()
            
            # --- RESET LOGIC ---
            if data.startswith('{'):
                try:
                    msg = json.loads(data)
                    if msg.get("action") == "reset":
                        tracker = RigidAnchor() 
                        start_time = None
                        lock_timer = 0
                        await websocket.send_text(json.dumps({"status": "scanning"}))
                        continue
                except: pass
                continue 

            # --- IMAGE PROCESSING ---
            img_bytes = base64.b64decode(data.split(",")[1])
            frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            if frame is None: continue

            h, w = frame.shape[:2]

            if tracker.is_tracking:
                result = tracker.update(frame)
                if result:
                    px, py, scale = result
                    elapsed = time.time() - start_time
                    sprite, pw, ph = generate_portal_texture(BASE_SIZE, elapsed, tracker.portal_seed, scale)
                    _, buf = cv2.imencode('.png', sprite)
                    tex = base64.b64encode(buf).decode('utf-8')
                    
                    await websocket.send_text(json.dumps({
                        "status": "tracking", "visible": True,
                        "nx": float(px / w), "ny": float(py / h),
                        "nw": float(pw / w), "nh": float(ph / h),
                        "tex": tex
                    }))
                else:
                    await websocket.send_text(json.dumps({"status": "tracking", "visible": False}))
            else:
                cx, cy = w//2, h//2
                is_valid = False
                try:
                    crop = frame[cy-40:cy+40, cx-40:cx+40]
                    inputs = processor(images=crop, return_tensors="pt").to(device)
                    with torch.no_grad(): outputs = model(**inputs)
                    ups = torch.nn.functional.interpolate(outputs.logits, size=(80, 80), mode="bilinear", align_corners=False)
                    cls = torch.max(torch.nn.functional.softmax(ups, dim=1), dim=1)[1].cpu().numpy()[0]
                    if np.mean(cls == CLASS_WALL) > 0.6: is_valid = True
                except: pass

                if is_valid:
                    lock_timer += 1
                    await websocket.send_text(json.dumps({
                        "status": "locking", "progress": lock_timer / REQUIRED_FRAMES
                    }))
                    if lock_timer > REQUIRED_FRAMES:
                        if tracker.set_anchor(frame, cx, cy):
                            start_time = time.time()
                            print("✅ LOCKED")
                        else:
                            lock_timer = 0
                else:
                    lock_timer = 0
                    await websocket.send_text(json.dumps({"status": "scanning"}))

    except Exception as e: print(f"Disconnected: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)