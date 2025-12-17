import cv2
import torch
import numpy as np
import time
import math
import random
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# --- CONFIGURATION ---
MODEL_NAME = "nvidia/segformer-b2-finetuned-ade-512-512" 
CLASS_WALL = 0
RESIZE_WIDTH = 640 

# Portal Settings
GROWTH_DURATION = 12.0
BASE_SIZE = 100 

# --- SMOOTHING CLASS ---
class PositionSmoother:
    def __init__(self, alpha=0.3):
        self.x = None
        self.y = None
        self.alpha = alpha 

    def update(self, new_x, new_y):
        if self.x is None:
            self.x = new_x
            self.y = new_y
        else:
            self.x = (self.alpha * new_x) + ((1 - self.alpha) * self.x)
            self.y = (self.alpha * new_y) + ((1 - self.alpha) * self.y)
        return int(self.x), int(self.y)

def generate_lens_portal(target_size, elapsed, seed_val, is_vertical=True):
    """ Generates a thin, jagged "Convex Lens" (Vesica Piscis) shape. """
    progress = min(1.0, elapsed / GROWTH_DURATION)
    
    # 1. Dimensions: UPDATED for wider look
    if is_vertical:
        # Increased width multiplier from 0.8 to 1.1
        max_w = target_size * 1.1
        max_h = target_size * 4.0
    else:
        max_w = target_size * 4.0
        # Increased height multiplier from 0.8 to 1.1
        max_h = target_size * 1.1

    if progress < 0.2:
        crack_prog = progress / 0.2
        scale_long = crack_prog * 0.8 + 0.2 
        scale_short = 0.02 
    else:
        open_prog = (progress - 0.2) / 0.8
        scale_short = 0.02 + (0.38 * (1.0 - math.pow(1.0 - open_prog, 3)))
        scale_long = 1.0

    if is_vertical:
        cur_w = int(max_w * scale_short)
        cur_h = int(max_h * scale_long)
    else:
        cur_w = int(max_w * scale_long)
        cur_h = int(max_h * scale_short)

    cur_w = max(10, cur_w)
    cur_h = max(20, cur_h)
    
    if cur_h % 2 == 0: cur_h += 1
    if cur_w % 2 == 0: cur_w += 1
    
    current_h = cur_h
    current_w = cur_w
    
    y, x = np.ogrid[-current_h//2:current_h//2, -current_w//2:current_w//2]
    norm_x = x / (current_w / 2.0)
    norm_y = y / (current_h / 2.0)
    
    sharpness = 0.8 
    base_dist = np.sqrt(norm_x**2 + (norm_y**2 * sharpness))

    # 4. Jagged Distortion: UPDATED for rougher look
    np.random.seed(seed_val)
    noise_map = np.random.rand(current_h, current_w)
    noise_map = cv2.GaussianBlur(noise_map, (3, 3), 0)
    np.random.seed(None)
    
    # Increased distortion strength from 0.4 to 0.65
    dist = base_dist + (noise_map * 0.65)
    
    noise_scroll = int(elapsed * 90)
    fire_noise = np.random.rand(current_h, current_w)
    fire_noise = cv2.GaussianBlur(fire_noise, (3, 3), 0)
    
    if is_vertical:
        fire_noise = np.roll(fire_noise, -noise_scroll, axis=0) 
    else:
        fire_noise = np.roll(fire_noise, -noise_scroll, axis=1) 
        
    glow = 1.0 - dist
    glow = np.clip(glow, 0, 1)
    
    fire = glow + (fire_noise * 0.25)
    mask_float = fire > 0.45
    
    fire_visual = np.clip(fire * 255, 0, 255).astype(np.uint8)
    colored_portal = cv2.applyColorMap(fire_visual, cv2.COLORMAP_INFERNO)
    
    white_core = fire_visual > 230
    colored_portal[white_core] = [255, 255, 255]
    
    return mask_float, colored_portal, current_w, current_h

class ZeroDriftTracker:
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=3000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.ref_img = None
        self.ref_kp = None
        self.ref_des = None
        self.spawn_point = None
        self.is_tracking = False
        self.ref_corners = None
        self.portal_seed = random.randint(0, 10000)
        self.is_vertical = True
        self.smoother = PositionSmoother(alpha=0.3) 

    def analyze_wall_and_start(self, frame, wall_mask):
        h, w = wall_mask.shape
        
        # 1. Determine Orientation based on wall shape
        req_w_v = int(BASE_SIZE * 1.1) # Updated width requirement
        req_h_v = int(BASE_SIZE * 4.0)
        
        req_w_h = int(BASE_SIZE * 4.0)
        req_h_h = int(BASE_SIZE * 1.1) # Updated height requirement
        
        # Test Fits
        kernel_v = np.ones((req_h_v, req_w_v), np.uint8) 
        safe_zone_v = cv2.erode(wall_mask.astype(np.uint8), kernel_v[:req_h_v//2, :req_w_v//2], iterations=1)
        
        kernel_h = np.ones((req_w_h, req_h_h), np.uint8)
        safe_zone_h = cv2.erode(wall_mask.astype(np.uint8), kernel_h[:req_w_h//2, :req_h_h//2], iterations=1)
        
        valid_pixels_v = cv2.countNonZero(safe_zone_v)
        valid_pixels_h = cv2.countNonZero(safe_zone_h)
        
        best_safe_zone = None
        
        if valid_pixels_v > valid_pixels_h and valid_pixels_v > 0:
            self.is_vertical = True
            best_safe_zone = safe_zone_v
            print("✅ Fits better VERTICALLY.")
        elif valid_pixels_h > 0:
            self.is_vertical = False
            best_safe_zone = safe_zone_h
            print("✅ Fits better HORIZONTALLY.")
        else:
            print("❌ Wall too cluttered/small to fit wider portal.")
            return False

        # 3. Find Best Spot
        valid_y, valid_x = np.where(best_safe_zone > 0)
        if len(valid_x) == 0: return False

        center_x, center_y = w // 2, h // 2
        distances = np.sqrt((valid_x - center_x)**2 + (valid_y - center_y)**2)
        min_idx = np.argmin(distances)
        spawn_x = valid_x[min_idx]
        spawn_y = valid_y[min_idx]
        
        print(f"✅ Spawn Point: ({spawn_x}, {spawn_y})")

        # 4. Lock Reference
        self.ref_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.spawn_point = np.array([[[spawn_x, spawn_y]]], dtype=np.float32)
        h_img, w_img = self.ref_img.shape
        self.ref_corners = np.float32([[0, 0], [0, h_img-1], [w_img-1, h_img-1], [w_img-1, 0]]).reshape(-1, 1, 2)
        
        self.ref_kp, self.ref_des = self.orb.detectAndCompute(self.ref_img, None)
        
        if self.ref_kp is None or len(self.ref_kp) < 50:
            print("❌ Wall too smooth. Need texture to lock.")
            return False
            
        self.is_tracking = True
        self.portal_seed = random.randint(0, 10000)
        self.smoother = PositionSmoother(alpha=0.3)
        return True

    def update(self, frame):
        if not self.is_tracking: return None

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(curr_gray, None)
        if des is None: return None

        matches = self.bf.knnMatch(self.ref_des, des, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        if len(good_matches) < 25: return None 

        src_pts = np.float32([self.ref_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if M is not None:
            try:
                dst_corners = cv2.perspectiveTransform(self.ref_corners, M)
                if cv2.isContourConvex(np.int32(dst_corners)):
                    area = cv2.contourArea(dst_corners)
                    frame_area = frame.shape[0] * frame.shape[1]
                    
                    if (frame_area * 0.05) < area < (frame_area * 4):
                        raw_pos = cv2.perspectiveTransform(self.spawn_point, M)
                        raw_x = raw_pos[0][0][0]
                        raw_y = raw_pos[0][0][1]
                        
                        if -2000 < raw_x < 5000 and -2000 < raw_y < 5000:
                            return self.smoother.update(raw_x, raw_y)
            except: pass
        return None

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading AI on: {device}...")
    
    try:
        processor = SegformerImageProcessor.from_pretrained(MODEL_NAME, do_resize=True, size={"height": 512, "width": 512})
        model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"AI Load Error: {e}")
        return

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret: return
    frame = cv2.resize(frame, (RESIZE_WIDTH, int(frame.shape[0] * (RESIZE_WIDTH / frame.shape[1]))))
    h, w, _ = frame.shape
    
    tracker = ZeroDriftTracker()
    start_time = None 
    
    print("--- READY ---")
    print("Aim at a textured wall and press 'r'.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (w, h))
        display = frame.copy()
        
        # --- 1. AI SEGMENTATION ---
        wall_mask = None
        try:
            inputs = processor(images=frame, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            
            logits = outputs.logits
            upsampled = torch.nn.functional.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
            probs = torch.nn.functional.softmax(upsampled, dim=1)
            top_classes = torch.max(probs, dim=1)[1].cpu().numpy()[0]
            
            wall_mask_raw = (top_classes == CLASS_WALL)
            kernel = np.ones((5,5), np.uint8)
            wall_mask = cv2.morphologyEx(wall_mask_raw.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
        except:
            wall_mask = None

        # --- 2. TRACKER UPDATE ---
        pos = tracker.update(frame)
        
        # --- 3. RENDERING ---
        if pos is not None and start_time is not None:
            px, py = pos
            
            if -100 < px < w + 100 and -100 < py < h + 100:
                elapsed = time.time() - start_time
                p_mask, p_tex, pw, ph = generate_lens_portal(BASE_SIZE, elapsed, tracker.portal_seed, tracker.is_vertical)
                
                x1 = px - pw // 2
                y1 = py - ph // 2
                x2 = x1 + pw
                y2 = y1 + ph
                
                p_x1, p_y1 = 0, 0
                p_x2, p_y2 = pw, ph
                
                if x1 < 0: p_x1 = -x1; x1 = 0
                if y1 < 0: p_y1 = -y1; y1 = 0
                if x2 > w: p_x2 = pw - (x2 - w); x2 = w
                if y2 > h: p_y2 = ph - (y2 - h); y2 = h
                
                if x2 > x1 and y2 > y1 and wall_mask is not None:
                    tex_crop = p_tex[p_y1:p_y2, p_x1:p_x2]
                    alpha_crop = p_mask[p_y1:p_y2, p_x1:p_x2]
                    bg_crop = display[y1:y2, x1:x2]
                    
                    scene_wall_crop = wall_mask[y1:y2, x1:x2]
                    final_mask = alpha_crop & scene_wall_crop
                    
                    if np.any(final_mask):
                        render_idx = final_mask > 0
                        bg_crop[render_idx] = cv2.addWeighted(bg_crop[render_idx], 0.1, tex_crop[render_idx], 0.9, 0)
                        display[y1:y2, x1:x2] = bg_crop
        
        if not tracker.is_tracking:
            cv2.putText(display, "Press 'r' to SPAWN", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Stranger Things AR (Wider & Jagged)', display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord('r'):
            if wall_mask is not None:
                success = tracker.analyze_wall_and_start(frame, wall_mask)
                if success:
                    start_time = time.time()
                else:
                    print("Finding a better spot...")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()