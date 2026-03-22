import cv2
import numpy as np
import re
import math
import os
import argparse
from datetime import datetime
from ultralytics import YOLO

# ---------------------------------------------------------
# Formatting & Validation Modules 
# ---------------------------------------------------------

def clean_and_correct_text(text):
    """
    Cleans up the text by removing special characters and spaces, 
    and applies simple logic corrections (e.g., O vs 0).
    """
    text = re.sub(r'[^A-Z0-9]', '', text.upper())
    
    dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5', 'B': '8', 'Z': '2'}
    dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S', '8': 'B', '2': 'Z'}
    
    if len(text) in [9, 10]:
        clean_text = ""
        for i, char in enumerate(text):
            if i in [0, 1]:
                if char in dict_int_to_char: clean_text += dict_int_to_char[char]
                else: clean_text += char
            elif i in [2, 3]:
                if char in dict_char_to_int: clean_text += dict_char_to_int[char]
                else: clean_text += char
            elif i >= len(text) - 4:
                if char in dict_char_to_int: clean_text += dict_char_to_int[char]
                else: clean_text += char
            else:
                if char in dict_int_to_char: clean_text += dict_int_to_char[char]
                else: clean_text += char
        return clean_text
        
    return text

def validate_plate_text(text):
    text = clean_and_correct_text(text)
    pattern = r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$"
    is_valid = bool(re.match(pattern, text))
    return text, is_valid

# ---------------------------------------------------------
# Two-Stage YOLO Models: Plate Detection + Character Detection
# ---------------------------------------------------------

def detect_and_crop_plate(model, frame, conf_thresh=0.20):
    """Stage 1: Detects plates and returns crops."""
    results = model(frame, device=0, half=True, verbose=False, imgsz=640, conf=conf_thresh)
    bboxes, crops = [], []
    h, w = frame.shape[:2]
    
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if (y2 - y1) < 10 or (x2 - x1) < 10:
            continue
            
        bboxes.append((x1, y1, x2, y2))
        crops.append(frame[y1:y2, x1:x2])
        
    return bboxes, crops, results[0].plot()

def extract_text_with_yolo(char_model, crop_img, conf_thresh=0.30):
    """
    Stage 2: Runs the YOLO Character model over the cropped plate image,
    sorts bounding boxes left-to-right, and builds the string.
    """
    if crop_img is None or crop_img.size == 0:
        return "", 0.0
        
    # We use a very small imgsz (320) because characters are simple and the crop is tiny
    results = char_model(crop_img, device=0, half=True, verbose=False, imgsz=320, conf=conf_thresh)
    
    boxes = results[0].boxes
    if len(boxes) == 0:
        return "", 0.0
        
    detected_chars = []
    for box in boxes:
        # Get x-coordinate to sort letters left-to-right
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        center_x = (x1 + x2) // 2
        
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        char_label = char_model.names[cls_id] # Look up the character by its class ID (0-35)
        
        detected_chars.append({"char": char_label, "x": center_x, "conf": conf})
        
    # Sort strictly from left to right
    detected_chars.sort(key=lambda item: item["x"])
    
    final_string = "".join(item["char"] for item in detected_chars)
    avg_conf = sum(item["conf"] for item in detected_chars) / len(detected_chars)
    
    return final_string, avg_conf

# ---------------------------------------------------------
# Fast Vehicle Tracker Cache
# ---------------------------------------------------------

class SpeedTracker:
    def __init__(self, dist_thresh=100, save_dir="plateresults"):
        self.objects = []
        self.dist_thresh = dist_thresh
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
    def get_known_plate(self, center):
        closest, min_d = None, float('inf')
        for obj in self.objects:
            cx, cy = obj['center']
            d = math.hypot(center[0]-cx, center[1]-cy)
            if d < self.dist_thresh and d < min_d:
                closest, min_d = obj, d
                
        if closest:
            return closest['text'], closest['conf'], closest['is_valid']
        return "", 0.0, False

    def update_and_save(self, center, text, conf, is_valid, orig_crop):
        closest, min_d = None, float('inf')
        
        for obj in self.objects:
            cx, cy = obj['center']
            d = math.hypot(center[0]-cx, center[1]-cy)
            if d < self.dist_thresh and d < min_d:
                closest, min_d = obj, d
                
        if closest is not None:
            closest['center'] = center
            val_upgrade = is_valid and not closest['is_valid']
            conf_upgrade = conf > closest['conf'] and (is_valid == closest['is_valid'])
            
            if val_upgrade or conf_upgrade:
                closest['text'] = text
                closest['conf'] = conf
                closest['is_valid'] = is_valid
        else:
            closest = {'center': center, 'text': text, 'conf': conf, 'is_valid': is_valid, 'saved': False}
            self.objects.append(closest)
            
        if closest['is_valid'] and not closest['saved']:
            closest['saved'] = True
            safe_text = closest['text']
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            print(f">>> [SUCCESS] Saved validated YOLO Stage-2 plate: {safe_text} at {closest['conf']:.2f} acc!")
            cv2.imwrite(os.path.join(self.save_dir, f"{safe_text}_{ts}.jpg"), orig_crop)
            
        return closest['text'], closest['is_valid']

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser("YOLO Two-Stage ANPR")
    # Weights for detecting the huge rectanglar plates
    parser.add_argument("--plate-weights", type=str, default=r"runs\detect\runs\detect\anpr_train\weights\best.pt")
    # Weights for detecting small characters inside that plate
    parser.add_argument("--char-weights", type=str, default=r"runs\detect\char_detect2\weights\best.pt") 
    parser.add_argument("--source", type=str, default="0")
    args = parser.parse_args()
    
    print(f"[INFO] Initializing Real-Time Double YOLO Pipeline...")
    source = 0 if args.source == '0' else os.path.abspath(args.source)
    
    print(f"[INFO] Loading Stage 1: Plate Model ({args.plate_weights})...")
    plate_model = YOLO(args.plate_weights)
    
    print(f"[INFO] Loading Stage 2: Character Model ({args.char_weights})...")
    char_model = YOLO(args.char_weights)
    
    tracker = SpeedTracker(dist_thresh=100)
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        return

    frame_count = 0
    print("[INFO] Zero-Lag Stream started...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"[INFO] Stream finished cleanly after {frame_count} frames.")
            break
            
        frame_count += 1
        bboxes, crops, annotated_frame = detect_and_crop_plate(plate_model, frame, conf_thresh=0.20)
        
        for bbox, crop in zip(bboxes, crops):
            x1, y1, x2, y2 = bbox
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            mem_txt, mem_conf, mem_val = tracker.get_known_plate((cx, cy))
            
            do_inference = False
            # Read character again if not fully valid, but occasionally to save extreme FPS
            if not mem_val and frame_count % 3 == 0:
                do_inference = True 
            elif mem_txt == "":
                do_inference = True 
                
            final_text = ""
            is_valid = False
            
            if do_inference:
                # INSTANT text extraction leveraging the lightweight Character YOLO Model
                raw_text, conf = extract_text_with_yolo(char_model, crop, conf_thresh=0.25)
                
                if raw_text:
                    valid_text, is_val = validate_plate_text(raw_text)
                    final_text, is_valid = tracker.update_and_save((cx, cy), valid_text, conf, is_val, crop)
                else:
                    final_text, is_valid = tracker.update_and_save((cx, cy), "", 0.0, False, crop)
            else:
                final_text, is_valid = tracker.update_and_save((cx, cy), mem_txt, mem_conf, mem_val, crop)
                
            if final_text:
                color = (0, 255, 0) if is_valid else (0, 0, 255)
                cv2.putText(annotated_frame, final_text, (x1, max(30, y1 - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)

        cv2.imshow("Production Two-Stage YOLO ANPR", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
