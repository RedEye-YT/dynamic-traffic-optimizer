import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import cv2
import numpy as np
from ultralytics import YOLO
from stable_baselines3 import PPO
from collections import defaultdict

# ==========================================
# 1. INITIALIZE "EYES" AND "BRAIN"
# ==========================================
print("Loading YOLO Perception Model...")
vision_model = YOLO('yolo11l.pt')

print("Loading PPO Intelligence Model...")
# Update this path to wherever your trained model is saved!
# e.g., "models/ppo_traffic_agent.zip"
rl_agent = PPO.load("models/ppo_traffic_optimizer")

# Open the video feed
cap = cv2.VideoCapture('traffic.mp4')

# Tracking variables
line_y_red = 430
crossed_ids = set()
TARGET_CLASSES = [1, 2, 3, 5, 6, 7] # Vehicles only

# ==========================================
# 2. THE REAL-TIME INFERENCE LOOP
# ==========================================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # --- A. PERCEPTION PHASE (YOLO) ---
    results = vision_model.track(frame, persist=True, classes=TARGET_CLASSES, verbose=False)
    
    current_frame_count = 0 # Count how many vehicles cross this specific frame
    
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        
        # Draw the counting line
        cv2.line(frame, (690, line_y_red), (1130, line_y_red), (0, 0, 255), 3)

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            cy = (y1 + y2) // 2
            
            # Draw standard bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Check if it crossed the line
            if cy > line_y_red and track_id not in crossed_ids:
                crossed_ids.add(track_id)
                current_frame_count += 1 

    # --- B. TRANSLATION PHASE (The Bridge) ---
    # Your PPO environment likely expects an array representing traffic in multiple lanes.
    # For this prototype, we will map our single line count to the state vector.
    # (If your gym environment expects a 4-element array for N, S, E, W lanes, we structure it here).
    
    total_crossed = len(crossed_ids)
    
    # Example: [Lane_1_Count, Lane_2_Count, Lane_3_Count, Lane_4_Count]
    # We are simulating the other lanes with 0 for this 1-camera test.
    state_vector = np.array([total_crossed, 0, 0, 0, 0], dtype=np.float32)

    # --- C. INTELLIGENCE PHASE (PPO) ---
    # Feed the real-time state to the RL agent to get the optimal green light phase
    action, _states = rl_agent.predict(state_vector, deterministic=True)
    
    # For a 4-way intersection, 'action' is usually an array of 4 green-light durations
    phase_n_duration = action[0]

    # --- D. DASHBOARD RENDERING ---
    # Display the AI's real-time decision on the screen
    cv2.putText(frame, f"Total Vehicles: {total_crossed}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"AI Recommended Green Light: {phase_n_duration:.1f}s", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Dynamic AI Traffic Optimizer - Live Inference", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()