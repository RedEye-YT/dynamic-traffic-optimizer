import cv2
import streamlit as st
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
from ultralytics import YOLO
from stable_baselines3 import PPO
from utils.tracker import process_frame_detections

# ==========================================
# 1. SETUP UI FIRST (Must be the very first Streamlit command!)
# ==========================================
st.set_page_config(page_title="Dynamic AI Traffic Optimizer", layout="wide")

# ==========================================
# 2. DEFINE VIDEO SOURCE EARLY 
# ==========================================
VIDEO_SOURCE = 0  # 0 for webcam. Change back to r"d:\VS Code Terminal\traffic.mp4" for video.

# ==========================================
# 3. LOAD MODELS
# ==========================================
@st.cache_resource
def load_models():
    vision = YOLO('yolov8n.pt') 
    rl = PPO.load("models/ppo_traffic_optimizer")
    return vision, rl

vision_model, rl_agent = load_models()

# ==========================================
# 4. DEFINE ADVANCED FUNCTIONS (Auto-Calibration)
# ==========================================
def auto_calibrate_lanes(video_source, num_lanes=4, max_frames=150):
    cap = cv2.VideoCapture(video_source, cv2.CAP_DSHOW)
    vehicle_points = []

    st.sidebar.info("Gathering traffic telemetry... Please wait.")
    progress_bar = st.sidebar.progress(0)
    
    for frame_count in range(max_frames):
        ret, frame = cap.read()
        if not ret: break
        
        results = vision_model(frame, classes=[2, 3, 5, 7], verbose=False)
        if results[0].boxes is not None:
            for box in results[0].boxes.xyxy.cpu().numpy():
                cx = int((box[0] + box[2]) / 2)
                # FIX 1: Track the bottom edge of the car (tires) to match tracker.py!
                cy = int(box[3]) 
                vehicle_points.append([cx, cy])
                
        progress_bar.progress((frame_count + 1) / max_frames)
    
    cap.release()
    progress_bar.empty() 
    
    X = np.array(vehicle_points)
    kmeans = KMeans(n_clusters=num_lanes, random_state=42, n_init=10).fit(X)
    
    auto_rois = {}
    
    # FIX 2: Keep the names as North/South so the Streamlit UI can find the data!
    lane_names = ["North", "South", "East", "West"] 
    
    for i in range(num_lanes):
        lane_points = X[kmeans.labels_ == i]
        if len(lane_points) > 3: 
            hull = ConvexHull(lane_points)
            auto_rois[lane_names[i]] = lane_points[hull.vertices].tolist()
            
    return auto_rois

# ==========================================
# 5. SIDEBAR UI
# ==========================================
st.sidebar.title("⚙️ System Configuration")
mode = st.sidebar.radio(
    "Intersection Type", 
    ["4-Way Intersection", "2-Way Street"],
    help="Switching to 2-Way will isolate the AI focus to the primary lanes."
)

if st.sidebar.button("Auto-Calibrate Lanes"):
    target_lanes = 4 if mode == "4-Way Intersection" else 2
    st.session_state['custom_rois'] = auto_calibrate_lanes(VIDEO_SOURCE, num_lanes=target_lanes)
    st.sidebar.success("Lanes successfully mapped via K-Means Clustering!")

# ==========================================
# 6. MAIN DASHBOARD UI
# ==========================================
st.title("🚥 Dynamic AI Traffic Flow Optimizer")
st.markdown("Live perception and reinforcement learning telemetry.")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Feed")
    video_placeholder = st.empty() 

with col2:
    st.subheader("Live Telemetry")
    state_metrics = st.empty()
    action_metrics = st.empty()

# ==========================================
# 7. MAIN VIDEO LOOP
# ==========================================
cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_DSHOW)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.warning("Video stream ended. Please restart the app.")
        break
    
    # Run Perception (with confidence threshold)
    results = vision_model(frame, classes=[2, 3, 5, 7, 9], conf=0.5, verbose=False)
    
    # Fetch active Auto-Calibrated lanes
    active_rois = st.session_state.get('custom_rois', None)
    
    # Pass active_rois into tracker
    lane_counts, ev_flag, _ = process_frame_detections(results[0].boxes, frame=frame, custom_rois=active_rois)
    
    annotated_frame = results[0].plot()
    
    # Zero-Padding trick for 2-Way vs 4-Way
    if mode == "4-Way Intersection":
        state_vector = np.array([
            lane_counts.get("North", 0), lane_counts.get("South", 0), 
            lane_counts.get("East", 0), lane_counts.get("West", 0), ev_flag
        ], dtype=np.float32)
    else:
        state_vector = np.array([
            lane_counts.get("North", 0), lane_counts.get("South", 0), 
            0, 0, ev_flag
        ], dtype=np.float32)
    
    # Run Reinforcement Learning Agent
    action, _ = rl_agent.predict(state_vector, deterministic=True)
    
    # Update the Dashboard
    video_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)
    
    with state_metrics.container():
        st.metric(label="Emergency Vehicle Detected", value="YES 🚨" if ev_flag == 1 else "NO")
        st.write("**Current Lane Densities:**")
        
        if mode == "4-Way Intersection":
            st.write(f"🟢 **North:** {lane_counts.get('North', 0)} vehicles")
            st.write(f"🟡 **South:** {lane_counts.get('South', 0)} vehicles")
            st.write(f"🔴 **East:** {lane_counts.get('East', 0)} vehicles")
            st.write(f"🔵 **West:** {lane_counts.get('West', 0)} vehicles")
        else:
            st.write(f"🟢 **Lane 1:** {lane_counts.get('North', 0)} vehicles")
            st.write(f"🟡 **Lane 2:** {lane_counts.get('South', 0)} vehicles")

    with action_metrics.container():
        st.write("---")
        st.write("**🧠 AI Calculated Green Light Durations:**")
        if mode == "4-Way Intersection":
            st.success(f"Phase 1 (North-South): {int(action[0])} seconds")
            st.info(f"Phase 2 (East-West): {int(action[1])} seconds")
            st.warning(f"Phase 3 (N-Turn): {int(action[2])} seconds")
            st.error(f"Phase 4 (S-Turn): {int(action[3])} seconds")
        else:
            st.success(f"Phase 1 (Lane 1 Green): {int(action[0])} seconds")
            st.info(f"Phase 2 (Lane 2 Green): {int(action[1])} seconds")

cap.release()