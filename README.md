# 🚥 Dynamic AI Traffic Flow Optimizer

An enterprise-grade, real-time traffic management system that uses a **Dual-Brain Computer Vision Architecture (YOLOv8)** for vehicle and emergency routing, **Unsupervised Machine Learning (K-Means)** for automatic lane mapping, and **Deep Reinforcement Learning (PPO)** to optimize traffic light timings dynamically.

Built with a live interactive dashboard using **Streamlit**. Designed as a smart-city civic solution to eliminate wait times and prioritize emergency responders.

## ✨ Key Features

* **🧠 Dual-Brain Perception (New!):** Runs two YOLO models simultaneously. A base YOLOv8 model tracks standard traffic (cars, buses, trucks), while a specialized, high-confidence YOLOv8 model acts as an emergency override, detecting ambulances and instantly triggering priority routing.
* **📐 Auto-Calibrating Lanes (The "Desire Path" Method):** The system watches live traffic for 150 frames, tracks vehicle trajectories, and uses **K-Means Clustering** and **Convex Hull** algorithms to automatically draw geometric lane boundaries around actual traffic flow.
* **🤖 Reinforcement Learning Optimizer:** A pre-trained Proximal Policy Optimization (PPO) agent from `stable-baselines3` analyzes real-time lane densities to dynamically calculate the optimal green light duration for each phase.
* **🎛️ Futuristic Streamlit Dashboard:** A sleek, real-time UI with a dark-mode sci-fi aesthetic that overlays geometric tracking telemetry onto the live video feed.

## 🏗️ System Architecture

1.  **Dual-Perception Layer (`app.py` & YOLO):** Captures live video/webcam feed and runs high-speed object detection using layered inference to eliminate false positives.
2.  **Spatial Logic Layer (`utils/tracker.py`):** Translates raw bounding boxes into spatial intelligence. Employs Point-in-Polygon (PiP) mathematical tests tracking the *bottom edge* of vehicles for highly accurate lane counting.
3.  **Intelligence Layer (`stable-baselines3`):** Receives the array of lane densities (e.g., `[North, South, East, West, EV_Flag]`) and outputs deterministic phase timings.

## 🚀 Installation & Setup

**1. Clone the repository:**
```bash
https://github.com/RedEye-YT/dynamic-traffic-optimizer.git
```
dynamic-traffic-optimizer

2. Install dependencies:
Make sure you have Python 3.8+ installed.
```
pip install -r requirements.txt
```
(Required packages: opencv-python, streamlit, numpy, ultralytics, stable-baselines3, scikit-learn, scipy)

3.Run the Dashboard:
```
streamlit run app.py
```

🎮 How to Use
Launch the app. You will see the live webcam/video feed and the raw YOLO bounding boxes.

Open the ⚙️ System Configuration sidebar on the left.

Select your intersection type (4-Way or 2-Way).

Click "Auto-Calibrate Lanes".

Wait a few seconds while the AI gathers telemetry. Once finished, custom yellow geometric boundaries will snap onto the screen.

Test the Emergency Override: Hold up a photo of an ambulance to the camera to watch the secondary AI model instantly flag the vehicle and trigger the YES 🚨 override!

🔮 Future Roadmap
Audio-Based Emergency Detection: Introduce an audio-frequency thread to detect ambulance sirens, bypassing occlusion issues in heavy traffic.

Civic Dashboard Integration: Connect the telemetry data to a broader civic network for city planners to analyze long-term traffic flow and emergency response times.

🤝 Contributing
***

This makes the project look incredibly robust. The "Dual-Brain Perception" bullet point alone is going to look fantastic on a resume. 

Would you like to brainstorm some quick talking points on how to present this project if you are demonstrating it live to a panel?




