# 🚥 Dynamic AI Traffic Flow Optimizer

An enterprise-grade, real-time traffic management system that uses **Computer Vision (YOLOv8)** for vehicle detection, **Unsupervised Machine Learning (K-Means)** for automatic lane mapping, and **Deep Reinforcement Learning (PPO)** to optimize traffic light timings dynamically.

Built with a live interactive dashboard using **Streamlit**.

## ✨ Key Features

* **🧠 Auto-Calibrating Lanes (The "Desire Path" Method):** No hardcoded lane coordinates! The system watches live traffic for 150 frames, tracks vehicle trajectories, and uses **K-Means Clustering** and **Convex Hull** algorithms to automatically draw geometric lane boundaries around actual traffic flow.
* **👁️ Robust Vehicle Perception:** Utilizes YOLOv8 to detect cars, buses, trucks, and motorcycles in real-time. Employs Point-in-Polygon (PiP) mathematical tests tracking the *bottom edge* of bounding boxes (where tires meet the road) for highly accurate lane counting.
* **🤖 Reinforcement Learning Optimizer:** A pre-trained Proximal Policy Optimization (PPO) agent from `stable-baselines3` analyzes real-time lane densities to dynamically calculate the optimal green light duration for each phase, preventing empty-intersection delays.
* **🎛️ Live Streamlit Dashboard:** A sleek, real-time UI that overlays geometric tracking telemetry onto the live video/webcam feed. Includes a toggle to instantly switch the AI's state-vector padding between 4-Way Intersections and 2-Way Streets.

## 🏗️ System Architecture

   **Perception Layer (`app.py` & YOLO):** Captures live video/webcam feed and runs high-speed object detection with customized confidence thresholds to eliminate false positives.
**Spatial Logic Layer (`utils/tracker.py`):** Translates raw bounding boxes into spatial intelligence. Determines if a vehicle's coordinates fall inside the dynamically generated mathematical polygons.
**Intelligence Layer (`stable-baselines3`):** Receives the array of lane densities (e.g., `[North_Count, South_Count, East_Count, West_Count, EV_Flag]`) and outputs deterministic phase timings.

## 🚀 Installation & Setup

**1. Clone the repository:**

git clone [https://github.com/RedEye-YT/dynamic-traffic-optimizer.git](https://github.com/RedEye-YT/dynamic-traffic-optimizer.git)
cd dynamic-traffic-optimizer

2. Install dependencies:
Make sure you have Python 3.8+ installed.

pip install -r requirements.txt
(Required packages: opencv-python, streamlit, numpy, ultralytics, stable-baselines3, scikit-learn, scipy)

3.Run the Dashboard:

streamlit run app.py

🎮 How to Use
Launch the app. You will see the live webcam/video feed and the raw YOLO bounding boxes.

Open the ⚙️ System Configuration sidebar on the left.

Select your intersection type (4-Way or 2-Way).

Click "Auto-Calibrate Lanes".

Wait a few seconds while the AI gathers telemetry. Once finished, custom yellow geometric boundaries will snap onto the screen, and the Reinforcement Learning agent will instantly begin calculating optimal light timings based on the live vehicle counts.

🔮 Future Roadmap
Audio-Based Emergency Detection: Replace standard visual class detection for emergency vehicles with an audio-frequency thread to detect ambulance sirens, bypassing occlusion issues in heavy traffic.

Fine-Tuned Ambulance Model: Integrate a custom-trained YOLOv8 model utilizing the Roboflow Emergency Vehicle dataset for improved visual emergency routing.

🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

