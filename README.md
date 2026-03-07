# 🚥 Dynamic AI Traffic Flow Optimizer

An end-to-end, real-time AI prototype that uses Computer Vision to detect traffic density and Reinforcement Learning to dynamically optimize traffic light durations.

## 🧠 System Architecture

1. **Perception (Ultralytics YOLOv8):** Processes live intersection video feeds to detect vehicles.
2. **State Translation (OpenCV):** Maps bounding boxes to custom Region of Interest (ROI) polygons to calculate directional lane density.
3. **Intelligence (Stable-Baselines3 PPO):** A Reinforcement Learning agent trained in a custom `Gymnasium` environment calculates optimal green-light phases based on the live state vector.
4. **Dashboard (Streamlit):** A decoupled web UI that renders the live computer vision feed alongside real-time system telemetry and decision metrics.

## 📂 Project Structure

```text
├── environment.py      # Custom Gymnasium environment with Green Corridor reward logic
├── train.py            # PPO agent training script (100k timesteps)
├── app.py              # Streamlit web dashboard and live inference loop
├── utils/
│   ├── tracker.py      # OpenCV point-polygon ROI mapping logic
│   └── roi_selector.py # GUI tool for calibrating intersection camera angles
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation

🚀 How to Run Locally
1. Install Dependencies

Ensure you have Python installed, then install the required libraries:

pip install -r requirements.txt

2. Calibrate the Camera (Optional)

If you are using a new intersection video feed, you must map the physical lanes (North, South, East, West) to the camera's perspective. Run the calibration tool and click the 4 corners of each lane:

python utils/roi_selector.py

(Copy the generated dictionary from the terminal and paste it into utils/tracker.py)

3. Train the AI "Brain"

Train the Proximal Policy Optimization (PPO) reinforcement learning agent:

python train.py

(This will simulate 100,000 traffic timesteps and save the model to the models/ directory).

4. Launch the Live Dashboard

Start the real-time Streamlit web UI to run the perception and decision loop:

streamlit run app.py
