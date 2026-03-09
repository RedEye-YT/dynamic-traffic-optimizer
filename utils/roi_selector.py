import cv2
import numpy as np

# Configuration
VIDEO_SOURCE = r"d:\VS Code Terminal\traffic.mp4" # Replace with your video path
LANE_NAMES = ["North", "South", "East", "West"]

# Global variables for the drawing state
current_points = []
final_rois = {}
current_lane_idx = 0
frame_copy = None

def click_event(event, x, y, flags, param):
    global current_points, frame_copy, current_lane_idx, final_rois
    
    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append((x, y))
        
        # Draw a small circle at the clicked point
        cv2.circle(frame_copy, (x, y), 5, (0, 255, 0), -1)
        
        # Connect the points with a line if there's more than one
        if len(current_points) > 1:
            cv2.line(frame_copy, current_points[-2], current_points[-1], (0, 255, 0), 2)
            
        cv2.imshow("ROI Selector", frame_copy)

        # Once 4 points are clicked, save the polygon and move to the next lane
        if len(current_points) == 4:
            # Close the polygon visually
            cv2.line(frame_copy, current_points[-1], current_points[0], (0, 255, 0), 2)
            
            lane_name = LANE_NAMES[current_lane_idx]
            final_rois[lane_name] = np.array(current_points, np.int32)
            print(f"[SUCCESS] {lane_name} ROI saved.")
            
            # Reset for the next lane
            current_points = []
            current_lane_idx += 1
            
            if current_lane_idx < len(LANE_NAMES):
                print(f"\n-> Next: Click 4 points for the {LANE_NAMES[current_lane_idx]} lane.")
            else:
                print("\n[DONE] All ROIs selected! Press any key to close and generate code.")

def run_selector():
    global frame_copy
    
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error: Could not read video. Check the VIDEO_SOURCE path.")
        return

    frame_copy = frame.copy()
    cv2.namedWindow("ROI Selector")
    cv2.setMouseCallback("ROI Selector", click_event)
    
    print("=== Traffic Optimizer ROI Calibration ===")
    print(f"-> Start by clicking 4 points for the {LANE_NAMES[current_lane_idx]} lane (e.g., top-left, top-right, bottom-right, bottom-left).")

    while True:
        # Display instructions on the screen
        display_frame = frame_copy.copy()
        if current_lane_idx < len(LANE_NAMES):
            text = f"Click 4 points for: {LANE_NAMES[current_lane_idx]} Lane"
            cv2.putText(display_frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("ROI Selector", display_frame)
        
        # Wait for a key press; exit if 'q' is pressed or all lanes are done
        key = cv2.waitKey(1) & 0xFF
        if current_lane_idx >= len(LANE_NAMES) and key != 255: # Any key press after finishing
            break
        elif key == ord('q'):
            print("Calibration aborted.")
            break

    cv2.destroyAllWindows()
    
    # Generate the output for tracker.py
    if len(final_rois) == len(LANE_NAMES):
        print("\n" + "="*50)
        print("[COPY] COPY AND PASTE THIS INTO `utils/tracker.py`:")
        print("="*50)
        print("LANE_ROIS = {")
        for name, points in final_rois.items():
            pts_str = ", ".join([f"[{x}, {y}]" for x, y in points])
            print(f'    "{name}": np.array([{pts_str}], np.int32),')
        print("}")
        print("="*50)

if __name__ == "__main__":
    run_selector()