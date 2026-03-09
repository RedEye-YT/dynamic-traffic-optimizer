import cv2
import numpy as np

def get_box_center(xyxy):
    """Calculates the bottom-center (x, y) of a bounding box."""
    x1, y1, x2, y2 = xyxy
    # We use bottom-center because that's where the car touches the road
    return (int((x1 + x2) / 2), int(y2))

def process_frame_detections(boxes, frame=None, custom_rois=None):
    """
    Maps bounding boxes to lanes using ONLY Auto-Calibrated coordinates.
    Starts completely clean with no ugly default lines.
    """
    
    # Start with empty ROIs if the user hasn't clicked "Auto-Calibrate" yet
    active_rois = custom_rois if custom_rois is not None else {}

    # Always initialize base counts so the Streamlit UI doesn't crash before calibration
    counts = {"North": 0, "South": 0, "East": 0, "West": 0}
    ev_flag = 0

    if boxes is None:
        return counts, ev_flag, frame

    # Draw the custom polygons on the frame ONLY if they exist
    if frame is not None and active_rois:
        for lane, polygon in active_rois.items():
            poly_array = np.array(polygon, np.int32)
            cv2.polylines(frame, [poly_array], isClosed=True, color=(0, 255, 255), thickness=2)

    for box in boxes:
        # Get coordinates and class ID
        xyxy = box.xyxy[0].cpu().numpy()
        cls_id = int(box.cls[0])

        # Check for Emergency Vehicle (Assuming class 9 is EV)
        if cls_id == 9:
            ev_flag = 1

        # Get vehicle footprint location
        center = get_box_center(xyxy)

        # Check which ROI the vehicle is in using Point-in-Polygon test
        if active_rois:
            for lane, polygon in active_rois.items():
                poly_array = np.array(polygon, np.int32)
                
                # pointPolygonTest returns +1 if inside, 0 if on contour, -1 if outside
                if cv2.pointPolygonTest(poly_array, center, False) >= 0:
                    # Safely add to count
                    counts[lane] = counts.get(lane, 0) + 1

                    # Draw a red dot on the detected vehicle's tires for visual confirmation
                    if frame is not None:
                        cv2.circle(frame, center, 4, (0, 0, 255), -1)
                    break # Move to the next vehicle once assigned to a lane

    return counts, ev_flag, frame