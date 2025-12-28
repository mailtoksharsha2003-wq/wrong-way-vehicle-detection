import cv2
import numpy as np
from ultralytics import YOLO
from sort.sort import Sort
from collections import deque, defaultdict
import os
from tqdm import tqdm
import shutil


# ====================== ENHANCED SETTINGS ==========================
video_path = r"path"
output_root = r"path"
os.makedirs(output_root, exist_ok=True)
existing = [d for d in os.listdir(output_root) if d.startswith("wrongside_output_")]
output_dir = os.path.join(output_root, f"wrongside_output_{len(existing)+1}")
os.makedirs(output_dir, exist_ok=True)
video_output_path = os.path.join(output_dir, "output_final.mp4")

fps = 30
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Enhanced history and thresholds
history_length_direction = 30   # Increased for better stability
history_length_wrongway = 15    # Increased for confirmation
hysteresis_frames = 7           # More frames for direction change confirmation

# Timing thresholds
RED_HOLD_FRAMES = int(2 * fps)     # must stay red for at least 2 sec
GREEN_HOLD_FRAMES = int(3 * fps)   # must stay green for 3 sec before valid red detection

# Enhanced detection settings
CONFIDENCE_THRESHOLD = 0.25        # Better balance for small objects
IOU_THRESHOLD = 0.3               # Adjusted for better tracking
MIN_DETECTION_SIZE = 25           # Minimum bbox size to consider
MAX_EGO_MOTION = 10.0            # Maximum allowed ego-motion per frame
USE_MULTI_SCALE = True           # Enable multi-scale detection
ENABLE_SAHI = False              # Enable SAHI for small object detection (requires sahi package)

# SAHI settings (if enabled)
SAHI_SLICE_SIZE = 640
SAHI_OVERLAP_RATIO = 0.2

# ================== GLOBAL VARIABLES ======================
cap = cv2.VideoCapture(video_path)
frame_size = None
out = None

# Enhanced background subtractor with better parameters
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500, 
    varThreshold=50, 
    detectShadows=True
)

centroid_history_direction = {}
centroid_history_wrongway = {}
velocity_history = {}
fixed_directions = {}
direction_confirm_counter = {}
direction_timers = {}
wrongway_saved = set()
ego_motion_history = deque(maxlen=30)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# New: Store smoothed positions using simple moving average
smoothed_positions = {}
position_buffer_size = 5

# New: Track detection quality
detection_quality = {}
blur_threshold = 100.0  # Laplacian variance threshold


# ================= ENHANCED HELPER FUNCTIONS ======================

def detect_motion_blur(frame):
    """
    Detect if frame has motion blur using Laplacian variance
    Returns: (is_blurry, variance_score)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < blur_threshold, variance


def load_model_tracker():
    """Load YOLOv8 model with optimized settings"""
    # Use medium model for better accuracy on small objects
    model = YOLO('yolov8m.pt')
    
    # Enhanced SORT tracker with better parameters
    tracker = Sort(
        max_age=50,      # Increased for better long-term tracking
        min_hits=3,      # Require more hits before confirming track
        iou_threshold=IOU_THRESHOLD
    )
    return model, tracker


def calculate_centroid(box):
    """Calculate centroid of bounding box"""
    x1, y1, x2, y2 = box
    return int((x1 + x2)/2), int((y1 + y2)/2)


def expand_bbox(bbox, frame_shape, scale=1.4):
    """Expand bounding box for vehicle crop"""
    x1, y1, x2, y2 = map(int, bbox)
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w/2, y1 + h/2
    new_w, new_h = w*scale, h*scale
    new_x1 = max(0, int(cx - new_w/2))
    new_y1 = max(0, int(cy - new_h/2))
    new_x2 = min(frame_shape[1]-1, int(cx + new_w/2))
    new_y2 = min(frame_shape[0]-1, int(cy + new_h/2))
    
    # Shift upward slightly
    shift_y = int((new_y2 - new_y1) * 0.15)
    new_y1 = max(0, new_y1 - shift_y)
    new_y2 = min(frame_shape[0]-1, new_y2 - shift_y)
    return [new_x1, new_y1, new_x2, new_y2]


def smooth_position(track_id, current_position):
    """
    Apply simple moving average to smooth position tracking
    """
    if track_id not in smoothed_positions:
        smoothed_positions[track_id] = deque(maxlen=position_buffer_size)
    
    smoothed_positions[track_id].append(current_position)
    
    if len(smoothed_positions[track_id]) > 0:
        avg_x = np.mean([p[0] for p in smoothed_positions[track_id]])
        avg_y = np.mean([p[1] for p in smoothed_positions[track_id]])
        return (int(avg_x), int(avg_y))
    
    return current_position


def improved_ego_motion_estimation(prev_gray, curr_gray, vehicle_boxes=None):
    """
    Enhanced ego-motion estimation using ORB features with RANSAC
    More robust than simple optical flow
    """
    # Create mask excluding vehicle regions
    mask = np.ones(prev_gray.shape, dtype=np.uint8) * 255
    if vehicle_boxes is not None and len(vehicle_boxes) > 0:
        for box in vehicle_boxes:
            x1, y1, x2, y2 = map(int, box)
            # Expand exclusion area slightly
            margin = 10
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(mask.shape[1], x2 + margin)
            y2 = min(mask.shape[0], y2 + margin)
            mask[y1:y2, x1:x2] = 0
    
    # Use ORB detector for robust feature detection
    orb = cv2.ORB_create(nfeatures=1000)
    
    # Detect keypoints and compute descriptors
    kp1, des1 = orb.detectAndCompute(prev_gray, mask)
    kp2, des2 = orb.detectAndCompute(curr_gray, None)
    
    if des1 is None or des2 is None or len(kp1) < 10:
        # Fallback to optical flow if ORB fails
        return optical_flow_ego_motion(prev_gray, curr_gray, vehicle_boxes)
    
    # Match features using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    if len(matches) < 10:
        return optical_flow_ego_motion(prev_gray, curr_gray, vehicle_boxes)
    
    # Extract matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    # Use RANSAC to find robust transformation
    try:
        # Estimate affine transformation
        M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        
        if M is not None and inliers is not None:
            # Extract vertical translation (dy)
            dy = M[1, 2]
            
            # Limit extreme values
            dy = np.clip(dy, -MAX_EGO_MOTION, MAX_EGO_MOTION)
            return dy
    except:
        pass
    
    # Fallback to median
    dy_values = dst_pts[:, 1] - src_pts[:, 1]
    dy = np.median(dy_values)
    dy = np.clip(dy, -MAX_EGO_MOTION, MAX_EGO_MOTION)
    
    return dy


def optical_flow_ego_motion(prev_gray, curr_gray, vehicle_boxes=None):
    """
    Fallback optical flow-based ego-motion estimation
    """
    mask = np.ones(prev_gray.shape, dtype=np.uint8)
    if vehicle_boxes is not None and len(vehicle_boxes) > 0:
        for box in vehicle_boxes:
            x1, y1, x2, y2 = map(int, box)
            mask[y1:y2, x1:x2] = 0
    
    features = cv2.goodFeaturesToTrack(
        prev_gray, 
        mask=mask, 
        maxCorners=500, 
        qualityLevel=0.3, 
        minDistance=7
    )
    
    if features is None or len(features) < 5:
        return 0
    
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, features, None)
    
    if st is None or np.sum(st) < 5:
        return 0
    
    good_old = features[st == 1]
    good_new = p1[st == 1]
    
    dy_values = good_new[:, 1] - good_old[:, 1]
    
    # Use robust statistics
    dy = np.median(dy_values)
    dy = np.clip(dy, -MAX_EGO_MOTION, MAX_EGO_MOTION)
    
    return dy


def determine_direction_from_history(track_id, ego_dy=0):
    """
    Enhanced direction determination with better noise filtering
    """
    if track_id not in centroid_history_direction:
        return "Neutral"
    
    history = list(centroid_history_direction[track_id])
    
    if len(history) < 10:  # Need more data
        return "Neutral"
    
    # Extract y-coordinates
    y_coords = [c[1] for c in history]
    
    # Apply moving average smoothing
    window_size = min(5, len(y_coords))
    smoothed_y = np.convolve(y_coords, np.ones(window_size)/window_size, mode='valid')
    
    if len(smoothed_y) < 5:
        return "Neutral"
    
    # Calculate derivatives
    y_deltas = np.diff(smoothed_y)
    
    if len(y_deltas) < 3:
        return "Neutral"
    
    # Robust statistics
    avg_dy = np.median(y_deltas)  # Use median instead of mean
    std_dy = np.std(y_deltas)
    
    # Adaptive threshold based on ego-motion
    base_threshold = 0.5
    threshold = max(base_threshold, abs(ego_dy) * 0.3)
    
    # Check consistency
    consistency = np.sum(np.sign(y_deltas) == np.sign(avg_dy)) / len(y_deltas)
    
    # Direction determination with consistency check
    if consistency > 0.7:  # At least 70% consistency
        if avg_dy > threshold and std_dy < 3:
            return "Wrongway"
        elif avg_dy < -threshold and std_dy < 3:
            return "Rightway"
    
    return "Neutral"


def multi_scale_detection(model, frame):
    """
    Perform multi-scale detection for better small object detection
    """
    scales = [640, 800, 960]  # Different input sizes
    all_detections = []
    
    for scale in scales:
        # Resize maintaining aspect ratio
        aspect_ratio = frame.shape[0] / frame.shape[1]
        target_height = int(scale * aspect_ratio)
        target_width = scale
        
        resized = cv2.resize(frame, (target_width, target_height))
        
        # Run detection
        results = model.predict(
            resized, 
            verbose=False,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD
        )
        
        detections = results[0].boxes.data.cpu().numpy()
        
        # Scale detections back to original frame size
        scale_x = frame.shape[1] / target_width
        scale_y = frame.shape[0] / target_height
        
        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det[:6]
            cls_id = int(cls_id)
            
            # Filter for vehicles
            if cls_id in [2, 3, 5, 7]:
                x1_orig = x1 * scale_x
                y1_orig = y1 * scale_y
                x2_orig = x2 * scale_x
                y2_orig = y2 * scale_y
                
                bbox_width = x2_orig - x1_orig
                bbox_height = y2_orig - y1_orig
                
                # Filter by minimum size
                if bbox_width >= MIN_DETECTION_SIZE and bbox_height >= MIN_DETECTION_SIZE:
                    all_detections.append([
                        x1_orig, y1_orig, x2_orig, y2_orig, conf, cls_id
                    ])
    
    # Apply NMS to remove duplicates across scales
    if len(all_detections) > 0:
        all_detections = np.array(all_detections)
        boxes = all_detections[:, :4]
        scores = all_detections[:, 4]
        
        # OpenCV NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), 
            scores.tolist(), 
            CONFIDENCE_THRESHOLD, 
            IOU_THRESHOLD
        )
        
        if len(indices) > 0:
            indices = indices.flatten()
            return all_detections[indices]
    
    return np.array([])


def single_scale_detection(model, frame):
    """
    Standard single-scale detection with enhanced parameters
    """
    # Resize for detection
    target_size = 640
    aspect_ratio = frame.shape[0] / frame.shape[1]
    target_height = int(target_size * aspect_ratio)
    
    small_frame = cv2.resize(frame, (target_size, target_height))
    
    # Run YOLOv8 detection
    results = model.predict(
        small_frame, 
        verbose=False,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        agnostic_nms=True  # Class-agnostic NMS
    )
    
    detections = results[0].boxes.data.cpu().numpy()
    
    # Scale back to original frame
    scale_x = frame.shape[1] / target_size
    scale_y = frame.shape[0] / target_height
    
    vehicle_detections = []
    
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det[:6]
        cls_id = int(cls_id)
        
        # Filter for vehicles: car, motorcycle, bus, truck
        if cls_id in [2, 3, 5, 7]:
            x1o = x1 * scale_x
            y1o = y1 * scale_y
            x2o = x2 * scale_x
            y2o = y2 * scale_y
            
            bbox_width = x2o - x1o
            bbox_height = y2o - y1o
            
            # Filter by minimum size
            if bbox_width >= MIN_DETECTION_SIZE and bbox_height >= MIN_DETECTION_SIZE:
                vehicle_detections.append([x1o, y1o, x2o, y2o, conf, cls_id])
    
    return np.array(vehicle_detections) if vehicle_detections else np.empty((0, 6))


def setDirectionColor(track, frame, frame_count, ego_dy, is_blurry):
    """
    Enhanced direction color setting with quality checks
    """
    track_id = int(track[4])
    bbox = track[:4]
    centroid = calculate_centroid(bbox)
    
    # Apply position smoothing
    smoothed_centroid = smooth_position(track_id, centroid)
    
    # Initialize tracking data structures
    if track_id not in centroid_history_direction:
        centroid_history_direction[track_id] = deque(maxlen=history_length_direction)
        velocity_history[track_id] = deque(maxlen=history_length_direction)
        direction_confirm_counter[track_id] = 0
        direction_timers[track_id] = {"green_frames": 0, "red_frames": 0}
        detection_quality[track_id] = []
    
    # Store smoothed centroid
    centroid_history_direction[track_id].append(smoothed_centroid)
    detection_quality[track_id].append(not is_blurry)
    
    # Calculate ego-motion compensated velocity
    if len(centroid_history_direction[track_id]) >= 2:
        prev_c = centroid_history_direction[track_id][-2]
        dy = (smoothed_centroid[1] - prev_c[1]) - ego_dy
        velocity_history[track_id].append(dy)
    
    # Determine direction
    dirn = determine_direction_from_history(track_id, ego_dy)
    
    # Direction confirmation with hysteresis
    if dirn != "Neutral":
        if track_id not in fixed_directions:
            direction_confirm_counter[track_id] += 1
            if direction_confirm_counter[track_id] >= hysteresis_frames:
                fixed_directions[track_id] = dirn
                direction_confirm_counter[track_id] = 0
        else:
            if fixed_directions[track_id] != dirn:
                direction_confirm_counter[track_id] += 1
                if direction_confirm_counter[track_id] >= hysteresis_frames:
                    fixed_directions[track_id] = dirn
                    direction_confirm_counter[track_id] = 0
            else:
                direction_confirm_counter[track_id] = 0
    
    # Visualization and saving
    if track_id in fixed_directions:
        direction = fixed_directions[track_id]
        timers = direction_timers[track_id]
        
        # Update timers
        if direction == "Rightway":
            timers["green_frames"] += 1
            timers["red_frames"] = 0
        elif direction == "Wrongway":
            timers["red_frames"] += 1
        else:
            timers["red_frames"] = timers["green_frames"] = 0
        
        # Color coding
        box_color = (0, 255, 0) if direction == "Rightway" else (0, 0, 255)
        
        # Draw bounding box and info
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
        cv2.circle(frame, smoothed_centroid, 5, (255, 255, 255), -1)
        
        # Add quality indicator
        quality_score = np.mean(detection_quality[track_id][-10:]) if len(detection_quality[track_id]) > 0 else 0
        quality_text = f"Q:{quality_score:.2f}"
        
        label = f"ID:{track_id} {direction} {quality_text}"
        cv2.putText(
            frame, label,
            (x1, y1 - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 255), 
            2
        )
        
        # Save wrong-way vehicle with quality checks
        if (direction == "Wrongway" 
            and timers["red_frames"] >= RED_HOLD_FRAMES
            and timers["green_frames"] >= GREEN_HOLD_FRAMES
            and track_id not in wrongway_saved
            and quality_score > 0.5):  # Only save if quality is good
            
            clean_frame = frame.copy()
            expanded_bbox = expand_bbox(bbox, frame.shape)
            x1e, y1e, x2e, y2e = map(int, expanded_bbox)
            vehicle_crop = clean_frame[y1e:y2e, x1e:x2e]
            
            wrongway_vehicle_dir = os.path.join(output_dir, "wrongway_vehicles")
            os.makedirs(wrongway_vehicle_dir, exist_ok=True)
            
            if vehicle_crop.size > 0:
                filename = f"vehicle_{track_id}_frame{frame_count}_q{quality_score:.2f}.jpg"
                cv2.imwrite(
                    os.path.join(wrongway_vehicle_dir, filename), 
                    vehicle_crop
                )
                wrongway_saved.add(track_id)
                print(f"✓ Saved wrong-way vehicle ID {track_id} at frame {frame_count}")


def execute():
    """Main execution function with all enhancements"""
    model, tracker = load_model_tracker()
    pbar = tqdm(total=total_frames, desc="Processing Frames")
    out = None
    
    # Read first frame for initialization
    ret, prev_frame = cap.read()
    if not ret:
        print("❌ Failed to read video")
        return
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Statistics
    total_detections = 0
    blurry_frames = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        pbar.update(1)
        
        # Initialize video writer
        if out is None:
            frame_size = (frame.shape[1], frame.shape[0])
            out = cv2.VideoWriter(video_output_path, fourcc, fps, frame_size)
        
        # Check for motion blur
        is_blurry, blur_score = detect_motion_blur(frame)
        if is_blurry:
            blurry_frames += 1
        
        # Convert to grayscale for ego-motion
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # ---------- DETECTION SECTION ----------
        if USE_MULTI_SCALE and frame_count % 3 == 0:  # Multi-scale every 3rd frame
            detections = multi_scale_detection(model, frame)
        else:
            detections = single_scale_detection(model, frame)
        
        total_detections += len(detections)
        
        # Extract bounding boxes for ego-motion estimation
        vehicle_boxes = detections[:, :4] if len(detections) > 0 else np.empty((0, 4))
        
        # Estimate ego-motion
        ego_dy = improved_ego_motion_estimation(prev_gray, curr_gray, vehicle_boxes)
        ego_motion_history.append(ego_dy)
        
        # Smooth ego-motion
        smoothed_ego_dy = np.median(list(ego_motion_history))
        
        prev_gray = curr_gray.copy()
        
        # Prepare detections for tracker (only bbox + score)
        tracker_input = vehicle_boxes if len(vehicle_boxes) > 0 else np.empty((0, 4))
        
        # Update tracker
        tracks = tracker.update(tracker_input)
        
        # Process each track
        for track in tracks:
            setDirectionColor(track, frame, frame_count, smoothed_ego_dy, is_blurry)
        
        # Add frame info overlay
        info_text = f"Frame: {frame_count} | Detections: {len(detections)} | Ego: {smoothed_ego_dy:.2f}"
        cv2.putText(
            frame, info_text, 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 255), 
            2
        )
        
        if is_blurry:
            cv2.putText(
                frame, "BLUR DETECTED", 
                (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 0, 255), 
                2
            )
        
        out.write(frame)
    
    # Cleanup
    pbar.close()
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    
    # Save script
    script_path = os.path.abspath(__file__)
    shutil.copy(script_path, os.path.join(output_dir, os.path.basename(script_path)))
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"✅ Processing complete!")
    print(f"{'='*60}")
    print(f"📊 Statistics:")
    print(f"   Total frames processed: {total_frames}")
    print(f"   Total detections: {total_detections}")
    print(f"   Blurry frames: {blurry_frames} ({blurry_frames/total_frames*100:.1f}%)")
    print(f"   Wrong-way vehicles saved: {len(wrongway_saved)}")
    print(f"   Output directory: {output_dir}")
    print(f"{'='*60}")


# ====================== RUN ==============================
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"🚗 ENHANCED WRONG-SIDE DETECTION SYSTEM")
    print(f"{'='*60}")
    print(f"⚙️  Configuration:")
    print(f"   Model: YOLOv8m (Medium)")
    print(f"   Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    print(f"   Multi-scale Detection: {USE_MULTI_SCALE}")
    print(f"   History Length: {history_length_direction}")
    print(f"   Min Detection Size: {MIN_DETECTION_SIZE}px")
    print(f"{'='*60}\n")
    
    try:
        execute()
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
