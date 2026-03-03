
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import cv2
import mediapipe as mp
import argparse
import sys
import os

# Use Agg backend for non-interactive environments
plt.switch_backend('Agg')

# =========================
# MEDIAPIPE CONFIG
# =========================
mp_holistic = mp.solutions.holistic
FACE_INDICES = (
    list(range(61, 81)) +      # mouth
    list(range(291, 311)) +    # mouth
    [33, 133, 362, 263]        # eyes
)

def get_edges():
    """
    Returns the list of edges (connections between joints) for the 119-point skeleton.
    Based on graph.py logic.
    """
    neighbor_link = []

    # =====================================================
    # 1. POSE EDGES (MediaPipe Pose Topology)
    # =====================================================
    pose_connections = [
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5),
        (5, 6), (6, 8), (9, 10), (11, 12), (11, 13),
        (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
        (18, 20), (11, 23), (12, 24), (23, 24), (23, 25),
        (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
        (29, 31), (30, 32), (27, 31), (28, 32)
    ]
    neighbor_link += pose_connections

    # =====================================================
    # 2. HAND EDGES
    # =====================================================
    hand_connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),         # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),         # Index
        (0, 9), (9, 10), (10, 11), (11, 12),    # Middle
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
        (0, 17), (17, 18), (18, 19), (19, 20)   # Pinky
    ]
    # Left Hand (Offset 33)
    neighbor_link += [(i + 33, j + 33) for i, j in hand_connections]
    # Right Hand (Offset 54)
    neighbor_link += [(i + 54, j + 54) for i, j in hand_connections]

    # =====================================================
    # 4. INTER-PART CONNECTIONS
    # =====================================================
    neighbor_link.append((15, 33))  # L Wrist -> L Hand Root
    neighbor_link.append((16, 54))  # R Wrist -> R Hand Root
    # Face connections removed

    return neighbor_link

def process_video_to_skeleton(video_path):
    """
    Extracts skeleton from video using MediaPipe.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    skeleton_sequence = []
    
    print(f"Processing video: {video_path}...")
    
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1, 
        smooth_landmarks=True
    ) as holistic:
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame)
            
            joints = []
            
            # Helper to add joints
            def add_joints(landmarks, num):
                if landmarks:
                    for lm in landmarks.landmark:
                        joints.append([lm.x, -lm.y]) # Flip Y for correct plot orientation
                else:
                    joints.extend([[0.0, 0.0]] * num)

            # Note: We only need X, Y for 2D plotting, Z is optional but good for 3D
            # Let's stick to 2D projections for the graph plot as per request image which looks 2D/3D mixed.
            # Using X and -Y (because image coords Y is down)
            
            # Pose (33)
            add_joints(results.pose_landmarks, 33)
            # Left Hand (21)
            add_joints(results.left_hand_landmarks, 21)
            # Right Hand (21)
            add_joints(results.right_hand_landmarks, 21)
            
            # Face (Removed)
                
            skeleton_sequence.append(joints)
            frame_count += 1
            
    cap.release()
    print(f"Extracted {len(skeleton_sequence)} frames.")
    
    # Get a sample frame (middle frame) for visualization background
    cap = cv2.VideoCapture(video_path)
    if len(skeleton_sequence) > 0:
        mid_frame_idx = len(skeleton_sequence) // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame = None
    else:
        frame = None
    cap.release()
    
    return np.array(skeleton_sequence), frame

def plot_graph(skeleton_data, edges, background_frame=None, output_file='st_graph_video.png'):
    """
    Generates the plot.
    skeleton_data: (T, V, 2)
    """
    T, V, C = skeleton_data.shape
    
    # Use 10 frames for ST-Graph to show the flow nicely
    num_frames_to_plot = 10
    if T > num_frames_to_plot:
        # Pick frames centered around the middle
        mid = T // 2
        start = max(0, mid - num_frames_to_plot // 2)
        end = min(T, start + num_frames_to_plot)
        indices = np.arange(start, end)
        data_to_plot = skeleton_data[indices]
    else:
        data_to_plot = skeleton_data
        num_frames_to_plot = T
        
    fig = plt.figure(figsize=(12, 6))
    
    # ---------------------------------------------------------
    # 1. Spatial Graph (Overlaid on Video Frame)
    # ---------------------------------------------------------
    ax1 = fig.add_subplot(121)
    ax1.set_title("(a) Spatial Graph (on Video)")
    
    # Show background image if available
    if background_frame is not None:
        ax1.imshow(background_frame)
        img_h, img_w = background_frame.shape[:2]
        # Skeleton is normalized 0-1? No, MediaPipe is normalized 0-1.
        # Need to scale to image size.
        scale_x, scale_y = img_w, img_h
    else:
        scale_x, scale_y = 1, 1
        ax1.invert_yaxis() # If no image, flip Y back to normal plot (up is positive)
    
    # Plot middle frame from the selected slice
    mid_idx_slice = num_frames_to_plot // 2
    skel_spatial_norm = data_to_plot[mid_idx_slice] # Normalized
    
    # Scale coordinates
    skel_spatial = skel_spatial_norm.copy()
    skel_spatial[:, 0] *= scale_x
    skel_spatial[:, 1] *= -scale_y if background_frame is None else scale_y # If image, Y is down (positive), no flip needed from MP (which is 0-1 top-down).
    # Wait, my previous extraction flipped Y: `joints.append([lm.x, -lm.y])`
    # Let's revert that flip if we are plotting on image.
    
    # Actually, let's fix the extraction to be raw (0-1) and handle plotting here.
    # Current extraction: `joints.append([lm.x, -lm.y])`
    # If plotting on image (Y down), we want `lm.y * H`.
    # Let's assume input skel is (X, -Y) from previous steps. 
    # To map to image (0 at top, H at bottom):
    # Image Y = -InputY * H.
    
    # Plot nodes
    ax1.scatter(skel_spatial[:, 0], -skel_spatial[:, 1] if background_frame is not None else skel_spatial[:, 1], c='red', s=15, alpha=0.8)
    
    # Plot edges
    for i, j in edges:
        p1 = skel_spatial[i]
        p2 = skel_spatial[j]
        # Check zeros (missing)
        if np.abs(p1[0]) > 0.001 and np.abs(p2[0]) > 0.001: 
             x_vals = [p1[0], p2[0]]
             y_vals = [-p1[1], -p2[1]] if background_frame is not None else [p1[1], p2[1]]
             ax1.plot(x_vals, y_vals, c='yellow', linewidth=1.5, alpha=0.6)
    
    ax1.axis('off')

    # ---------------------------------------------------------
    # 2. Spatio-Temporal Graph (3D Stack)
    # ---------------------------------------------------------
    ax3 = fig.add_subplot(122, projection='3d')
    ax3.set_title("(b) Spatio-Temporal Graph")
    
    # "Landscape" orientation: Time on X axis.
    # X_plot = Time
    # Y_plot = Spatial X
    # Z_plot = Spatial Y
    
    # Color gradient for time
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, num_frames_to_plot))
    
    for t in range(num_frames_to_plot):
        skel = data_to_plot[t]
        
        # Time mapped to X-axis
        # Using a spacing factor
        t_pos = t * 0.2
        time_arr = np.full(skel.shape[0], t_pos)
        
        # Valid points mask
        valid = (np.abs(skel[:, 0]) > 0.001)
        
        # Spatial Coordinates (Y_plot=X, Z_plot=Y)
        xs_spatial = skel[valid, 0]
        ys_spatial = skel[valid, 1] 
        # Note: Spatial Y was inverted in 2D plot. 
        # In 3D, let's keep upright. Input is (X, -Y).
        # So Z_plot = -Y_spatial to be upright?
        # Actually input extraction was: joints.append([lm.x, -lm.y])
        # So ys_spatial is already negative.
        # If we plot raw ys_spatial on Z, it will be upside down (-Y is down).
        # We want Z to be UP. So we should plot -ys_spatial?
        # Input: (0.5, -0.2) -> Head. (0.5, -0.8) -> Feet.
        # Z-axis: -0.2 is higher than -0.8. So plotting raw ys_spatial works if Z-axis is standard.
        
        # Plot Scatter
        # X=Time, Y=SpatialX, Z=SpatialY
        ax3.scatter(time_arr[valid], xs_spatial, ys_spatial, c=[colors[t]], s=10, alpha=0.7)
        
        # Draw Spatial Edges (within frame)
        for i, j in edges:
            if valid[i] and valid[j]:
                p1, p2 = skel[i], skel[j]
                # x=Time, y=p[0], z=p[1]
                ax3.plot([t_pos, t_pos], [p1[0], p2[0]], [p1[1], p2[1]], c=colors[t], linewidth=1, alpha=0.5)
        
        # Draw Temporal Edges (between frames)
        if t < num_frames_to_plot - 1:
            next_skel = data_to_plot[t+1]
            next_t_pos = (t + 1) * 0.2
            next_valid = (np.abs(next_skel[:, 0]) > 0.001)
            
            # Connect joints
            for i in range(len(skel)):
                if valid[i] and next_valid[i]:
                    p1 = skel[i]
                    p2 = next_skel[i]
                    ax3.plot([t_pos, next_t_pos], [p1[0], p2[0]], [p1[1], p2[1]], c='gray', alpha=0.2, linewidth=0.5, linestyle='--')

    # Remove all axis labels and ticks
    ax3.set_xlabel('')
    ax3.set_ylabel('')
    ax3.set_zlabel('')
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_zticks([])
    
    # Hide the panes (transparent background)
    ax3.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax3.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax3.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # Hide the axis lines
    ax3.set_axis_off()

    # Set view angle: Look from the "front-side" so Time goes Left->Right
    # elev=10 (slightly up), azim=-90 (Standard Side view?)
    # Default azim=-60. 
    # Try azim=-90? 
    # If X is Time. Y is Depth. Z is Height.
    # Front view (X-Z plane) -> azim=-90.
    ax3.view_init(elev=10, azim=-110) # Angled slightly to see depth
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Graph visualization saved to {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    args = parser.parse_args()
    
    try:
        data, frame = process_video_to_skeleton(args.video)
        edges = get_edges()
        
        # If output filename not specified, derive from video name
        vid_name = os.path.splitext(os.path.basename(args.video))[0]
        out_name = f'st_graph_{vid_name}.png'
        
        plot_graph(data, edges, background_frame=frame, output_file=out_name)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
