import cv2
import numpy as np
import os

# CONFIG
NPY_PATH = r"e:/projects/sign language/Videos_tensors/He is going into the room/MVI_6503.npy"
VIDEO_PATH = r"e:/projects/sign language/ISL_CSLRT_Corpus/ISL_CSLRT_Corpus/Videos_Sentence_Level/He is going into the room/MVI_6503.MP4"
OUTPUT_VIDEO = "verification_output01.mp4"

import mediapipe as mp

# LANDMARK INDICES RANGES
# 33 Pose + 21 Left + 21 Right + 44 Face
POSE_START = 0
LH_START = 33
RH_START = 33 + 21
FACE_START = 33 + 21 + 21

# Load connection topologies
mp_holistic = mp.solutions.holistic
POSE_CONNECTIONS = mp_holistic.POSE_CONNECTIONS
HAND_CONNECTIONS = mp_holistic.HAND_CONNECTIONS

def draw_landmarks(frame, landmarks):
    h, w, _ = frame.shape
    
    # helper
    def to_pix(x, y):
        return int(x * w), int(y * h)

    def draw_line(idx1, idx2, color):
        p1 = landmarks[idx1]
        p2 = landmarks[idx2]
        # Check for visibility/zero
        if (p1[0] == 0 and p1[1] == 0) or (p2[0] == 0 and p2[1] == 0):
            return
        cv2.line(frame, to_pix(p1[0], p1[1]), to_pix(p2[0], p2[1]), color, 2)

    def draw_point(idx, color, radius=3):
        p = landmarks[idx]
        if p[0] == 0 and p[1] == 0:
            return
        cv2.circle(frame, to_pix(p[0], p[1]), radius, color, -1)

    # ------------------
    # POSE
    # ------------------
    # Draw connections
    for p1, p2 in POSE_CONNECTIONS:
        draw_line(p1, p2, (0, 255, 0)) # Green Lines
    
    # Draw points
    for i in range(POSE_START, LH_START):
        draw_point(i, (0, 200, 0), 4)

    # ------------------
    # LEFT HAND
    # ------------------
    # Draw connections
    for p1, p2 in HAND_CONNECTIONS:
        draw_line(p1 + LH_START, p2 + LH_START, (0, 0, 255)) # Red Lines
    
    # Draw points
    for i in range(LH_START, RH_START):
        draw_point(i, (0, 0, 200), 3)

    # ------------------
    # RIGHT HAND
    # ------------------
    # Draw connections
    for p1, p2 in HAND_CONNECTIONS:
        draw_line(p1 + RH_START, p2 + RH_START, (255, 0, 0)) # Blue Lines
        
    # Draw points
    for i in range(RH_START, FACE_START):
        draw_point(i, (200, 0, 0), 3)

    # ------------------
    # FACE (Subset)
    # ------------------
    # Just drawing points for face as we have a sparse subset
    for i in range(FACE_START, len(landmarks)):
        draw_point(i, (0, 255, 255), 1) # Yellow Dots

def verify():
    if not os.path.exists(NPY_PATH):
        print(f"NPY not found: {NPY_PATH}")
        return
    if not os.path.exists(VIDEO_PATH):
        print(f"Video not found: {VIDEO_PATH}")
        return

    skeleton_data = np.load(NPY_PATH)
    print(f"Loaded skeleton data shape: {skeleton_data.shape}")

    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video Info: {width}x{height} @ {fps}fps, {length} frames")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if idx < len(skeleton_data):
            draw_landmarks(frame, skeleton_data[idx])
        
        out.write(frame)
        idx += 1

    cap.release()
    out.release()
    print(f"Verification video saved to {os.path.abspath(OUTPUT_VIDEO)}")

if __name__ == "__main__":
    verify()
