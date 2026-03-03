import cv2
import mediapipe as mp
import torch
import torch.nn.functional as F
import numpy as np
import os
import collections
import argparse
from stgcn import Model

# CONFIGURATION
DATA_ROOT = r"e:/projects/sign language/Videos_tensors"
CHECKPOINT_DIR = "checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_FRAMES = 100 # Matches training
CONFIDENCE_THRESHOLD = 0.5

# MediaPipe Setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def load_classes(data_root):
    return sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])

def extract_keypoints(results):
    """
    Extracts 119 keypoints: Pose(33) + Left Hand(21) + Right Hand(21) + Face(44 selected)
    Matches logic in final_preprocessing.py (simplified for speed)
    """
    # 1. Pose: 33 landmarks (x, y, z, visibility)
    if results.pose_landmarks:
        pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark])
    else:
        pose = np.zeros((33, 3))
        
    # 2. Left Hand: 21 landmarks
    if results.left_hand_landmarks:
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark])
    else:
        lh = np.zeros((21, 3))

    # 3. Right Hand: 21 landmarks
    if results.right_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark])
    else:
        rh = np.zeros((21, 3))
        
    # 4. Face: 468 landmarks -> Select specific 44
    # MATCHING final_preprocessing.py logic EXACTLY
    FACE_INDICES = (
        list(range(61, 81)) +      # mouth (20)
        list(range(291, 311)) +    # mouth (20)
        [33, 133, 362, 263]        # eyes (4)
    )
    
    if results.face_landmarks:
        face_all = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark])
        # Select specific indices
        try:
             face = face_all[FACE_INDICES]
        except:
             face = np.zeros((44, 3))
    else:
        face = np.zeros((44, 3))

    # Concatenate: (119, 3)
    # Note: If we use zeros for face, accuracy might drop for face-heavy signs, 
    # but Hand/Pose are dominant.
    # To match dimensions: Pose(33) + LH(21) + RH(21) + Face(44) = 119
    return np.concatenate([pose, lh, rh, face])

def greedy_decoder(outputs, classes):
    prob = F.softmax(outputs, dim=2)
    max_probs, indices = torch.max(prob, dim=2)
    indices = indices.squeeze().cpu().numpy()
    
    decoded = []
    last_idx = -1
    for idx in indices:
        if idx != last_idx:
            if idx != 0:
                decoded.append(classes[idx-1])
            last_idx = idx
    return " ".join(decoded)

def main():
    # 1. Load Resources
    classes = load_classes(DATA_ROOT)
    ctc_metrics = len(classes) + 1
    
    print("Loading Model...")
    graph_args = {'strategy': 'spatial'}
    model = Model(in_channels=3, num_class=ctc_metrics, graph_args=graph_args, edge_importance_weighting=True)
    
    # Find latest checkpoint
    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pth")]
    latest = max(checkpoints, key=lambda x: int(x.split('_')[2].split('.')[0]))
    model_path = os.path.join(CHECKPOINT_DIR, latest)
    
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print(f"Model Loaded: {latest}")

    # 2. Setup Input
    cap = cv2.VideoCapture(0)
    
    # buffers
    sequence = collections.deque(maxlen=MAX_FRAMES)
    sentence = ""
    frame_counter = 0
    inference_interval = 5 # Run model every 5 frames
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process Frame
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Draw Landmarks
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
            # Extract Keypoints
            keypoints = extract_keypoints(results) # (119, 3)
            sequence.append(keypoints)
            
            # Inference Block
            frame_counter += 1
            if len(sequence) == MAX_FRAMES and frame_counter % inference_interval == 0:
                # Prepare Data
                data = np.array(sequence) # (100, 119, 3)
                
                # NORMALIZATION (Centering)
                origin = data[0, 0, :] # First frame nose
                data = data - origin
                
                # Reshape for ST-GCN: (N, C, T, V, M)
                # (100, 119, 3) -> (3, 100, 119) -> (1, 3, 100, 119, 1)
                data = data.transpose(2, 0, 1)
                data = data[np.newaxis, :, :, :, np.newaxis]
                
                input_tensor = torch.from_numpy(data).float().to(DEVICE)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    sentence = greedy_decoder(output, classes)
            
            # Display
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, sentence, (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.imshow('ST-GCN Real-Time', image)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
