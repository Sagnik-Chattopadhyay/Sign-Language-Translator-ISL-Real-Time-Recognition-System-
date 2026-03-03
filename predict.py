import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
import cv2
import mediapipe as mp
from stgcn import Model
from final_preprocessing import process_video_to_skeleton

# MediaPipe Setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# CONFIGURATION
DATA_ROOT = r"e:/projects/sign language/Videos_tensors"
CHECKPOINT_DIR = "checkpoints"
NUM_CLASSES = 101 # Standard for this dataset
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_classes(data_root):
    classes = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
    return classes

def greedy_decoder(outputs, classes):
    """
    Decodes CTC Output (T, N, C) or (N, T, C) -> Label String
    """
    # outputs: (1, T, Num_Classes)
    probs = F.softmax(outputs, dim=2)
    max_probs, indices = torch.max(probs, dim=2)
    
    indices = indices.squeeze().cpu().numpy() # (T,)
    
    decoded_sequence = []
    last_idx = -1
    
    for idx in indices:
        if idx != last_idx: # Merge repeats
            if idx != 0: # 0 is Blank
                # Map back: Model Index 1 -> Class Index 0
                class_idx = idx - 1
                if 0 <= class_idx < len(classes):
                    decoded_sequence.append(classes[class_idx])
            last_idx = idx
            
    return " ".join(decoded_sequence)

def annotate_video(video_path, output_path, text):
    print(f"Generating annotated video: {output_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video for annotation.")
        return

    # Video Writer
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Draw Skeleton
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                                     mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                     mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
            
            # Draw Subtitle Box
            cv2.rectangle(image, (0, height-60), (width, height), (0, 0, 0), -1)
            
            # Draw Text (Centered)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 2
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = (width - text_size[0]) // 2
            text_y = height - 20
            
            cv2.putText(image, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
            
            out.write(image)
            
    cap.release()
    out.release()
    print("Video annotation complete.")

def predict(video_path, model_path=None, output_path=None):
    # 1. Load Classes
    classes = load_classes(DATA_ROOT)
    ctc_num_classes = len(classes) + 1
    
    # 2. Load Model
    print(f"Loading Model with {ctc_num_classes} classes (incl. blank)...")
    graph_args = {'strategy': 'spatial'}
    model = Model(in_channels=3, num_class=ctc_num_classes, graph_args=graph_args, edge_importance_weighting=True)
    
    if model_path is None:
        # Find latest
        checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pth")]
        if not checkpoints:
            print("No checkpoints found!")
            return
        latest = max(checkpoints, key=lambda x: int(x.split('_')[2].split('.')[0]))
        model_path = os.path.join(CHECKPOINT_DIR, latest)
    
    print(f"Loading weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    # 3. Process Video
    print(f"Processing video: {video_path}")
    try:
        # Returns (T, V, 3)
        skeleton = process_video_to_skeleton(video_path) 
    except Exception as e:
        print(f"Error processing video: {e}")
        return

    # 4. Preprocess for Model
    # ==========================
    # NORMALIZATION (CRITICAL)
    # ==========================
    # Must match dataset.py: Center around first frame's nose
    if skeleton.shape[0] > 0:
        origin = skeleton[0, 0, :] # Nose of first frame
        skeleton = skeleton - origin
        
    # (T, V, 3) -> (3, T, V)
    data = skeleton.transpose(2, 0, 1)
    # Add Person: (3, T, V, 1)
    data = data[:, :, :, np.newaxis]
    # Add Batch: (1, 3, T, V, 1)
    data = data[np.newaxis, :, :, :, :]
    
    tensor = torch.from_numpy(data).float().to(DEVICE)
    
    # 5. Inference
    with torch.no_grad():
        # Output: (1, T_out, Num_Classes)
        outputs = model(tensor)
        
    # 6. Decode
    result = greedy_decoder(outputs, classes)
    print("\n" + "="*30)
    print(f"PREDICTION: {result}")
    print("="*30 + "\n")
    
    # 7. Annotate Video
    if output_path is None:
        name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = f"pred_{name}.mp4"
        
    annotate_video(video_path, output_path, f"Prediction: {result}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--model", type=str, default=None, help="Path to checkpoint (optional)")
    parser.add_argument("--output", type=str, default=None, help="Path to output video (optional)")
    args = parser.parse_args()
    
    predict(args.video, args.model, args.output)
