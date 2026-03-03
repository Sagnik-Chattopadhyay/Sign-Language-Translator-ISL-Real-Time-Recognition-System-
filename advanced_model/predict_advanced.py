import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
import sys
import json
import cv2
import mediapipe as mp

# Add parent dir to path to find stgcn.py if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from st_transformer import SignTransformer
from final_preprocessing import process_video_to_skeleton

# CONFIGURATION
SENTENCE_MAP_FILE = r"e:/projects/sign language/advanced_model/sentence_map.json"
CHECKPOINT_PATH = r"e:/projects/sign language/advanced_model/checkpoints_sentence/sentence_model_epoch_100.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MediaPipe for Annotation
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def load_vocab(map_file):
    with open(map_file, 'r') as f:
        data = json.load(f)
        return data['vocab']

def ctc_decode(log_probs, vocab):
    """
    Decodes CTC Output (T, N, C) -> Text
    """
    # log_probs: (1, T, Num_Classes)
    # Argmax
    probs = torch.exp(log_probs)
    max_probs, indices = torch.max(probs, dim=2)
    indices = indices.squeeze().cpu().numpy()
    
    decoded = []
    last_idx = -1
    blank_idx = 0
    
    for idx in indices:
        if idx != last_idx:
            if idx != blank_idx:
                # Map ID -> Word
                # ID 0 is Blank. ID 1 is Vocab[0].
                word_idx = idx - 1
                if 0 <= word_idx < len(vocab):
                    decoded.append(vocab[word_idx])
            last_idx = idx
            
    return " ".join(decoded)

def annotate_video(video_path, output_path, text):
    print(f"Generating annotated video: {output_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video.")
        return

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
                
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Skeleton
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                                     mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                     mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
            
            # Subtitle
            cv2.rectangle(image, (0, height-60), (width, height), (0, 0, 0), -1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = (width - text_size[0]) // 2
            text_y = height - 15
            cv2.putText(image, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
            
            out.write(image)
            
    cap.release()
    out.release()
    print("Annotation Complete.")

def predict(video_path, output_path=None):
    # 1. Load Vocab
    vocab = load_vocab(SENTENCE_MAP_FILE)
    num_classes = len(vocab) + 1
    
    # 2. Load Model
    print(f"Loading Model ({num_classes} classes)...")
    model = SignTransformer(num_classes=num_classes, phase='pretrain') # CTC mode uses 'pretrain' structure
    
    if not os.path.exists(CHECKPOINT_PATH):
        print("Checkpoint not found!")
        return
        
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    # 3. Process Video
    print(f"Processing Video: {video_path}")
    try:
        skeleton = process_video_to_skeleton(video_path) # (T, 119, 3)
    except Exception as e:
        print(f"Error: {e}")
        return
        
    # 4. Preprocess (Center + Reshape)
    if skeleton.shape[0] > 0:
        origin = skeleton[0, 0, :]
        skeleton = skeleton - origin
        
    data = skeleton.transpose(2, 0, 1) # (3, T, 119)
    data = data[:, :, :, np.newaxis] # (3, T, 119, 1)
    data = data[np.newaxis, :, :, :, :] # (1, 3, T, 119, 1)
    
    tensor = torch.from_numpy(data).float().to(DEVICE)
    
    # 5. Inference
    with torch.no_grad():
        # Encode features
        features = model.encoder(tensor) # (N, 256, T_out)
        features = features.permute(0, 2, 1) # (N, T_out, 256)
        
        # Classify
        outputs = model.classifier(features) # (N, T_out, Num_Classes)
        
        # Log Softmax
        # (N, T_out, Num_Classes) -> Permute? No, just keep batch first for manual decode if needed
        # But ctc_decode usually takes (1, T, C)
        
    # 6. Decode
    result = ctc_decode(outputs, vocab)
    print("\n" + "="*40)
    print(f"PREDICTED TRANSLATION: {result}")
    print("="*40 + "\n")
    
    with open("prediction_result.txt", "w") as f:
        f.write(result)
        
    # 7. Annotate
    if output_path is None:
        name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = f"advanced_pred_{name}.mp4"
        
    annotate_video(video_path, output_path, f"Predicted: {result}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--output", type=str, default=None, help="Output video path")
    args = parser.parse_args()
    
    predict(args.video, args.output)
