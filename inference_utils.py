import os
import torch
import torch.nn.functional as F
import numpy as np
import mediapipe as mp
import cv2

# Ensure stgcn is found (assuming stgcn.py is in the root directory like realtime_predict.py)
from stgcn import Model

class SignLanguageModel:
    def __init__(self, data_root="Videos_tensors", checkpoint_dir="checkpoints", device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_root = data_root
        self.checkpoint_dir = checkpoint_dir
        
        self.classes = self._load_classes()
        self.ctc_metrics = len(self.classes) + 1
        self.model = self._load_model()
        
        print(f"Model loaded successfully on {self.device}")

    def _load_classes(self):
        if not os.path.exists(self.data_root):
            print(f"Warning: Data root {self.data_root} not found. Ensure the directory exists.")
            return []
        return sorted([d for d in os.listdir(self.data_root) if os.path.isdir(os.path.join(self.data_root, d))])

    def _load_model(self):
        graph_args = {'strategy': 'spatial'}
        model = Model(in_channels=3, num_class=self.ctc_metrics, graph_args=graph_args, edge_importance_weighting=True)
        
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            print(f"Created checkpoint directory: {self.checkpoint_dir}")

        checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.endswith(".pth")]
        
        # If no checkpoints exist (e.g., during first cloud deployment before download),
        # return the uninitialized model so the server can start and download it later.
        if not checkpoints:
            print(f"Warning: No .pth checkpoints found in {self.checkpoint_dir}. Model weights are randomly initialized.")
            model.to(self.device)
            return model
            
        latest = max(checkpoints, key=lambda x: int(x.split('_')[2].split('.')[0]))
        model_path = os.path.join(self.checkpoint_dir, latest)
        
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def predict_sequence(self, sequence):
        """
        Runs inference on a sequence of keypoints of shape (T, 119, 3)
        """
        # (T, 119, 3) -> (100, 119, 3)
        data = np.array(sequence)
        
        # NORMALIZATION (Centering)
        if data.shape[0] > 0:
            origin = data[0, 0, :] # First frame nose
            data = data - origin
            
        # Reshape for ST-GCN: (N, C, T, V, M)
        # (T, 119, 3) -> (3, T, 119) -> (1, 3, T, 119, 1)
        data = data.transpose(2, 0, 1)
        data = data[np.newaxis, :, :, :, np.newaxis]
        
        input_tensor = torch.from_numpy(data).float().to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            sentence = self.greedy_decoder(output)
            
        return sentence

    def greedy_decoder(self, outputs):
        prob = F.softmax(outputs, dim=2)
        max_probs, indices = torch.max(prob, dim=2)
        indices = indices.squeeze().cpu().numpy()
        
        # Ensure indices is iterable even if scalar
        if indices.ndim == 0:
            indices = [indices.item()]

        decoded = []
        last_idx = -1
        for idx in indices:
            if idx != last_idx:
                if idx != 0:
                    if idx - 1 < len(self.classes):
                        decoded.append(self.classes[idx-1])
                last_idx = idx
        return " ".join(decoded)

def extract_keypoints(results):
    """
    Extracts 119 keypoints: Pose(33) + Left Hand(21) + Right Hand(21) + Face(44 selected)
    """
    # 1. Pose: 33 landmarks
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
        
    # 4. Face: 44 selected landmarks
    FACE_INDICES = (
        list(range(61, 81)) +      # mouth (20)
        list(range(291, 311)) +    # mouth (20)
        [33, 133, 362, 263]        # eyes (4)
    )
    
    if results.face_landmarks:
        face_all = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark])
        try:
             face = face_all[FACE_INDICES]
        except:
             face = np.zeros((44, 3))
    else:
        face = np.zeros((44, 3))

    return np.concatenate([pose, lh, rh, face])
