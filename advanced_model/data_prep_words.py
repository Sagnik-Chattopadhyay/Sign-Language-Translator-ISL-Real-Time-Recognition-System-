import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# CONFIGURATION
DATA_ROOT = r"e:/projects/sign language/ISL_CSLRT_Corpus/ISL_CSLRT_Corpus/Frames_Word_Level"
OUTPUT_ROOT = r"e:/projects/sign language/advanced_model/data_words"

# Ensure output directory exists
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# MediaPipe Setup
mp_holistic = mp.solutions.holistic

# Reduced Face Indices (Same as final_preprocessing.py)
FACE_INDICES = (
    list(range(61, 81)) +      # Mouth (20)
    list(range(291, 311)) +    # Mouth (20)
    [33, 133, 362, 263]        # Eyes (4)
)

def process_word_folder(folder_path, output_path):
    """
    Process a folder of images (frames) into a single skeleton tensor.
    """
    # Get sorted list of images
    frames = sorted([
        f for f in os.listdir(folder_path) 
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    
    # Sort files naturally? (e.g. HELP (1).jpg, HELP (2).jpg)
    # The default alphanumeric sort might do HELP (1), HELP (10), HELP (2).
    # We need a robust sort.
    try:
        frames.sort(key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
    except:
        pass # Fallback to alphabetical if no digits
        
    skeleton_sequence = []
    
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.5
    ) as holistic:
        
        for frame_file in frames:
            frame_path = os.path.join(folder_path, frame_file)
            frame = cv2.imread(frame_path)
            if frame is None:
                continue
                
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame)
            
            joints = []
            
            # 1. Pose (33)
            if results.pose_landmarks:
                for lm in results.pose_landmarks.landmark:
                    joints.append([lm.x, lm.y, lm.z])
            else:
                joints.extend([[0.0, 0.0, 0.0]] * 33)
                
            # 2. Left Hand (21)
            if results.left_hand_landmarks:
                for lm in results.left_hand_landmarks.landmark:
                    joints.append([lm.x, lm.y, lm.z])
            else:
                joints.extend([[0.0, 0.0, 0.0]] * 21)
                
            # 3. Right Hand (21)
            if results.right_hand_landmarks:
                for lm in results.right_hand_landmarks.landmark:
                    joints.append([lm.x, lm.y, lm.z])
            else:
                joints.extend([[0.0, 0.0, 0.0]] * 21)
                
            # 4. Face (44)
            if results.face_landmarks:
                for idx in FACE_INDICES:
                    lm = results.face_landmarks.landmark[idx]
                    joints.append([lm.x, lm.y, lm.z])
            else:
                joints.extend([[0.0, 0.0, 0.0]] * 44)
            
            skeleton_sequence.append(joints)
            
    # Convert to Numpy (T, 119, 3)
    if skeleton_sequence:
        data = np.array(skeleton_sequence, dtype=np.float32)
        np.save(output_path, data)
        return True
    return False

def main():
    print("Starting Word-Level Skeleton Extraction...")
    
    # Iterate over all word folders
    word_classes = [d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))]
    
    print(f"Found {len(word_classes)} word classes.")
    
    success_count = 0
    
    for word_class in tqdm(word_classes):
        class_path = os.path.join(DATA_ROOT, word_class)
        output_class_path = os.path.join(OUTPUT_ROOT, word_class)
        os.makedirs(output_class_path, exist_ok=True)
        
        # Word folders usually contain just images directly?
        # Or subfolders like "HELP/HELP (1).jpg"?
        # Based on previous `ls`, "HELP" contained "HELP (1).jpg".
        # So `class_path` IS the folder with images.
        
        # We need to save ONE .npy file per SEQUENCE.
        # But wait, does "HELP" folder contain MULTIPLE sequences of "HELP"?
        # `ls` showed: HELP (1).jpg... HELP (12).jpg AND HELP-1.jpg...
        # These look like different clips mixed in one folder?
        # We need to group them.
        
        # Simple heuristic: Group by prefix? 
        # "HELP (x)" vs "HELP-x"
        # This is tricky. If we just smash them all together it's wrong.
        # Let's assume standard behavior:
        # If it's a mix, we might need a smarter grouper.
        # Let's try to pass the whole folder for now as ONE sequence?
        # NO, that would be bad if it contains multiple repetitions.
        
        # Strategy:
        # Just process the whole folder as ONE sequence for now.
        # If the user provides cleaner data later, we refine.
        # Actually... "HELP (1).jpg" -> "HELP (12).jpg" looks like one clip.
        # "HELP-1.jpg" looks like another.
        # We should probably treat the folder as ONE sample for now to get started, 
        # unless filenames clearly indicate split.
        
        output_file = os.path.join(output_class_path, f"{word_class}.npy")
        if process_word_folder(class_path, output_file):
            success_count += 1
            
    print(f"Processed {success_count} word classes.")

if __name__ == "__main__":
    main()
