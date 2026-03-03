import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import os
import sys



# =========================
# PATH CONFIGURATION
# =========================

DATASET_ROOT = r"ISL_CSLRT_Corpus/ISL_CSLRT_Corpus/Videos_Sentence_Level"
OUTPUT_ROOT = r"Videos_tensors"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# =========================
# MEDIAPIPE INITIALIZATION
# =========================

mp_holistic = mp.solutions.holistic

# Reduced face landmarks (mouth + eyes)
FACE_INDICES = (
    list(range(61, 81)) +      # mouth
    list(range(291, 311)) +    # mouth
    [33, 133, 362, 263]        # eyes
)

# =========================
# CORE FUNCTION
# =========================

def process_video_to_skeleton(video_path, img_size=(224, 224)):
    """
    Converts a video into a skeleton tensor (T, V, 3)
    without saving frames or video tensors.
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    skeleton_sequence = []

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True
    ) as holistic:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # -------------------------
            # Frame preprocessing
            # -------------------------
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # frame = cv2.resize(frame, img_size, interpolation=cv2.INTER_LINEAR) # Removed for accuracy

            # -------------------------
            # MediaPipe inference
            # -------------------------
            results = holistic.process(frame)

            joints = []

            # ---- Pose (33) ----
            if results.pose_landmarks:
                for lm in results.pose_landmarks.landmark:
                    joints.append([lm.x, lm.y, lm.z])
            else:
                joints.extend([[0.0, 0.0, 0.0]] * 33)

            # ---- Left Hand (21) ----
            if results.left_hand_landmarks:
                for lm in results.left_hand_landmarks.landmark:
                    joints.append([lm.x, lm.y, lm.z])
            else:
                joints.extend([[0.0, 0.0, 0.0]] * 21)

            # ---- Right Hand (21) ----
            if results.right_hand_landmarks:
                for lm in results.right_hand_landmarks.landmark:
                    joints.append([lm.x, lm.y, lm.z])
            else:
                joints.extend([[0.0, 0.0, 0.0]] * 21)

            # ---- Reduced Face ----
            if results.face_landmarks:
                for idx in FACE_INDICES:
                    lm = results.face_landmarks.landmark[idx]
                    joints.append([lm.x, lm.y, lm.z])
            else:
                joints.extend([[0.0, 0.0, 0.0]] * len(FACE_INDICES))

            skeleton_sequence.append(joints)

    cap.release()

    # Convert to numpy tensor
    skeleton_tensor = np.array(skeleton_sequence, dtype=np.float32)
    return skeleton_tensor

if __name__ == "__main__":
    # =========================
    # DATASET-LEVEL PROCESSING
    # =========================

    for sentence_name in tqdm(os.listdir(DATASET_ROOT), desc="Processing sentences"):
        sentence_path = os.path.join(DATASET_ROOT, sentence_name)

        if not os.path.isdir(sentence_path):
            continue

        output_sentence_folder = os.path.join(OUTPUT_ROOT, sentence_name)
        os.makedirs(output_sentence_folder, exist_ok=True)

        for video_file in os.listdir(sentence_path):
            if not video_file.lower().endswith((".mp4", ".avi", ".mov")):
                continue

            video_path = os.path.join(sentence_path, video_file)
            video_name = os.path.splitext(video_file)[0]

            try:
                skeleton = process_video_to_skeleton(video_path)

                save_path = os.path.join(output_sentence_folder, f"{video_name}.npy")
                np.save(save_path, skeleton)

            except Exception as e:
                print(f"Error processing {video_path}: {e}")

    print(" Skeleton extraction completed.")
