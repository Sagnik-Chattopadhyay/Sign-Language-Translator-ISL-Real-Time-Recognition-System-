import base64
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from inference_utils import SignLanguageModel, extract_keypoints
import mediapipe as mp
import collections
import os
import gc
import torch

# Globally disable PyTorch gradients to save memory
torch.set_grad_enabled(False)

app = FastAPI(title="Sign Language Translator API")

# Mount static files for the frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

# Ensure checkpoint dir exists
os.makedirs("checkpoints", exist_ok=True)

# WARNING: TO DEPLOY, replace 'FILE_ID' below with your uploaded .pth Google Drive link ID.
MODEL_FILE_ID = "1ym8R4i81A6x-8WV4f1YCSvCM0lYdVcYd"
MODEL_PATH = "checkpoints/stgcn_model.pth"

if not os.path.exists(MODEL_PATH) and MODEL_FILE_ID != "YOUR_GOOGLE_DRIVE_FILE_ID_HERE":
    import gdown
    print("Downloading model weights for cloud deployment...")
    gdown.download(id=MODEL_FILE_ID, output=MODEL_PATH, quiet=False)

# Load model globally
model = SignLanguageModel(data_root="Videos_tensors", checkpoint_dir="checkpoints")

# If the model had to be downloaded *after* the initial init (or we want to ensure it caught it)
# we can force it to load the state dict now if it hasn't already.
if os.path.exists(MODEL_PATH):
    model.model.load_state_dict(torch.load(MODEL_PATH, map_location=model.device))
    # Try quantizing again just in case it was newly downloaded
    try:
        model.model = torch.quantization.quantize_dynamic(
            model.model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
        )
    except: pass
    model.model.eval()

# Force clear RAM of any lingering download/init artifacts
gc.collect()

# MediaPipe
mp_holistic = mp.solutions.holistic

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    await websocket.accept()
    
    MAX_FRAMES = 100
    sequence = collections.deque(maxlen=MAX_FRAMES)
    frame_counter = 0
    inference_interval = 5
    current_prediction = ""

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        try:
            while True:
                # Receive frame as base64 string
                data = await websocket.receive_text()
                
                # Decode image
                header, encoded = data.split(",", 1)
                img_data = base64.b64decode(encoded)
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is None:
                    continue
                
                # Process with MediaPipe
                image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)
                
                # Extract keypoints
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                
                frame_counter += 1
                
                # Run inference periodically
                if len(sequence) >= 20 and frame_counter % inference_interval == 0:
                    current_prediction = model.predict_sequence(list(sequence)[-MAX_FRAMES:])
                    
                    # Prevent memory creeping up over time during active streams
                    if frame_counter % (inference_interval * 10) == 0:
                        gc.collect()
                
                # Send result back
                await websocket.send_json({
                    "prediction": current_prediction
                })
                
        except WebSocketDisconnect:
            print("Client disconnected")
        except Exception as e:
            print(f"Error in websocket loop: {e}")
            try:
                await websocket.close()
            except:
                pass
