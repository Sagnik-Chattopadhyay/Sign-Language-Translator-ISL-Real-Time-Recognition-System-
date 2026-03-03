import base64
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from inference_utils import SignLanguageModel, extract_keypoints
import mediapipe as mp
import collections

app = FastAPI(title="Sign Language Translator API")

# Mount static files for the frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model globally
model = SignLanguageModel(data_root="Videos_tensors", checkpoint_dir="checkpoints")

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
