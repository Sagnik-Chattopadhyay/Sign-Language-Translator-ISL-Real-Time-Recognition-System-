# ISL Real-Time Recognition System

A real-time Indian Sign Language (ISL) recognition system powered by a Spatial-Temporal Graph Convolutional Network (ST-GCN). This project uses MediaPipe Holistic for skeleton-based gesture representation (extracting 119 body keypoints per frame) and features a modern web UI for real-time translation using FastAPI and WebSockets.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)

## Features
- **ST-GCN Backbone**: Designed and trained on the ISL-CSLRT Corpus, effectively handling spatial-temporal skeleton data.
- **MediaPipe Keypoint Extraction**: Extracts pose, hands, and facial landmarks for robust 3D representation.
- **Live Inference**: Predicts sequences with CTC greedy decoding to string together seamless sentences.
- **Modern Web Interface**: A sleek UI that connects to the webcam to stream frames to the server via WebSockets for real-time closed captioning.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/sign-language-translator.git
   cd sign-language-translator
   ```

2. **Create a virtual environment (Optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements_deploy.txt
   ```

## Model Checkpoints
Due to size limitations, the `checkpoints/` folder containing the pre-trained `.pth` models is ignored in this repository. 
> *Ensure your `.pth` model file is placed in the `checkpoints/` directory before running the application.*

## Running the Web App
Start the FastAPI server:
```bash
uvicorn app:app --reload
```
Next, open your browser and navigate to `http://localhost:8000`. Click "Start Camera" and start signing!

## Project Structure
- `app.py`: FastAPI server handling WebSocket video streaming and inference routing.
- `static/index.html`: Web UI frontend.
- `inference_utils.py`: Reusable utilities dictating how MediaPipe extracts frames and processes them through the ST-GCN model.
- `stgcn.py`: The core PyTorch skeleton model definition.
- `train.py` / `dataset.py`: Training routines for the ISL-CSLRT Corpus (Local use only, expects `Videos_tensors` dataloader).

## License
MIT
