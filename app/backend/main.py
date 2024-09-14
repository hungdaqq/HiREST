# API libraries
from fastapi import FastAPI, WebSocket, APIRouter, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import uvicorn

# Utility libraries
import json
import os
from tqdm import tqdm
import sys
import pathlib

sys.path.append("./EVA_clip")
from eva_clip import build_eva_model_and_transforms

# Model libraries
import torch

# Prompt text handler
import clip

# Initialize FastAPI app
router = APIRouter(prefix="/api/v1")


class promptCheckRequest(BaseModel):
    prompt: str
    top_k: int


class PredictRequest(BaseModel):
    video_file_name: str
    v_duration: float
    prompt: str


clip_model, clip_preprocess = build_eva_model_and_transforms(
    "EVA_CLIP_g_14", pretrained="./pretrained_weights/eva_clip_psz14.pt"
)

DEVICE = "cpu"
clip_model = clip_model.to(DEVICE)
clip_model.eval()

VIDEO_DIR = "./custom_pipeline/videos"
VIDEO_FEATURE_DIR = "./custom_pipeline/eva_clip_features"
all_video_ids = os.listdir(VIDEO_DIR)
all_video_embeds = []
for video_id in tqdm(all_video_ids):
    video_embeds = torch.load(
        os.path.join(VIDEO_FEATURE_DIR, f"{video_id}.pt"), map_location="cpu"
    )
    video_embeds /= video_embeds.norm(dim=-1, keepdim=True)
    all_video_embeds.append(video_embeds.mean(dim=0))  # Avgpool
all_video_embeds = torch.stack(all_video_embeds).to(DEVICE)


@router.post("/video_retrival")
async def video_retrival(request: promptCheckRequest):
    prompts = [request.prompt]

    with torch.no_grad():
        text_token = clip.tokenize(prompts).to(DEVICE)
        text_embed = clip_model.encode_text(text_token).float()
        text_embed /= text_embed.norm(dim=-1, keepdim=True)

    text_to_video_scores = torch.matmul(text_embed, all_video_embeds.T)

    # Sort videos based on the scores and take top-k videos
    top_scores, top_indices = torch.topk(text_to_video_scores, request.top_k, dim=1)

    top_videos = [all_video_ids[i] for i in top_indices[0]]
    top_scores = top_scores[0].tolist()

    # Zip the scores with the videos
    top_results = dict(zip(top_videos, top_scores))

    return {"data": top_results, "message": "Video retrieval successful"}


@router.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    await websocket.accept()
    try:
        # Receive the message from the client
        data = await websocket.receive_text()
        predict_request = PredictRequest.parse_raw(data)

        # Simulate processing and logging
        log_messages = [
            "Processing request...",
            f"Received video file: {predict_request.video_file_name}",
            f"Duration: {predict_request.v_duration}",
            f"Prompt: {predict_request.prompt}",
        ]

        for message in log_messages:
            await websocket.send_text(json.dumps({"log": message}))
            await asyncio.sleep(2)  # Simulate processing delay

        # Simulate result
        result = {
            f"{predict_request.prompt}": {
                f"{predict_request.video_file_name}": {
                    "relevant": True,
                    "clip": True,
                    "v_duration": predict_request.v_duration,
                    "bounds": [0, 1],
                    "steps": [],
                }
            }
        }

        # Send result
        await websocket.send_text(json.dumps({"result": result}))

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        await websocket.send_text(json.dumps({"log": f"Error: {str(e)}"}))
    finally:
        await websocket.close()


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can specify a list of origins here if needed
    allow_credentials=True,
    allow_methods=["*"],  # You can restrict allowed methods here
    allow_headers=["*"],  # You can restrict allowed headers here
)

app.include_router(router)


if __name__ == "__main__":
    # Run the FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=8000)
