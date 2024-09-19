# API libraries
from fastapi import FastAPI, WebSocket, APIRouter, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import math

# Utility libraries
import json
import os
from tqdm import tqdm
import sys
import asyncio
import subprocess

sys.path.append("./EVA_clip")
from eva_clip import build_eva_model_and_transforms

# Model libraries
import torch

# Prompt text handler
import clip

# Initialize FastAPI app
router = APIRouter(prefix="/api/v1")


class videoRetrievalReq(BaseModel):
    prompt: str
    top_k: int


class momentRetrievalReq(BaseModel):
    prompt: str
    video_file_name: str
    v_duration: float


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


@router.post("/video_retrieval")
async def video_retrieval(request: videoRetrievalReq):
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

    results = {
        video: score for video, score in zip(top_videos, top_scores) if score >= 0.2
    }
    return {
        "message": "Video retrieval successful",
        "data": results,
    }


@router.post("/moment_retrieval")
async def moment_retrieval(request: momentRetrievalReq):
    data = {
        f"{request.prompt}": {
            f"{request.video_file_name}": {
                "relevant": True,
                "clip": True,
                "v_duration": math.ceil(request.v_duration),
                "bounds": [0, 1],
                "steps": [],
            }
        }
    }
    with open("./custom_pipeline/splits/all_data_test.json", "w") as f:
        json.dump(data, f)

    command = [
        "python",
        "run.py",
        "--data_dir",
        "./custom_pipeline/splits/",
        "--video_feature_dir",
        "./custom_pipeline/eva_clip_features",
        "--asr_dir",
        "./custom_pipeline/ASR",
        "--asr_feature_dir",
        "./custom_pipeline/ASR_feats_all-MiniLM-L6-v2",
        "--eval_batch_size",
        "1",
        "--task_moment_retrieval",
        "--task_moment_segmentation",
        "--task_step_captioning",
        "--ckpt_dir",
        "./checkpoints/hirest_joint_model/",
        "--end_to_end",
    ]

    # Run the command and capture output
    process = subprocess.run(command)

    if process.returncode == 0:
        with open("./checkpoints/hirest_joint_model/final_results.json", "r") as file:
            data = json.load(file)
        return {
            "message": "Moment retrieval successful",
            "data": data,
        }
    else:
        return {"message": "Error: Process failed"}


async def long_running_task(websocket: WebSocket):
    print("Starting long running task")
    command = [
        "python",
        "run.py",
        "--data_dir",
        "./custom_pipeline/splits/",
        "--video_feature_dir",
        "./custom_pipeline/eva_clip_features",
        "--asr_dir",
        "./custom_pipeline/ASR",
        "--asr_feature_dir",
        "./custom_pipeline/ASR_feats_all-MiniLM-L6-v2",
        "--eval_batch_size",
        "1",
        "--task_moment_retrieval",
        "--task_moment_segmentation",
        "--task_step_captioning",
        "--ckpt_dir",
        "./checkpoints/hirest_joint_model/",
        "--end_to_end",
    ]

    # Run the command and capture output
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    # Print stdout and stderr in real-time
    with process.stdout, process.stderr:
        for stdout_line in process.stdout:
            await websocket.send_text(json.dumps({"log": stdout_line}))
        for stderr_line in process.stderr:
            await websocket.send_text(json.dumps({"log": stderr_line}))

    # Wait for the process to finish and capture return code
    return_code = process.wait()
    if return_code == 0:
        with open("./checkpoints/hirest_joint_model/final_results.json", "r") as file:
            data = json.load(file)

        # Send result
        await websocket.send_text(json.dumps({"data": data}))
    else:
        await websocket.send_text(json.dumps({"log": "Error: Process failed"}))


@router.websocket("/ws/moment_retrieval")
async def websocket_predict(websocket: WebSocket):
    await websocket.accept()
    try:
        # Receive the message from the client
        data = await websocket.receive_text()
        request = momentRetrievalReq.parse_raw(data)
        print(request)
        data = {
            f"{request.prompt}": {
                f"{request.video_file_name}": {
                    "relevant": True,
                    "clip": True,
                    "v_duration": math.ceil(request.v_duration),
                    "bounds": [0, 1],
                    "steps": [],
                }
            }
        }
        with open("./custom_pipeline/splits/all_data_test.json", "w") as f:
            json.dump(data, f)

        await websocket.send_text(json.dumps({"log": "Moment retrieval started"}))
        task_future = asyncio.create_task(long_running_task(websocket))
        print(task_future.done())

        while not task_future.done():
            print(task_future.done())
            await websocket.send_text(json.dumps({"heartbeat": "alive"}))
            await asyncio.sleep(3)

        # Wait for the task to complete
        await task_future

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
