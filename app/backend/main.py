from fastapi import FastAPI, WebSocket, APIRouter, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os
import asyncio
import uvicorn
import logging
import time

# Initialize FastAPI app
router = APIRouter(prefix="/api/v1")


class promptCheckRequest(BaseModel):
    prompt: str


@router.post("/promptcheck")
async def promptcheck(request: promptCheckRequest):
    # data = {
    #     f"{request.prompt}": ["custom_video_pipeline/data/video/vid1.mp4", "custom_video_pipeline/data/video/vid2.mp4", "custom_video_pipeline/data/video/vid3.mp4", "custom_video_pipeline/data/video/vid4.mp4", "custom_video_pipeline/data/video/vid5.mp4"]
    # }
    data = {
        f"data": ["vid1.mp4", "vid2.mp4", "vid3.mp4", "vid4.mp4", "vid5.mp4"]
    }
    file_path = "./custom_video_pipeline2/all_data_test.json"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f)

    # command = [
    #     "python",
    #     "run.py",
    #     "--data_dir",
    #     "./custom_video_pipeline/data/splits/",
    #     "--video_feature_dir",
    #     "./custom_video_pipeline/data/eva_clip_features",
    #     "--asr_dir",
    #     "./custom_video_pipeline/data/ASR",
    #     "--asr_feature_dir",
    #     "./custom_video_pipeline/data/ASR_feats_all-MiniLM-L6-v2",
    #     "--eval_batch_size",
    #     "1",
    #     "--task_moment_retrieval",
    #     "--task_moment_segmentation",
    #     "--task_step_captioning",
    #     "--ckpt_dir",
    #     "./checkpoints/hirest_joint_model/",
    #     "--end_to_end",
    # ]

    # # Run the command and capture output
    # result = subprocess.run(command)

    # with open(
    #     "./checkpoints/hirest_joint_model/final_end_to_end_results.json", "r"
    # ) as file:
    #     data = json.load(file)

    return data

class PredictRequest(BaseModel):
    video_file_name: str
    v_duration: float
    prompt: str

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
            f"Prompt: {predict_request.prompt}"
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
