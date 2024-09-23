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


@router.post("/video_retrieval")
async def promptcheck(request: promptCheckRequest):
    data = {
        "data": {
            "vid1.mp4": 0.85,
            "vid2.mp4": 0.78,
            "vid3.mp4": 0.92,
            "vid4.mp4": 0.60,
            "vid5.mp4": 0.77
        }
    }
    file_path = "./custom_video_pipeline2/all_data_test.json"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f)

    return data


class PredictRequest(BaseModel):
    video_file_name: str
    v_duration: float
    prompt: str


async def long_running_task(predict_request: PredictRequest, websocket: WebSocket):
    """
    Simulates a long-running machine learning task.
    Sends periodic progress updates to avoid WebSocket timeout.
    """
    try:
        # Simulate processing and logging
        log_messages = [
            "Processing request...",
            f"Received video file: {predict_request.video_file_name}",
            f"Duration: {predict_request.v_duration}",
            f"Prompt: {predict_request.prompt}"
        ]

        for message in log_messages:
            await websocket.send_text(json.dumps({"log": message}))
            await asyncio.sleep(1)  # Simulate some processing delay

        # Long-running task simulation with periodic progress updates
        for i in range(0, 100, 10):
            await asyncio.sleep(1)  # Simulate a chunk of long processing (5s per step)
            await websocket.send_text(json.dumps({"progress": f"{i}% complete"}))

        # Simulate final result after completion
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

        # Send final result
        await websocket.send_text(json.dumps({"data": result}))

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        await websocket.send_text(json.dumps({"log": f"Error: {str(e)}"}))


@router.websocket("/ws/moment_retrieval")
async def websocket_predict(websocket: WebSocket):
    await websocket.accept()
    try:
        # Receive the message from the client
        data = await websocket.receive_text()
        predict_request = PredictRequest.parse_raw(data)

        # Start the long-running task and keep the connection alive
        task_future = asyncio.create_task(long_running_task(predict_request, websocket))

        # Keeping the connection alive by sending a heartbeat every 30 seconds
        while not task_future.done():
            await websocket.send_text(json.dumps({"heartbeat": "alive"}))
            await asyncio.sleep(30)

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