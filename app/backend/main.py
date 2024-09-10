from fastapi import FastAPI, Query
from pydantic import BaseModel
import subprocess
import json

# Initialize FastAPI app
router = APIRouter(prefix="/api/v1")


class PredictRequest(BaseModel):
    video_file_name: str
    v_duration: float
    promt: str


@router.post("/predict")
async def predict(request: PredictRequest):
    data = {
        f"{request.promt}": {
            f"{request.video_file_name}": {
                "relevant": True,
                "clip": True,
                "v_duration": request.v_duration,
                "bounds": [0, 1],
                "steps": [],
            }
        }
    }
    with open("./custom_video_pipeline/data/splits/all_data_test.json", "w") as f:
        json.dump(data, f)

    command = [
        "python",
        "run.py",
        "--data_dir",
        "./custom_video_pipeline/data/splits/",
        "--video_feature_dir",
        "./custom_video_pipeline/data/eva_clip_features",
        "--asr_dir",
        "./custom_video_pipeline/data/ASR",
        "--asr_feature_dir",
        "./custom_video_pipeline/data/ASR_feats_all-MiniLM-L6-v2",
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
    result = subprocess.run(command)

    with open(
        "./checkpoints/hirest_joint_model/final_end_to_end_results.json", "r"
    ) as file:
        data = json.load(file)

    return data


app = FastAPI()
app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
