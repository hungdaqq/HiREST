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
import subprocess


def long_running_task():
    """
    Simulates a long-running machine learning task.
    Sends periodic progress updates to avoid WebSocket timeout.
    """
    try:
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
                print(json.dumps({"log": stdout_line}))
            for stderr_line in process.stderr:
                print(json.dumps({"log": stderr_line}))

        # Wait for the process to finish and capture return code
        return_code = process.wait()
        if return_code == 0:
            with open(
                "./checkpoints/hirest_joint_model/final_results.json", "r"
            ) as file:
                data = json.load(file)

            # Send result
            print(json.dumps({"data": data}))
        else:
            print(json.dumps({"log": "Error: Process failed"}))
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(json.dumps({"log": f"Error: {str(e)}"}))


if __name__ == "__main__":
    # Run the FastAPI app
    long_running_task()
