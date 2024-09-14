import torch

video_embeds = torch.load(
    "custom_pipeline/eva_clip_features/1-SJGQ2HLp8.mp4.pt", map_location="cpu"
)
print(video_embeds)