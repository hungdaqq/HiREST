import os
import json
from moviepy.editor import VideoFileClip

# Folder containing video files
video_folder = "./custom_pipeline/videos"

# Path to your JSON file
json_file_path = "./custom_pipeline/data.json"

# Load the existing JSON data
with open(json_file_path, "r") as file:
    data = json.load(file)


# Function to get video duration
def get_video_duration(video_path):
    clip = VideoFileClip(video_path)
    duration = clip.duration
    clip.close()
    return duration


# Update JSON with video durations
for category, videos in data.items():
    print(category)
    for video_file in videos:
        video_path = os.path.join(video_folder, video_file)
        if os.path.isfile(video_path):
            duration = get_video_duration(video_path)
            videos[video_file]["v_duration"] = duration

# Save updated JSON data
with open(json_file_path, "w") as file:
    json.dump(data, file, indent=2)

print("JSON file updated with video durations.")
