import streamlit as st
import os
import json
import cv2
from PIL import Image
import json
import subprocess


def capture_frames_at_times(video_path, time_entries):
    frames = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error("Error: Could not open video.")
        return frames

    for entry in time_entries:
        lower_bound, upper_bound = entry["absolute_bounds"]
        cap.set(cv2.CAP_PROP_POS_MSEC, lower_bound * 1000)  # Set time in milliseconds
        ret, frame = cap.read()
        if ret:
            frames.append((frame, entry["heading"], lower_bound, upper_bound))
        else:
            st.error(f"Error: Could not read frame at {lower_bound} seconds.")

    cap.release()
    return frames


# Define the path to the video folder
MEDIA_FOLDER = "./custom_video_pipeline/data/video"

# List all video files in the specified directory
video_files = [
    f for f in os.listdir(MEDIA_FOLDER) if f.endswith((".mp4", ".mov", ".avi", ".mkv"))
]

# Set the number of videos to display at a time
videos_per_page = 3

# Set the layout of the Streamlit app
st.set_page_config(layout="wide")

# Initialize session state variables if not already set
if "selected_video" not in st.session_state:
    st.session_state.selected_video = None
if "submitted" not in st.session_state:
    st.session_state.submitted = None

# Create a container for the video panel
video_panel = st.container()

with video_panel:
    st.write("### Chọn Video")

    # Calculate the number of pages
    total_videos = len(video_files)
    total_pages = (total_videos - 1) // videos_per_page + 1

    # Create a slider to navigate through pages
    page = st.select_slider(
        "Select Page",
        options=list(range(1, total_pages + 1)),
        format_func=lambda x: f"Page {x}",
        label_visibility="collapsed",
    )

    # Calculate the range of videos to display
    start_index = (page - 1) * videos_per_page
    end_index = min(start_index + videos_per_page, total_videos)

    # Create columns for the current page videos
    cols = st.columns(videos_per_page)  # Extra columns for buttons

    # Display the videos and buttons for the current page
    for index, (col, video_file) in enumerate(
        zip(cols, video_files[start_index:end_index])
    ):
        with col:
            st.video(os.path.join(MEDIA_FOLDER, video_file))
            if st.button(
                f"Select Video {index + start_index + 1}", key=f"select_{index}"
            ):
                st.session_state.selected_video = video_file

# Display the selected video message if one is selected
if st.session_state.selected_video:
    st.write(f"Đã chọn Video: {st.session_state.selected_video}")

# Create a container for the input box and buttons
st.write("### Nhập câu truy vấn")

# Create a form for the input box and submit button
with st.form(key="input_form"):
    col1, col2, col3 = st.columns([10, 1, 1])  # Adjust column widths as needed

    # Input box
    with col1:
        user_input = st.text_input(
            "Nhập câu truy vấn",
            placeholder="Nhập câu truy vấn.",
            label_visibility="collapsed",
        )

    # Submit button
    with col2:
        submit_button = st.form_submit_button("Submit")
    with col3:
        micro_button = st.form_submit_button("🎙")

    if micro_button:
        print("micro button clicked")
        # st.write("Micro button clicked")

    # Handle form submission
    if submit_button:
        if not user_input:
            st.warning("Hãy nhập câu truy vấn.")
        else:
            if st.session_state.selected_video is None:
                st.warning("Hãy chọn một video để phân tích")
            else:
                st.session_state.submitted = True

if st.session_state.submitted:
    st.write("### Kết quả")
    video_file_path = os.path.join(MEDIA_FOLDER, st.session_state.selected_video)
    result_json = predict(st.session_state.selected_video, user_input)
    st.write("Kết quả dự đoán:")
    st.code(result_json)
    video_bound = next(iter(next(iter(result_json.values())).values()))["bounds"]
    times_input = next(iter(next(iter(result_json.values())).values()))["steps"]
    if times_input and video_bound:
        try:
            frames = capture_frames_at_times(video_file_path, times_input)
            if frames:
                st.write("Thông tin trích xuất:")
                images = [
                    Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    for frame, _, _, _ in frames
                ]
                captions = [
                    f"{heading} từ giây {lower_bound} đến giây {upper_bound}"
                    for _, heading, lower_bound, upper_bound in frames
                ]
                st.image(images, caption=captions, width=300)
                st.write("Video trích xuất:")
                st.video(
                    video_file_path, start_time=video_bound[0], end_time=video_bound[1]
                )
        except json.JSONDecodeError:
            st.error("Lỗi giải mã JSON")
        except ValueError:
            st.error("Lỗi giá trị không hợp lệ")
    else:
        st.write("Không có dữ liệu trả về.")
