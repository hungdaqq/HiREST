import streamlit as st
import os
import json
import cv2
from PIL import Image
import json
import requests
from moviepy.editor import VideoFileClip
import websockets
import asyncio


# Define the path to the video folder
VIDEO_DIR = "custom_pipeline/videos"
VIDEOS_PER_ROW = 5  # Adjust based on how many videos you want per row
VIDEOS_PER_PAGE = 10  # Adjust how many videos to show per page
HTTP_BASE_URL = "http://localhost:8000/api/v1"
WS_BASE_URL = "ws://localhost:8000/api/v1"


def get_video_duration(video_path):
    try:
        # Load the video
        clip = VideoFileClip(video_path)

        # Get the duration in seconds
        duration = clip.duration
        return duration
    except Exception as e:
        print(f"Error reading video duration: {e}")
        return None


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


async def send_predict_request(websocket, video_file_name, v_duration, prompt):
    # Define the request payload
    payload = {
        "video_file_name": video_file_name,
        "v_duration": v_duration,
        "prompt": prompt,
    }

    # Send the request payload as a JSON string
    await websocket.send(json.dumps(payload))

    try:
        while True:
            # Wait for the response
            response = await websocket.recv()
            data = json.loads(response)
            if "heartbeat" in data:
                print("Received heartbeat")
                continue
            # Check if the message contains 'log' key
            if "log" in data:
                # st.session_state.logs.append(data["log"])
                # # Update the display or log placeholder as needed
                # log_placeholder.markdown(
                #     f"""
                #     <div style="border: 2px solid #ddd; border-radius: 5px; padding: 10px; background-color: gray; height: 300px; overflow-y: auto;">
                #         <div>{'<br>'.join(st.session_state.logs)}</div>
                #     </div>
                #     """,
                #     unsafe_allow_html=True,
                # )
                print(data["log"])
                continue

            # Check if the message contains 'result' key
            elif "data" in data:
                st.session_state.result = data["data"]
                break  # Exit the loop when result is received
            else:
                print("Received unknown message:", data)

    except websockets.ConnectionClosedError as e:
        print(f"Connection closed with error: {e}")
    except websockets.ConnectionClosedOK:
        print("Connection closed cleanly")


def video_retrieval_request(prompt, top_k=5):
    url = f"{HTTP_BASE_URL}/video_retrieval"

    # Define the request payload
    payload = {"prompt": prompt, "top_k": top_k}
    # Send the POST request
    response = requests.post(url, json=payload)

    # Check for success
    if response.status_code == 200:
        print("Request successful!")

        # Extract the dictionary of video file paths and their confidence values from the response
        response_data = response.json()

        video_file_dict = response_data["data"]

        # Filter out non-existent files
        existing_files = {
            file_path: confidence
            for file_path, confidence in video_file_dict.items()
            if os.path.exists(os.path.join(VIDEO_DIR, file_path))
        }

        return existing_files

    else:
        print(f"Failed with status code: {response.status_code}")
        print("Response:", response.text)
        return None


def video_moment_retrieval(video_file_name, v_duration, prompt):
    url = f"{HTTP_BASE_URL}/moment_retrieval"

    payload = {
        "video_file_name": video_file_name,
        "v_duration": v_duration,
        "prompt": prompt,
    }
    # Send the POST request
    response = requests.post(url, json=payload)

    # Check for success
    if response.status_code == 200:
        print("Request successful!")

        # Extract the dictionary of video file paths and their confidence values from the response
        response_data = response.json()

        response = response_data["data"]

        return response

    else:
        print(f"Failed with status code: {response.status_code}")
        print("Response:", response.text)
        return None


# async def websocket_client(video_file_name, v_duration, prompt):
#     url = f"{WS_BASE_URL}/ws/moment_retrieval"
#     async with websockets.connect(url) as websocket:
#         await send_predict_request(websocket, video_file_name, v_duration, prompt)


async def websocket_client(video_file_name, v_duration, prompt):
    url = f"{WS_BASE_URL}/ws/moment_retrieval"
    with st.spinner("Đang phân tích dữ liệu..."):
        # Open the websocket connection
        async with websockets.connect(url, ping_interval=300, ping_timeout=None) as websocket:
            await send_predict_request(websocket, video_file_name, v_duration, prompt)


# Set the layout of the Streamlit app
st.set_page_config(layout="wide")


# Function to get all video files in the folder
def get_video_files():
    return [
        f
        for f in os.listdir(VIDEO_DIR)
        if os.path.isfile(os.path.join(VIDEO_DIR, f))
        and f.lower().endswith((".mp4", ".avi", ".mov"))
    ]


# Initialize session state variables if not already set
if "selected_video" not in st.session_state:
    st.session_state.selected_video = None
if "submitted" not in st.session_state:
    st.session_state.submitted = None
if "video_files" not in st.session_state:
    st.session_state.video_files = None
if "logs" not in st.session_state:
    st.session_state.logs = []
if "result" not in st.session_state:
    st.session_state.result = None
if "show_videos" not in st.session_state:
    st.session_state.show_videos = False

# Button to toggle video display
if st.button("Hiện danh sách video"):
    st.session_state.show_videos = not st.session_state.show_videos

# Pagination logic
if "page" not in st.session_state:
    st.session_state.page = 0


def next_page():
    st.session_state.page += 1


def prev_page():
    st.session_state.page -= 1


if st.session_state.show_videos:
    st.write("### Danh sách video")
    st.session_state.logs = []
    video_files = get_video_files()
    num_videos = len(video_files)
    num_rows = (VIDEOS_PER_PAGE + VIDEOS_PER_ROW - 1) // VIDEOS_PER_ROW
    current_page = st.session_state.page
    start_index = current_page * VIDEOS_PER_PAGE
    end_index = min(start_index + VIDEOS_PER_PAGE, num_videos)

    if start_index >= num_videos:
        st.write("Không có video nào.")
    else:
        videos_to_display = video_files[start_index:end_index]

        for row in range(0, len(videos_to_display), VIDEOS_PER_ROW):
            cols = st.columns(VIDEOS_PER_ROW)
            for col, video_file in zip(
                cols, videos_to_display[row : row + VIDEOS_PER_ROW]
            ):
                with col:
                    st.video(os.path.join(VIDEO_DIR, video_file))

    # Pagination controls
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if current_page > 0:
            st.button("Trang trước", on_click=prev_page)
    with col2:
        st.write(
            f"Trang {current_page + 1} trên {((num_videos + VIDEOS_PER_PAGE - 1) // VIDEOS_PER_PAGE)}"
        )
    with col3:
        if end_index < num_videos:
            st.button("Trang tiếp theo", on_click=next_page)


# Create a container for the input box and buttons
st.write("### Nhập câu truy vấn")

# Create a form for the input box and submit button
with st.form(key="input_form"):
    col1, col2 = st.columns([11, 1])  # Adjust column widths as needed

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
    # with col3:
    #     micro_button = st.form_submit_button("🎙")

    # if micro_button:
    #     print("micro button clicked")
    #     # st.write("Micro button clicked")

    # Handle form submission
    if submit_button:
        if not user_input:
            st.warning("Hãy nhập câu truy vấn.")
        else:
            st.session_state.selected_video = None
            st.session_state.video_files = video_retrieval_request(user_input)
            if not st.session_state.video_files:
                st.error("Không tìm thấy video phù hợp .")
            else:
                st.session_state.submitted = True

# if st.session_state.submitted:
# # Set the number of videos to display at a time
#     videos_per_page = 3
#     # Create a container for the video panel
#     video_panel = st.container()

#     with video_panel:
#         st.write("### Chọn Video")

#         # Calculate the number of pages
#         total_videos = len(st.session_state.video_files)
#         total_pages = (total_videos - 1) // videos_per_page + 1

#         # Create a slider to navigate through pages
#         page = st.select_slider(
#             "Select Page",
#             options=list(range(1, total_pages + 1)),
#             format_func=lambda x: f"Page {x}",
#             label_visibility="collapsed",
#         )

#         # Calculate the range of videos to display
#         start_index = (page - 1) * videos_per_page
#         end_index = min(start_index + videos_per_page, total_videos)

#         # Create columns for the current page videos
#         cols = st.columns(videos_per_page)
#         cols2 = st.columns(videos_per_page)

#         # Display the videos and buttons for the current page
#         for index, (col, video_file) in enumerate(zip(cols, st.session_state.video_files[start_index:end_index])):
#             with col:
#                 st.video(os.path.join(VIDEO_DIR, video_file))

#         for index, (col2, video_file) in enumerate(zip(cols2, st.session_state.video_files[start_index:end_index])):
#             with col2:
#                 if st.button(f"Select Video {index + start_index + 1}", key=f"select_{index}"):
#                     st.session_state.selected_video = video_file


if st.session_state.submitted:
    st.session_state.logs = []
    # Set the number of videos per row
    videos_per_row = 4
    # Create a container for the video panel
    video_panel = st.container()

    with video_panel:
        st.write("### Chọn Video")
        video_files = st.session_state.video_files

        # Create a grid layout for the videos
        num_videos = len(video_files)
        num_rows = (num_videos + videos_per_row - 1) // videos_per_row

        for row in range(num_rows):
            cols = st.columns(videos_per_row)
            start_index = row * videos_per_row
            end_index = min(start_index + videos_per_row, num_videos)
            videos_to_display = list(video_files.items())[start_index:end_index]

            for col, (video_file, confidence) in zip(cols, videos_to_display):
                with col:
                    st.video(os.path.join(VIDEO_DIR, video_file))
            cols2 = st.columns(videos_per_row)
            for col, (video_file, confidence) in zip(cols2, videos_to_display):
                with col:
                    st.write(f"Confidence: {confidence*100:.2f}%")
                    if st.button(
                        f"Select Video {start_index + list(video_files.keys()).index(video_file) + 1}",
                        key=f"select_{video_file}",
                    ):
                        st.session_state.selected_video = video_file


st.write("### Kết quả")
# Placeholder for real-time logs
log_placeholder = st.empty()


# Display the selected video message if one is selected
if st.session_state.selected_video:
    st.write(f"Đã chọn Video: {st.session_state.selected_video}")
    st.session_state.logs = []
    video_file_path = os.path.join(VIDEO_DIR, st.session_state.selected_video)
    duration = get_video_duration(video_file_path)
    asyncio.run(websocket_client(st.session_state.selected_video, duration, user_input))

# # Display the selected video message if one is selected
# if st.session_state.selected_video:
#     st.write(f"Đã chọn Video: {st.session_state.selected_video}")
#     st.session_state.logs = []
#     video_file_path = os.path.join(VIDEO_DIR, st.session_state.selected_video)
#     duration = get_video_duration(video_file_path)
#     asyncio.run(websocket_client(st.session_state.selected_video, duration, user_input))


if st.session_state.result:
    result_json = st.session_state.result
    video_bound = next(iter(next(iter(result_json.values())).values()))["bounds"]
    times_input = next(iter(next(iter(result_json.values())).values()))["steps"]
    outer_key = next(iter(result_json))
    video_name = list(result_json[outer_key].keys())[0]
    st.write(
        f"Kết quả dự đoán: Video {video_name} có chứa thông tin bạn cần truy vấn từ giây thứ {video_bound[0]} đến giây thứ {video_bound[1]}."
    )
    st.write("Chi tiết kết quả dự đoán:")
    st.code(result_json)
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
