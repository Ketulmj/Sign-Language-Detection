import streamlit as st
import ollama
import cv2
import tempfile
import os
import numpy as np
from PIL import Image
import io

def extract_four_frames(video_data: bytes):
    """Extract exactly 4 frames at equal intervals from the video"""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
        temp_video.write(video_data)
        temp_video_path = temp_video.name

    frames = []
    try:
        cap = cv2.VideoCapture(temp_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames < 4:
            st.error("Video has too few frames to extract 4 evenly spaced frames.")
            return []

        frame_positions = [int(total_frames * i / 4) for i in range(4)]  # Get 4 evenly spaced positions

        for pos in frame_positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)  # Move to the frame position
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                img_byte_arr = io.BytesIO()
                pil_image.save(img_byte_arr, format='JPEG')
                frames.append(img_byte_arr.getvalue())

        cap.release()
    finally:
        os.unlink(temp_video_path)
    
    return frames

def analyze_frame(frame, frame_number):
    """Analyze a single frame for human gesture recognition"""
    frame_prompt = (
        f"""You are an expert in human gesture recognition. Analyze the given frame ({frame_number}) 
        and determine the action being performed.
        Consider gestures such as:
        - **Pointing**: Directing towards an object or place.
        - **Waving**: Greeting or calling attention.
        - **Thumbs up/down**: Approval or disapproval.
        - **Hand to mouth**: Drinking, eating, or speaking.
        - **Open palm facing up**: Requesting or asking.
        - **Nodding/shaking head**: Agreement or disagreement.
        - **Leaning forward/backward**: Interest or withdrawal.

        **Your task:** Identify the most likely meaning of this gesture in one concise phrase.
        """
    )

    response = ollama.chat(
        model='gemma3',
        messages=[{'role': 'user', 'content': frame_prompt, 'images': [frame]}],
    )
    return response['message']['content']

def summarize_gestures(frame_analyses):
    """Summarizes all analyzed frames into one meaningful message"""
    summary_prompt = (
        "Based on the gestures detected in the analyzed frames, summarize what the person is trying to communicate.\n"
        f"Frame analyses:\n{frame_analyses}\n"
        "Provide a single, natural-language sentence summarizing the meaning of these gestures in the video."
    )

    response = ollama.chat(
        model='gemma3',
        messages=[{'role': 'user', 'content': summary_prompt}],
    )
    return response['message']['content']

# Streamlit UI
st.title("Human Gesture Recognition from Video")
st.write("Upload a video, and I'll analyze it to determine the meaning of gestures!")

uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

if uploaded_video:
    video_data = uploaded_video.read()
    frames = extract_four_frames(video_data)

    if frames:
        st.write("Analyzing gestures... Please wait.")
        frame_analyses = []

        for i, frame in enumerate(frames):
            analysis = analyze_frame(frame, i+1)
            frame_analyses.append(f"Frame {i+1}: {analysis}")
        
        final_summary = summarize_gestures("\n".join(frame_analyses))

        st.subheader("Extracted Gesture Meaning:")
        st.write(final_summary)
    else:
        st.error("No frames extracted from the video. Please try again with a different file.")