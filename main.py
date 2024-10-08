
import tempfile
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
import streamlit as st
import cv2
import mediapipe as mp
from transformers import pipeline
import speech_recognition as sr
from moviepy.editor import VideoFileClip

# Initialize models
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
revision = "714eb0f"
classifier = pipeline('sentiment-analysis', model=model_name, revision=revision)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
recognizer = sr.Recognizer()

def speech_to_text(audio_path):
    """Convert speech to text"""
    try:
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio, language='en-US')
            return text
    except Exception as e:
        st.error(f"Speech-to-text failed: {str(e)}")
        return None

def analyze_sentiment(text):
    """Perform sentiment analysis"""
    return classifier(text)[0]

def analyze_eye_gaze(video_file):
    """Detect eye gaze and cheating"""
    cap = cv2.VideoCapture(video_file)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb_frame)
        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                st.write("Eye gaze detected!")
                st.image(frame, channels="BGR")
    cap.release()

def main():
    st.title("Interview Video Analysis Platform")
    st.header("Upload your interview video")
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        st.video(uploaded_file)
        st.write("Analyzing video...")

        # Save uploaded file to temporary location
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.write("Video saved to temporary location.")

        # Sentiment analysis
        try:
            st.write("Starting sentiment analysis...")

            # Extract audio
            audio_path = "temp_audio.wav"
            with VideoFileClip(temp_file_path) as video:
                video.audio.write_audiofile(audio_path)
            st.write("Audio extracted.")

            # Verify audio file existence
            st.write(f"Audio file exists: {os.path.exists(audio_path)}")

            # Speech-to-text
            st.write("Starting speech-to-text...")
            st.write(f"Audio path: {audio_path}")
            text = speech_to_text(audio_path)
            if text is None:
                st.error("No text extracted. Sentiment analysis skipped.")
            else:
                st.write(f"Speech-to-text result: {text}")
                sentiment_result = analyze_sentiment(text)
                st.write(f"**Sentiment:** {sentiment_result['label']}, **Confidence:** {sentiment_result['score']:.2f}")
        except Exception as e:
            st.error(f"Sentiment analysis failed: {str(e)}")

        # Eye gaze and anti-cheating
        try:
            st.write("Starting eye gaze analysis...")
            analyze_eye_gaze(temp_file_path)
            st.write("Eye gaze analysis completed.")
        except Exception as e:
            st.error(f"Eye gaze analysis failed: {str(e)}")

        # Delete temporary file
        os.remove(temp_file_path)
        st.write("Temporary file deleted.")

if __name__ == "__main__":
    main()














