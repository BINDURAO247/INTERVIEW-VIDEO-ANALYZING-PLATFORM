import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import threading

# Initialize face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)

# Create Streamlit app
st.title("Webcam Analysis")

# Display webcam feed
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])

# Initialize webcam
cap = cv2.VideoCapture(0)

def analyze_eye_gaze(frame):
    # Eye gaze analysis
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)
    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            try:
                # Calculate eye centers
                left_eye_center = np.array([face_landmarks.landmark[474].x, face_landmarks.landmark[474].y])
                right_eye_center = np.array([face_landmarks.landmark[475].x, face_landmarks.landmark[475].y])

                # Calculate gaze direction vector
                gaze_direction = right_eye_center - left_eye_center

                # Calculate angle between gaze direction and camera's view direction
                angle = np.arctan2(gaze_direction[1], gaze_direction[0]) * 180 / np.pi

                # Determine if person is looking away from screen
                if abs(angle) > 30:
                    cv2.putText(frame, "Looking Away!", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    st.markdown("**Mannersism Detected:** Avoiding Eye Contact")
                else:
                    cv2.putText(frame, "Looking at Screen", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            except IndexError:
                cv2.putText(frame, "Insufficient Landmarks", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

def analyze_mannersisms(frame):
    # Analyze facial expressions
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        if roi.size == 0:
            continue

        # Detect smile
        smile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml').detectMultiScale(roi, 1.8, 20)
        if len(smile) > 0:
            st.markdown("**Mannersism Detected:** Smiling")
            st.markdown("**Sentiment:** Positive")
        else:
            st.markdown("**Mannersism Detected:** Neutral")
            st.markdown("**Sentiment:** Neutral")

def main():
    while run:
        ret, frame = cap.read()
        if not ret:
            break

        # Analyze frame
        analyze_eye_gaze(frame)
        analyze_mannersisms(frame)

        # Display analyzed frame
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", caption="Webcam Feed")

        # Exit on key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()
