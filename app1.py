

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import csv
import json

# Initialize face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.3)

# Create Streamlit app
st.title("Webcam Analysis")

# Display webcam feed
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])

# Initialize webcam
cap = cv2.VideoCapture(0)

# Check if webcam is opened
if not cap.isOpened():
    st.error("Webcam not detected. Ensure it's properly connected and configured.")
    exit()

# Initialize data storage
eye_gaze_data = []
mannersism_data = []
start_time = time.time()

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
                    st.markdown("*Mannersism Detected:* Avoiding Eye Contact")
                    eye_gaze_data.append(0)  # Store 0 for looking away
                else:
                    cv2.putText(frame, "Looking at Screen", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    eye_gaze_data.append(1)  # Store 1 for engaging eye contact
            except IndexError:
                cv2.putText(frame, "Insufficient Landmarks", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                eye_gaze_data.append(None)  # Store None for insufficient landmarks

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
            st.markdown("*Mannersism Detected:* Smiling")
            st.markdown("*Sentiment:* Positive")
            mannersism_data.append("Smiling")  # Store smiling mannersism
        else:
            st.markdown("*Mannersism Detected:* Neutral")
            st.markdown("*Sentiment:* Neutral")
            mannersism_data.append("Neutral")  # Store neutral mannersism

def main():
    global eye_gaze_data, mannersism_data, start_time
    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to read frame from webcam.")
            break

        # Analyze frame
        analyze_eye_gaze(frame)
        analyze_mannersisms(frame)

        # Display analyzed frame
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", caption="Webcam Feed")

        # Exit on key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release webcam
    cap.release()

    # Calculate averages and percentages
    eye_gaze_percentage = (eye_gaze_data.count(1) / len(eye_gaze_data)) * 100 if eye_gaze_data else 0
    mannersism_smiling_percentage = (mannersism_data.count("Smiling") / len(mannersism_data)) * 100 if mannersism_data else 0

    analysis_time = time.time() - start_time
    total_frames_analyzed = len(eye_gaze_data)
    average_eye_gaze_per_frame = sum(1 for x in eye_gaze_data if x == 1) / len(
        eye_gaze_data) * 100 if eye_gaze_data else 0

    # Display results
    st.header("Analysis Results")
    st.subheader("Eye Gaze Analysis")
    st.write("Eye Gaze Percentage: {:.2f}%".format(eye_gaze_percentage))
    st.write("Average Eye Gaze Per Frame: {:.2f}%".format(average_eye_gaze_per_frame))
    st.write("Total Frames Analyzed: {}".format(total_frames_analyzed))

    st.subheader("Mannersism Analysis")
    st.write("Smiling Percentage: {:.2f}%".format(mannersism_smiling_percentage))
    st.write("Analysis Time: {:.2f} seconds".format(analysis_time))

    # Save data to CSV
    with open('analysis_data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Eye Gaze Percentage", eye_gaze_percentage])
        writer.writerow(["Average Eye Gaze Per Frame", average_eye_gaze_per_frame])
        writer.writerow(["Total Frames Analyzed", total_frames_analyzed])
        writer.writerow(["Smiling Percentage", mannersism_smiling_percentage])
        writer.writerow(["Analysis Time", analysis_time])

    # Download CSV file
    with open('analysis_data.csv', 'rb') as csvfile:
        st.download_button(label="Download Analysis Report", data=csvfile.read(), file_name="analysis_data.csv",
                           mime="text/csv")


if __name__ == "__main__":
    main()

