import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import csv

# Initialize face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, static_image_mode=False)

# Create Streamlit app
st.title("Live Webcam Analysis")

# Display webcam feed
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize session state for data persistence across runs
if 'eye_gaze_data' not in st.session_state:
    st.session_state.eye_gaze_data = []
if 'mannersism_data' not in st.session_state:
    st.session_state.mannersism_data = []
if 'start_time' not in st.session_state:
    st.session_state.start_time = time.time()


def debug_print(variable_name, value):
    st.write(f"DEBUG: {variable_name} = {value}")


def visualize_landmarks(frame, face_landmarks):
    # Visualize all facial landmarks on the frame
    for landmark in face_landmarks.landmark:
        x = int(landmark.x * frame.shape[1])
        y = int(landmark.y * frame.shape[0])
        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)


def analyze_eye_gaze(frame):
    # Eye gaze analysis
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            visualize_landmarks(frame, face_landmarks)  # Visualize landmarks
            try:
                left_eye_landmarks = [face_landmarks.landmark[i] for i in range(133, 144)]
                right_eye_landmarks = [face_landmarks.landmark[i] for i in range(362, 373)]

                if len(left_eye_landmarks) > 0 and len(right_eye_landmarks) > 0:
                    left_eye_center = np.mean([[lm.x, lm.y] for lm in left_eye_landmarks], axis=0)
                    right_eye_center = np.mean([[lm.x, lm.y] for lm in right_eye_landmarks], axis=0)

                    # Calculate gaze direction vector
                    gaze_direction = right_eye_center - left_eye_center

                    # Calculate angle between gaze direction and camera's view direction
                    angle = np.arctan2(gaze_direction[1], gaze_direction[0]) * 180 / np.pi

                    # Determine if person is looking away from the screen
                    if abs(angle) > 30:
                        cv2.putText(frame, "Looking Away!", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        st.session_state.eye_gaze_data.append(0)  # Store 0 for looking away
                    else:
                        cv2.putText(frame, "Looking at Screen", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        st.session_state.eye_gaze_data.append(1)  # Store 1 for engaging eye contact
                else:
                    raise IndexError("Insufficient landmarks for eye areas")
            except IndexError:
                cv2.putText(frame, "Insufficient Landmarks", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                st.session_state.eye_gaze_data.append(None)  # Store None for insufficient landmarks
    else:
        cv2.putText(frame, "No Face Detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        st.session_state.eye_gaze_data.append(None)  # Store None if no face is detected

    debug_print("eye_gaze_data", st.session_state.eye_gaze_data)  # Debugging


def analyze_mannersisms(frame):
    # Analyze facial expressions
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(
        gray, 1.1, 4)

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            roi = gray[y: y + h, x: x + w]
            if roi.size == 0:
                continue

            # Detect smile
            smile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml').detectMultiScale(roi, 1.8,
                                                                                                            20)

            if len(smile) > 0:
                cv2.putText(frame, "Smiling", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                st.session_state.mannersism_data.append("Smiling")
            else:
                cv2.putText(frame, "Neutral", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                st.session_state.mannersism_data.append("Neutral")
    else:
        st.session_state.mannersism_data.append("None")

    debug_print("mannersism_data", st.session_state.mannersism_data)  # Debugging


def main():
    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video")
            break

        # Analyze frame
        analyze_eye_gaze(frame)
        analyze_mannersisms(frame)

        # Display analyzed frame
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    # Check if any data has been appended
    if len(st.session_state.eye_gaze_data) > 0 and len(st.session_state.mannersism_data) > 0:
        # Calculate averages and percentages
        eye_gaze_percentage = (st.session_state.eye_gaze_data.count(1) / len(
            [x for x in st.session_state.eye_gaze_data if
             x is not None])) * 100 if st.session_state.eye_gaze_data else 0
        mannersism_smiling_percentage = (st.session_state.mannersism_data.count("Smiling") / len(
            st.session_state.mannersism_data)) * 100 if st.session_state.mannersism_data else 0
        analysis_time = time.time() - st.session_state.start_time
        total_frames_analyzed = len(st.session_state.eye_gaze_data)

        # Debugging final analysis
        debug_print("eye_gaze_percentage", eye_gaze_percentage)
        debug_print("mannersism_smiling_percentage", mannersism_smiling_percentage)
        debug_print("total_frames_analyzed", total_frames_analyzed)

        # Display results
        st.header("Analysis Results")
        st.subheader("Eye Gaze Analysis")
        st.write("Eye Gaze Percentage: {:.2f}%".format(eye_gaze_percentage))
        st.write("Total Frames Analyzed: {}".format(total_frames_analyzed))
        st.subheader("Mannersism Analysis")
        st.write("Smiling Percentage: {:.2f}%".format(mannersism_smiling_percentage))
        st.write("Analysis Time: {:.2f} seconds".format(analysis_time))

        # Save data to CSV
        with open('analysis_data.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Metric", "Value"])
            writer.writerow(["Eye Gaze Percentage", eye_gaze_percentage])
            writer.writerow(["Total Frames Analyzed", total_frames_analyzed])
            writer.writerow(["Smiling Percentage", mannersism_smiling_percentage])
            writer.writerow(["Analysis Time", analysis_time])

        # Download CSV file
        with open('analysis_data.csv', 'rb') as csvfile:
            st.download_button(label="Download Analysis Report", data=csvfile.read(), file_name="analysis_data.csv",
                               mime="text/csv")


if __name__ == "__main__":
    main()
