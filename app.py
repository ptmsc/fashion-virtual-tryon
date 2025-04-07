import gradio as gr
import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
mp_drawing = mp.solutions.drawing_utils
mp_pose_landmark = mp_pose.PoseLandmark

def detect_pose(image):
    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run pose detection
    result = pose.process(image_rgb)

    keypoints = {}

    if result.pose_landmarks:
        # Draw landmarks on image
        mp_drawing.draw_landmarks(image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Get image dimensions
        height, width, _ = image.shape

        # Extract specific landmarks
        landmark_indices = {
            'left_shoulder': mp_pose_landmark.LEFT_SHOULDER,
            'right_shoulder': mp_pose_landmark.RIGHT_SHOULDER,
            'left_hip': mp_pose_landmark.LEFT_HIP,
            'right_hip': mp_pose_landmark.RIGHT_HIP
        }

        for name, index in landmark_indices.items():
            lm = result.pose_landmarks.landmark[index]
            x, y = int(lm.x * width), int(lm.y * height)
            keypoints[name] = (x, y)

            # Draw a circle + label for debug
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(image, name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return image, keypoints

# Gradio interface
iface = gr.Interface(
    fn=detect_pose,
    inputs=gr.Image(type="numpy", label="Upload Full-Body Image"),
    outputs=[
        gr.Image(type="numpy", label="Pose Visualization"),
        gr.JSON(label="Extracted Keypoints")
    ],
    title="Virtual Try-On - Pose Detection",
    description="Detects body keypoints using MediaPipe Pose and visualizes them. Shoulders and hips are labeled."
)

if __name__ == "__main__":
    iface.launch()