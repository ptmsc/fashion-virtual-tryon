import gradio as gr
import cv2
import numpy as np
import mediapipe as mp
import os

example_path = os.path.join(os.path.dirname(__file__), 'example')

garm_list = os.listdir(os.path.join(example_path, "cloth"))
garm_list_path = [os.path.join(example_path, "cloth", garm) for garm in garm_list]

human_list = os.listdir(os.path.join(example_path, "human"))
human_list_path = [os.path.join(example_path, "human", human) for human in human_list]

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

    return image


def process_image(human_img):
    # Convert PIL image to NumPy array
    human_img = np.array(human_img)

    processed_image = detect_pose(human_img)
    return processed_image


image_blocks = gr.Blocks().queue()
with image_blocks as demo:
    gr.HTML("<center><h1>Virtual Try-On</h1></center>")
    gr.HTML("<center><p>Upload an image of a person and an image of a garment ✨</p></center>")
    with gr.Row():
        with gr.Column():
            human_img = gr.Image(type="pil", label='Human', interactive=True)
            example = gr.Examples(
                inputs=human_img,
                examples_per_page=10,
                examples=human_list_path
            )

        with gr.Column():
            garm_img = gr.Image(label="Garment", type="pil", interactive=True)
            example = gr.Examples(
                inputs=garm_img,
                examples_per_page=8,
                examples=garm_list_path)
        with gr.Column():
            image_out = gr.Image(label="Processed image", type="pil")

    with gr.Row():
        try_button = gr.Button(value="Try-on", variant='primary')

    # Linking the button to the processing function
    try_button.click(fn=process_image, inputs=human_img, outputs=image_out)

image_blocks.launch()
