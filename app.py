import os

import cv2
import gradio as gr
import mediapipe as mp
import numpy as np
from PIL import Image
from gradio_client import Client, handle_file

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


def align_clothing(body_img, clothing_img):
    image_rgb = cv2.cvtColor(body_img, cv2.COLOR_BGR2RGB)
    result = pose.process(image_rgb)
    output = body_img.copy()

    if result.pose_landmarks:
        h, w, _ = output.shape

        # Extract key points
        def get_point(landmark_id):
            lm = result.pose_landmarks.landmark[landmark_id]
            return int(lm.x * w), int(lm.y * h)

        left_shoulder = get_point(mp_pose_landmark.LEFT_SHOULDER)
        right_shoulder = get_point(mp_pose_landmark.RIGHT_SHOULDER)
        left_hip = get_point(mp_pose_landmark.LEFT_HIP)
        right_hip = get_point(mp_pose_landmark.RIGHT_HIP)

        # Destination box (torso region)
        dst_pts = np.array([
            left_shoulder,
            right_shoulder,
            right_hip,
            left_hip
        ], dtype=np.float32)

        # Source box (clothing image corners)
        src_h, src_w = clothing_img.shape[:2]
        src_pts = np.array([
            [0, 0],
            [src_w, 0],
            [src_w, src_h],
            [0, src_h]
        ], dtype=np.float32)

        # Compute perspective transform and warp
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped_clothing = cv2.warpPerspective(clothing_img, matrix, (w, h), borderMode=cv2.BORDER_TRANSPARENT)

        # Handle transparency
        if clothing_img.shape[2] == 4:
            alpha = warped_clothing[:, :, 3] / 255.0
            for c in range(3):
                output[:, :, c] = (1 - alpha) * output[:, :, c] + alpha * warped_clothing[:, :, c]
        else:
            output = cv2.addWeighted(output, 0.8, warped_clothing, 0.5, 0)

    return output


def process_image(human_img_path, garm_img_path):
    client = Client("franciszzj/Leffa")

    result = client.predict(
        src_image_path=handle_file(human_img_path),
        ref_image_path=handle_file(garm_img_path),
        ref_acceleration=False,
        step=30,
        scale=2.5,
        seed=42,
        vt_model_type="viton_hd",
        vt_garment_type="upper_body",
        vt_repaint=False,
        api_name="/leffa_predict_vt"
    )

    print(result)
    generated_image_path = result[0]
    print("generated_image_path" + generated_image_path)
    generated_image = Image.open(generated_image_path)

    return generated_image


image_blocks = gr.Blocks().queue()
with image_blocks as demo:
    gr.HTML("<center><h1>Virtual Try-On</h1></center>")
    gr.HTML("<center><p>Upload an image of a person and an image of a garment ✨</p></center>")
    with gr.Row():
        with gr.Column():
            human_img = gr.Image(type="filepath", label='Human', interactive=True)
            example = gr.Examples(
                inputs=human_img,
                examples_per_page=10,
                examples=human_list_path
            )

        with gr.Column():
            garm_img = gr.Image(label="Garment", type="filepath", interactive=True)
            example = gr.Examples(
                inputs=garm_img,
                examples_per_page=8,
                examples=garm_list_path)
        with gr.Column():
            image_out = gr.Image(label="Processed image", type="pil")

    with gr.Row():
        try_button = gr.Button(value="Try-on", variant='primary')

    # Linking the button to the processing function
    try_button.click(fn=process_image, inputs=[human_img, garm_img], outputs=image_out)

image_blocks.launch(show_error=True)
