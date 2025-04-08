import gradio as gr
import cv2
import numpy as np
import mediapipe as mp
import os

example_path = os.path.join(os.path.dirname(__file__), 'example')
garm_list = os.listdir(os.path.join(example_path, "cloth"))
garm_list_path = [os.path.join(example_path, "cloth", garm) for garm in garm_list]

human_list = os.listdir(os.path.join(example_path, "cloth"))
human_list_path = [os.path.join(example_path, "cloth", garm) for garm in garm_list]

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
mp_drawing = mp.solutions.drawing_utils
mp_pose_landmark = mp_pose.PoseLandmark


def align_clothing(body_img, clothing_img):
    image_rgb = cv2.cvtColor(body_img, cv2.COLOR_BGR2RGB)
    result = pose.process(image_rgb)
    output = body_img.copy()
    keypoints = {}

    if result.pose_landmarks:
        height, width, _ = output.shape

        # Extract body keypoints
        points = {
            'left_shoulder': mp_pose_landmark.LEFT_SHOULDER,
            'right_shoulder': mp_pose_landmark.RIGHT_SHOULDER,
            'left_hip': mp_pose_landmark.LEFT_HIP
        }

        for name, idx in points.items():
            lm = result.pose_landmarks.landmark[idx]
            keypoints[name] = (int(lm.x * width), int(lm.y * height))

        # Draw for debug
        for name, (x, y) in keypoints.items():
            cv2.circle(output, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(output, name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Affine Transform
        if all(k in keypoints for k in ['left_shoulder', 'right_shoulder', 'left_hip']):
            src_tri = np.array([
                [0, 0],
                [clothing_img.shape[1], 0],
                [0, clothing_img.shape[0]]
            ], dtype=np.float32)

            dst_tri = np.array([
                keypoints['left_shoulder'],
                keypoints['right_shoulder'],
                keypoints['left_hip']
            ], dtype=np.float32)

            # Compute warp matrix and apply it
            warp_mat = cv2.getAffineTransform(src_tri, dst_tri)
            warped_clothing = cv2.warpAffine(clothing_img, warp_mat, (width, height), flags=cv2.INTER_LINEAR,
                                             borderMode=cv2.BORDER_TRANSPARENT)

            # Blend clothing over body
            if clothing_img.shape[2] == 4:  # has alpha
                alpha = warped_clothing[:, :, 3] / 255.0
                for c in range(3):
                    output[:, :, c] = (1 - alpha) * output[:, :, c] + alpha * warped_clothing[:, :, c]
            else:
                output = cv2.addWeighted(output, 0.8, warped_clothing, 0.5, 0)

    return output


image_blocks = gr.Blocks(theme="Nymbo/Alyx_Theme").queue()
with image_blocks as demo:
    gr.HTML("<center><h1>Virtual Try-On</h1></center>")
    gr.HTML("<center><p>Upload an image of a person and an image of a garment ✨</p></center>")
    with gr.Row():
        with gr.Column():
            imgs = gr.Image(type="pil", label='Human', interactive=True)
            example = gr.Examples(
                inputs=imgs,
                examples_per_page=10,
                examples=human_list_path
            )

        with gr.Column():
            garm_img = gr.Image(label="Garment", type="pil",interactive=True)
            example = gr.Examples(
                inputs=garm_img,
                examples_per_page=8,
                examples=garm_list_path)
        with gr.Column():
            image_out = gr.Image(label="Processed image", type="pil")

    with gr.Row():
        try_button = gr.Button(value="Try-on")
image_blocks.launch()
