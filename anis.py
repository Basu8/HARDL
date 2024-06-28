import os
import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import gradio as gr

# Constants
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 20
CLASSES_LIST = ['HorseRace', 'VolleyballSpiking', 'Biking', 'TaiChi', 'Punch', 'BreastStroke', 'Billiards', 'PoleVault', 'ThrowDiscus', 'BaseballPitch', 'HorseRiding', 'Mixing', 'HighJump', 'Skijet', 'SkateBoarding', 'MilitaryParade', 'Fencing', 'JugglingBalls', 'Swing', 'RockClimbingIndoor', 'SalsaSpin', 'PlayingTabla', 'Rowing', 'BenchPress', 'PushUps', 'Nunchucks', 'PlayingViolin']

# Load the trained model
model_path = 'model_final.h5'  # Replace with your actual model path
LRCN_model = tf.keras.models.load_model(model_path)

def frames_extraction(video_path):
    frames_list = []
    video_reader = cv2.VideoCapture(video_path)
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = max(int(video_frames_count / SEQUENCE_LENGTH), 1)

    for frame_counter in range(SEQUENCE_LENGTH):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        success, frame = video_reader.read()
        if not success:
            break
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255
        frames_list.append(normalized_frame)

    video_reader.release()
    return frames_list

def predict_on_video(video_file_path ):
    
    output_file_path=video_file_path+'_output.mp4'
    video_reader = cv2.VideoCapture(video_file_path)
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'),
                                   video_reader.get(cv2.CAP_PROP_FPS), (original_video_width, original_video_height))
    frames_queue = deque(maxlen=SEQUENCE_LENGTH)
    predicted_class_name = ''
    
    while video_reader.isOpened():
        ok, frame = video_reader.read()
        if not ok:
            break
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255
        frames_queue.append(normalized_frame)
        if len(frames_queue) == SEQUENCE_LENGTH:
            predicted_labels_probabilities = LRCN_model.predict(np.expand_dims(frames_queue, axis=0))[0]
            predicted_label = np.argmax(predicted_labels_probabilities)
            predicted_class_name = CLASSES_LIST[predicted_label]
        cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        video_writer.write(frame)
    
    video_reader.release()
    video_writer.release()
    return output_file_path , predicted_class_name
    

# if __name__ == '__main__':
#     input_video_file_path = 'v_Basketball_g05_c03.avi'  # Replace with your actual video path
#     #output_video_file_path = 'output_video.mp4'
#     output_video_file_path = 'v_Basketball_g05_c03.avi_output.mp4'
#     predict_on_video(input_video_file_path,  SEQUENCE_LENGTH)
#     print(f'Output saved to {output_video_file_path}')


import gradio as gr


with gr.Blocks() as demo:
    with gr.Row():
        input_video = gr.Video(label="Input Video")
        output_video = gr.Video(label="Output Video")

    output_text = gr.Textbox(label="Output Text")  # Add the Textbox component

    gr.Button("Predict").click(
        fn=predict_on_video,
        inputs=input_video, 
        outputs=[output_video, output_text]  # Pass both outputs
    )

demo.launch()
