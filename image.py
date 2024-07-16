import os
import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import gradio as gr
from PIL import Image
from keras.models import load_model
import matplotlib.pyplot as plt

# Constants
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 20
IMAGE_SIZE = 160 

CLASSES_LIST_IMAGE = ['Sitting', 'Using laptop', 'Hugging', 'Sleeping', 'Drinking', 'Clapping', 'Dancing', 'Cycling', 'Calling', 'Laughing', 'Eating', 'Fighting', 'Listening to music', 'Running', 'Texting']
CLASSES_LIST_VIDEO = ['HorseRace', 'VolleyballSpiking', 'Biking', 'TaiChi', 'Punch', 'BreastStroke', 'Billiards', 'PoleVault', 'ThrowDiscus', 'BaseballPitch', 'HorseRiding', 'Mixing', 'HighJump', 'Skijet', 'SkateBoarding', 'MilitaryParade', 'Fencing', 'JugglingBalls', 'Swing', 'RockClimbingIndoor', 'SalsaSpin', 'PlayingTabla', 'Rowing', 'BenchPress', 'PushUps', 'Nunchucks', 'PlayingViolin']

# Load the model
image_model_path = 'efficientnet_model.h5'
video_model_path = 'model_final.h5'
efficientnet_model = load_model(image_model_path)
LRCN_model = load_model(video_model_path)

def read_img(fn):
    img = Image.open(fn)
    return np.asarray(img.resize((160,160)))

def predict_image_class(test_image):
    result = efficientnet_model.predict(np.asarray([read_img(test_image)]))

    itemindex = np.where(result == np.max(result))
    prediction = itemindex[1][0]
    probability = np.max(result) * 100
    predicted_class = CLASSES_LIST_IMAGE[prediction]

    return probability, predicted_class

def display_prediction(test_image):
    probability, predicted_class = predict_image_class(test_image)

    fig, ax = plt.subplots()
    image = plt.imread(test_image)
    ax.imshow(image)
    ax.set_title(f"Predicted: {predicted_class} ({probability:.2f}%)")
    plt.axis('off')
    return fig

def predict_and_display(test_image):
    probability, predicted_class = predict_image_class(test_image)
    return predicted_class, probability

def predict_on_video(video_file_path, output_file_path, SEQUENCE_LENGTH):
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
            predicted_class_name = CLASSES_LIST_VIDEO[predicted_label]
        cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        video_writer.write(frame)
    video_reader.release()
    video_writer.release()
    return output_file_path, predicted_class_name

def process_image(image_path):
    predicted_class, probability = predict_and_display(image_path)
    fig = display_prediction(image_path)
    return predicted_class, f"{probability:.2f}%", fig

def process_video(video_path):
    output_path = "output_video.mp4"
    video_path, predicted_class = predict_on_video(video_path, output_path, SEQUENCE_LENGTH)
    return predicted_class, output_path, None

# # Gradio interface
# def gradio_predict(image):
#     predicted_class, probability = predict_and_display(image)
#     fig = display_prediction(image)
#     return predicted_class, f"{probability:.2f}%", fig

# interface = gr.Interface(
#     fn=gradio_predict, 
#     inputs=gr.Image(type="filepath"), 
#     outputs=[
#         gr.Textbox(label="Predicted Class"),
#         gr.Textbox(label="Probability"),
#         gr.Plot(label="Image with Prediction")
#     ],
#     title="Image Classification with EfficientNet",
#     description="Upload an image to get the predicted class and probability using EfficientNet."
# )

# interface.launch()

# Gradio interface
def gradio_predict(input_type, file):
    if input_type == "Image":
        #predicted_class, probability, fig = process_image(file)
        return process_image(file) #predicted_class, probability, fig
    elif input_type == "Video":
        #predicted_class, output_path = process_video(file)
        return process_video(file) #predicted_class, output_path

interface = gr.Interface(
    fn=gradio_predict, 
    inputs=[
        gr.Radio(["Image", "Video"], label="Input Type"),
        gr.File(label="Upload File")
    ],
    outputs=[
        gr.Textbox(label="Predicted Class"),
        gr.Textbox(label="Probability/Video Path"),
        gr.Plot(label="Image with Prediction", visible=False),
    ],
    title="Image and Video Classification",
    description="Upload an image or video to get the predicted class using EfficientNet for images and LRCN for videos."
)

interface.launch()

# with gr.Blocks() as demo:
#     with gr.Row():
#         radio = gr.Radio(choices=["Video", "Image"], label="Select Input Type")
    
#     with gr.Row():
#         input_video = gr.Video(label="Input Video", visible=False)
#         input_image = gr.Image(label="Input Image", visible=False)
        
#         radio.change(
#             fn=lambda x: (gr.update(visible=x == "Video"), gr.update(visible=x == "Image")),
#             inputs=radio,
#             outputs=[input_video, input_image]
#         )
    
#     with gr.Row():
#         output_class = gr.Textbox(label="Predicted Class", visible=True)
#         output_prob = gr.Textbox(label="Probability/Video Path", visible=True)
    
#     predict_button = gr.Button("Predict")
    
#     predict_button.click(
#         # fn=lambda input_type, video, image: gradio_predict(input_type, video if input_type == "Video" else image),
#         # inputs=[radio, input_video, input_image],
#         fn=lambda input_type, file: gradio_predict(input_type, file),
#         inputs=[radio, gr.File()],
#         outputs=[output_class, output_prob]
#     )

# demo.launch()