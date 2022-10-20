import os
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm

from src.architecture.model.model_handler import ModelHandler
from src.utils.image_processing import load_preprocess_image, interpolate_images
from src.architecture.autoencoder.backbones import Backbones

BACKBONE_TYPE = Backbones.VGG19

INTERPOLATION_LEVEL = 1.0

VIDEO_PATH = os.path.join("data","test","video_1.mp4")
STYLE_PATH = os.path.join("data","test","style_1.jpg")
RESULT_PATH = os.path.join("data","test","video result_1_1.mp4")

SIZE = (512,512)

LOAD_WEIGHTS = True

BUFFER_SIZE = 128
BATCH_SIZE = 1

def process_buffer(frames, style, model, batch_size, writer):

    contents = np.array(frames)
    styles = np.array([style]*len(frames))

    input = [contents, styles]

    results = model.predict(input, batch_size=batch_size)

    for i in range(len(frames)):

        result = results[i]
        content = contents[i]

        result = np.clip(result, 0, 255)
        result = interpolate_images(content, result, INTERPOLATION_LEVEL)
        result = result.astype(np.uint8)

        writer.write(result)

    frames.clear()

def stylize_video(video_path, style_path, result_path, buffer_size, frame_shape,
                backbone_type, model, batch_size):

    style = load_preprocess_image(style_path, backbone_type, image_resize=frame_shape)

    frames_buffer = []

    capture = cv2.VideoCapture(video_path)

    n_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(result_path, fourcc, fps, SIZE)

    print("Processing frames...")

    for i in tqdm(range(n_frames+1)):

        success, frame = capture.read()

        if success:
            processed_frame = cv2.resize(frame, frame_shape).astype(float)
            processed_frame = cv2.rotate(processed_frame, cv2.ROTATE_180)
            frames_buffer.append(processed_frame)
            if len(frames_buffer)>=buffer_size:
                process_buffer(frames_buffer, style, model, batch_size, writer)

        else:
            if len(frames_buffer)>0:
                process_buffer(frames_buffer, style, model, batch_size, writer)
            break

    cv2.destroyAllWindows()
    capture.release()
    writer.release()

if __name__ == '__main__':

    input_shape = (SIZE[0],SIZE[1],3)

    model_handler = ModelHandler(BACKBONE_TYPE, input_shape, LOAD_WEIGHTS)
    model_handler.build_model()

    stylized_frames = stylize_video(VIDEO_PATH,
                                    STYLE_PATH,
                                    RESULT_PATH,
                                    BUFFER_SIZE,
                                    SIZE,
                                    BACKBONE_TYPE,
                                    model_handler.model,
                                    BATCH_SIZE)

    print("Style transfer completed")
