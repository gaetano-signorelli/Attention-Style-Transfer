import os
import cv2
from PIL import Image
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import argparse

from src.architecture.model.model_handler import ModelHandler
from src.utils.image_processing import load_preprocess_image, interpolate_images
from src.architecture.autoencoder.backbones import Backbones

BACKBONE_TYPE = Backbones.VGG19

LOAD_WEIGHTS = True
WEIGHTS_PATH = os.path.join("weights",BACKBONE_TYPE)

BUFFER_SIZE = 128
BATCH_SIZE = 1

def parse_arguments():

    parser = argparse.ArgumentParser(description='Video style transfer')
    parser.add_argument('video_path', type=str, help='Path to the content video')
    parser.add_argument('style_path', type=str, help='Path to the style image')
    parser.add_argument('result_path', type=str, help='Path to the result of video style transfer')
    parser.add_argument('--h', type=int, help='Height of the result', default=512)
    parser.add_argument('--w', type=int, help='Width of the result', default=512)
    parser.add_argument('--mix', type=float, help='Interpolation level between original and stylized content', default=1.0)
    parser.add_argument('--cpu', action="store_true", help='Whether to use CPU instead of GPU')

    args = parser.parse_args()

    return args

def process_buffer(frames, style, model, batch_size, writer, run_on_cpu, interpolation_level):

    contents = np.array(frames)

    if BATCH_SIZE==1 or run_on_cpu:
        results = []

    if run_on_cpu:
        with tf.device('/cpu:0'):
            for content in contents:
                input = [np.array([content]), np.array([style])]
                result = model(input).numpy()
                results.append(result[0])

    else:
        if BATCH_SIZE==1:
            for content in contents:
                input = [np.array([content]), np.array([style])]
                result = model(input).numpy()
                results.append(result[0])
        else:
            styles = np.array([style]*len(frames))
            input = [contents, styles]
            results = model.predict(input, batch_size=batch_size)

    for i in range(len(frames)):

        result = results[i]
        content = contents[i]

        result = np.clip(result, 0, 255)
        result = interpolate_images(content, result, interpolation_level)
        result = result.astype(np.uint8)

        writer.write(result)

    frames.clear()

def stylize_video(video_path, style_path, result_path, buffer_size, frame_shape,
                backbone_type, model, batch_size, run_on_cpu, interpolation_level):

    style = load_preprocess_image(style_path, backbone_type, image_resize=frame_shape)

    frames_buffer = []

    capture = cv2.VideoCapture(video_path)

    n_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(result_path, fourcc, fps, (frame_shape[1],frame_shape[0]))

    print("Processing frames...")

    for i in tqdm(range(n_frames+1)):

        success, frame = capture.read()

        if success:
            original_h = frame.shape[0]
            original_w = frame.shape[1]
            processed_frame = cv2.resize(frame, (frame_shape[1],frame_shape[0])).astype(float)
            if original_h > original_w:
                processed_frame = cv2.rotate(processed_frame, cv2.ROTATE_180)
            frames_buffer.append(processed_frame)
            if len(frames_buffer)>=buffer_size:
                process_buffer(frames_buffer, style, model, batch_size, writer, run_on_cpu, interpolation_level)

        else:
            if len(frames_buffer)>0:
                process_buffer(frames_buffer, style, model, batch_size, writer, run_on_cpu, interpolation_level)
            break

    cv2.destroyAllWindows()
    capture.release()
    writer.release()

if __name__ == '__main__':

    args = parse_arguments()

    h = args.h
    w = args.w
    size = (h,w)
    video_path = args.video_path
    style_path = args.style_path
    result_path = args.result_path
    run_on_cpu = args.cpu
    interpolation_level = args.mix

    assert interpolation_level >=0.0 and interpolation_level<=1.0
    assert h % 16 == 0
    assert w % 16 == 0

    input_shape = (size[0],size[1],3)

    model_handler = ModelHandler(BACKBONE_TYPE, input_shape, LOAD_WEIGHTS, WEIGHTS_PATH)
    model_handler.build_model()

    stylized_frames = stylize_video(video_path,
                                    style_path,
                                    result_path,
                                    BUFFER_SIZE,
                                    size,
                                    BACKBONE_TYPE,
                                    model_handler.model,
                                    BATCH_SIZE,
                                    run_on_cpu,
                                    interpolation_level)

    print("Style transfer completed")
