'''
This script applies the transfer of a style into a content image.
'''

import os
from PIL import Image
import tensorflow as tf
import numpy as np
import argparse

from src.architecture.model.model_handler import ModelHandler
from src.utils.image_processing import load_preprocess_image, interpolate_images
from src.architecture.autoencoder.backbones import Backbones

BACKBONE_TYPE = Backbones.VGG19 #Type of backbone to be used

LOAD_WEIGHTS = True
WEIGHTS_PATH = os.path.join("weights",BACKBONE_TYPE)

def parse_arguments():

    parser = argparse.ArgumentParser(description='Image style transfer')
    parser.add_argument('content_path', type=str, help='Path to the content image')
    parser.add_argument('style_path', type=str, help='Path to the style image')
    parser.add_argument('result_path', type=str, help='Path to the result of image style transfer')
    parser.add_argument('--h', type=int, help='Height of the result', default=512)
    parser.add_argument('--w', type=int, help='Width of the result', default=512)
    parser.add_argument('--mix', type=float, help='Interpolation level between original and stylized content', default=1.0)
    parser.add_argument('--cpu', action="store_true", help='Whether to use CPU instead of GPU')

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    #Get parameters
    args = parse_arguments()

    h = args.h
    w = args.w
    size = (h,w)
    content_path = args.content_path
    style_path = args.style_path
    result_path = args.result_path
    run_on_cpu = args.cpu
    interpolation_level = args.mix

    assert interpolation_level >=0.0 and interpolation_level<=1.0
    assert h % 16 == 0
    assert w % 16 == 0

    input_shape = (size[0],size[1],3)

    #Load model
    model_handler = ModelHandler(BACKBONE_TYPE, input_shape, LOAD_WEIGHTS, WEIGHTS_PATH)
    model_handler.build_model()

    #Preprocess and load content image
    inference_content = load_preprocess_image(content_path,
                                            BACKBONE_TYPE,
                                            image_resize=size)

    #Preprocess and load style image
    inference_style = load_preprocess_image(style_path,
                                            BACKBONE_TYPE,
                                            image_resize=size)

    inference_content = np.array([inference_content])
    inference_style = np.array([inference_style])

    #Create stylized content
    if run_on_cpu:
        with tf.device('/cpu:0'):
            result = model_handler.model((inference_content, inference_style)).numpy()

    else:
        result = model_handler.model((inference_content, inference_style)).numpy()

    result = np.clip(result[0], 0, 255) #Move in range 0-255
    result = interpolate_images(inference_content[0], result, interpolation_level) #Interpolate with content
    result = result.astype(np.uint8) #Cast to int
    result[:,:,[2,0]] = result[:,:,[0,2]] #Convert from BGR to RGB

    #Save result
    image_result = Image.fromarray(result, mode="RGB")

    image_result.save(result_path)

    print("Style transfer completed")
