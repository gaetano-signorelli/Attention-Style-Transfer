import os
from PIL import Image
import numpy as np

from src.architecture.model.model_handler import ModelHandler
from src.utils.image_processing import load_preprocess_image
from src.architecture.autoencoder.backbones import Backbones

BACKBONE_TYPE = Backbones.VGG19

CONTENT_PATH = os.path.join("data","test","content_3.jpg")
STYLE_PATH = os.path.join("data","test","style_6.jpg")
RESULT_PATH = os.path.join("data","test","test result_3_6.jpg")

SIZE = (512,512)

LOAD_WEIGHTS = True

if __name__ == '__main__':

    input_shape = (SIZE[0],SIZE[1],3)

    model_handler = ModelHandler(BACKBONE_TYPE, input_shape, LOAD_WEIGHTS)
    model_handler.build_model()

    inference_content = load_preprocess_image(CONTENT_PATH,
                                            BACKBONE_TYPE,
                                            image_resize=SIZE)

    inference_style = load_preprocess_image(STYLE_PATH,
                                            BACKBONE_TYPE,
                                            image_resize=SIZE)

    inference_content = np.array([inference_content])
    inference_style = np.array([inference_style])

    result = model_handler.model((inference_content, inference_style)).numpy()

    result = np.clip(result[0], 0, 1) *255
    result = result.astype(np.uint8)
    result[:,:,[2,0]] = result[:,:,[0,2]]

    image_result = Image.fromarray(result, mode="RGB")

    image_result.save(RESULT_PATH)

    print("Style transfer completed")
