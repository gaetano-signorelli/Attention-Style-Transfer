import os

from src.architecture.autoencoder.backbones import Backbones

#Backbone
BACKBONE_TYPE = Backbones.VGG19

#Loss weights
WEIGHT_CONTENT = 4
WEIGHT_STYLE = 15
WEIGHT_IDENTITY = 70
WEIGHT_NOISE = 3000

#Optimization
LEARNING_RATE = 1e-4
LR_DECAY = 5e-5

#Training
TRAIN_STEPS = 160000
BATCH_SIZE = 8

IMAGE_RESIZE = (512,512)
IMAGE_CROP = (256,256)

#Saving/Loading
SAVE_MODEL = True
LOAD_MODEL = True
STEPS_BEFORE_SAVE = 10000

#Paths
WEIGHTS_PATH = os.path.join("weights",BACKBONE_TYPE)
DECODER_WEIGHTS_PATH = os.path.join(WEIGHTS_PATH,"decoder_{}.npy")
MCC_WEIGHTS_PATH = os.path.join(WEIGHTS_PATH,"mcc_{}.npy")

CONTENT_TRAIN_PATH = os.path.join("data","coco dataset")
STYLE_TRAIN_PATH = os.path.join("data","wikiart dataset")

VALIDATION_CONTENT_PATH = os.path.join("data","validation images", "content.jpg")
VALIDATION_STYLE_PATH = os.path.join("data","validation images", "style.jpg")
VALIDATION_RESULT_PATH = os.path.join("data","validation result_{}.jpg")
