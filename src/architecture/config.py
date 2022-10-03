from tensorflow.keras.optimizers import Adam

from src.architecture.autoencoder.backbones import Backbones
from src.architecture.model.lr_schedule import AdaptiveLearningRate

BACKBONE_TYPE = Backbones.VGG19

WEIGHT_CONTENT = 4
WEIGHT_STYLE = 15
WEIGHT_IDENTITY = 70
WEIGHT_NOISE = 3000

LEARNING_RATE = 1e-4
LR_DECAY = 5e-5
CURRENT_STEP = 0
adapative_lr = AdaptiveLearningRate(LEARNING_RATE, LR_DECAY, CURRENT_STEP)
OPTIMIZER = Adam(learning_rate=adapative_lr)

TRAIN_STEPS = 160000
BATCH_SIZE = 8
