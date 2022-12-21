from imutils import paths
imena=list(paths.list_images('imgs'))


import numpy as np
import cv2
import math
import pickle
import gc

from tensorflow.keras import models
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD

from tensorflow.keras import backend as K
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

ms=10000
bs=500

modelk=models.load_model('model18l')
modelk.save('model18l.h5')
del modelk


modelk=models.load_model('model40l')
modelk.save('model40l.h5')
del modelk

modelk=models.load_model('model20l16')
modelk.save('model20l16.h5')
del modelk