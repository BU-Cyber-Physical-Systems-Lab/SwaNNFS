import tensorflow as tf
import os

with open("actor.tflite", "wb") as f:
    f.write(tf.lite.TFLiteConverter.from_saved_model("actor").convert())

os.system("xxd -i actor.tflite > actor.h")