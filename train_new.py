import opendatasets as od
import os
import tensorflow as tf
from tensorflow.python import keras as keras
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.python.keras.callbacks import LearningRateScheduler
LOG_DIR = 'logs'
BATCH_SIZE = 32
NUM_CLASSES = 20
RESIZE_TO = 224
os.makedirs('./logs')

train_data = tf.keras.preprocessing.image_dataset_from_directory('./oregon-wildlife/oregon_wildlife/oregon_wildlife', labels='inferred',
            color_mode='rgb', batch_size=BATCH_SIZE, image_size=(RESIZE_TO, RESIZE_TO),
            shuffle=True, seed=41, validation_split=0.3, subset='training')
validation_data = tf.keras.preprocessing.image_dataset_from_directory('./oregon-wildlife/oregon_wildlife/oregon_wildlife', labels='inferred',
            color_mode='rgb', batch_size=BATCH_SIZE, image_size=(RESIZE_TO, RESIZE_TO),
            shuffle=True, seed=41, validation_split=0.3, subset='validation')
def input_preprocess(image, label):
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label

train_data = train_data.map(lambda image, label: (tf.image.resize(image,(RESIZE_TO, RESIZE_TO)), label))
validation_data = validation_data.map(lambda image, label: (tf.image.resize(image,(RESIZE_TO, RESIZE_TO)), label))
train_data = train_data.map(input_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)
validation_data = validation_data.map(input_preprocess)

def build_model():
    inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 3))
    base_model = tf.keras.applications.EfficientNetB0(input_tensor=inputs, include_top=False, weights="imagenet")
    base_model.trainable = False
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation=tf.keras.activations.softmax)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

model = build_model()
model.compile(
        optimizer=tf.optimizers.Adam(0.001),
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=[tf.keras.metrics.categorical_accuracy],
    )
model.fit(
    train_data,
    epochs=50,
    validation_data=validation_data,
    callbacks=[
        tf.keras.callbacks.TensorBoard(LOG_DIR),
    ]
  )
